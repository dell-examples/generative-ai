# Created by scalers.ai for Dell
"""
Llama 2 7B model full fine-tuning.
The script used huggingface accelerate with deepspeed for the fine-tuning.

The script is modified version of doc/source/templates/04_finetuning_llms_with_deepspeed/finetune_hf_llm.py
from https://github.com/ray-project/ray/tree/master repo.
"""

import argparse
import functools
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import huggingface_hub
import pandas as pd
import ray
import ray.util.scheduling_strategies
import torch
import torch.nn as nn
import tqdm
import tree
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from ray import train
from ray.train.torch import TorchTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def collate_fn(
    batch: dict, tokenizer: AutoTokenizer, block_size: int, device: str
) -> dict:
    """Tokenizes and load the tensors to input device.

    :param batch: Dictionary containing a list of text inputs under the key 'input'.
    :type batch: dict
    :param tokenizer: Tokenizer object.
    :type tokenizer: AutoTokenizer
    :param block_size: The maximum length of tokenized sequences after padding/truncation.
    :type block_size: int
    :param device: The device on which the tensors should be placed (e.g., 'cuda', 'cpu').
    :type device: str

    :returns: Tokenized and preprocessed batch of text data.
    """
    out_batch = tokenizer(
        list(batch["input"]),
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_tensors="pt",
    )
    out_batch["labels"] = out_batch["input_ids"].clone()

    out_batch = tree.map_structure(lambda x: x.to(device), out_batch)

    return out_batch


def get_tokenizer(model_name: str, special_tokens: List[str]):
    """Creates and configures a tokenizer for a specific language model.

    :param model_name: Name of the pre-trained language model.
    :type model_name: str
    :param special_tokens: List of special tokens to be added to the tokenizer.
    :type special_tokens: List[str]

    :returns: Configured tokenizer for the specified language model.
    :rtype: AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    return tokenizer


def create_ray_dataset(path: str):
    """Creates a Ray DataFrame from a JSONL file containing input data.

    :param path: Path to the JSONL file.
    :type path: str

    :returns: Ray DataFrame containing the input data.
    """
    with open(path, "r") as json_file:
        items = [json.loads(x) for x in json_file]

    dataset = {"input": []}
    for item in items:
        assert set(item.keys()) == {"input"}
        dataset["input"].append(item["input"])

    df = pd.DataFrame.from_dict(dataset)

    return ray.data.from_pandas(df)


def evaluate(
    *, model, eval_ds, accelerator, bsize, ds_kwargs
) -> Tuple[float, float]:
    """Run evaluation on the model using accelerate.

    :param model: The PyTorch model.
    :param eval_ds: Configuration for the evaluation dataset.
    :param bsize: Batch size
    :ds_kwargs: Datase keyword arguments
    :param accelerator: The accelerator for distributed training.

    :returns: Tuple containing perplexity and evaluation loss.
    :rtype: Tuple[float, float]
    """
    model.eval()
    losses = []

    eval_dataloader = eval_ds.iter_torch_batches(batch_size=bsize, **ds_kwargs)
    eval_ds_len = len(list(eval_ds.iter_batches(batch_size=1)))
    for _, batch in tqdm.tqdm(
        enumerate(eval_dataloader), total=eval_ds_len // (bsize + 1)
    ):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        # The tensors are gathered by concatenating them on the first dimension, so we
        # add a new dimension to the scalar loss to get a tensor of shape (K,) for K
        # workers.
        losses.append(accelerator.gather(loss[None]))

    # We stack losses so that we have a tensor of shape (T, K) where T is the number of
    # steps and K is the number of workers.
    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return perplexity, eval_loss


def checkpoint_model(
    checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs
):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again.
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    # In here model will be a DeepspeedEngine object
    model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    print(status_msg)


def training_function(kwargs: dict):
    """Train LLM model using Accelerate, DeepSpeed."""
    print("training_function called")

    # Train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
    # properly on multi-gpu nodes
    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = cuda_visible_device[local_rank]
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"

    config = kwargs["config"]
    args = argparse.Namespace(**kwargs["args"])
    special_tokens = kwargs.get("special_tokens", [])

    # sign in to huggingface hub
    huggingface_hub.login(token=args.hf_token)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["learning_rate"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])

    # Get deepspeed config to setup the batch size per device
    ds_plugin = config["ds_plugin"]
    ds_plugin.hf_ds_config.config[
        "train_micro_batch_size_per_gpu"
    ] = batch_size

    # Initialize accelerator
    accelerator = Accelerator(
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=args.mx,
    )

    set_seed(seed)

    # train_ds is the local shard for this model
    train_ds = train.get_dataset_shard("train")
    valid_ds = train.get_dataset_shard("valid")

    train_ds_len = len(list(train_ds.iter_batches(batch_size=1)))

    tokenizer = get_tokenizer(
        model_name=args.model_name, special_tokens=special_tokens
    )
    collate_partial = functools.partial(
        collate_fn,
        tokenizer=tokenizer,
        block_size=config["block_size"],
        device=accelerator.device,
    )

    # Get the trial directory from Ray Train
    # This will be local to every node (and will get synced to remote storage if
    # provided.)
    ckpt_path = tempfile.mkdtemp(dir=config["output_dir"])

    s = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # `use_cache=True` is incompatible with gradient checkpointing.
        use_cache=False,
    )
    print(f"Done loading model in {time.time() - s} seconds.")

    model.resize_token_embeddings(len(tokenizer))

    print("Model initialized with pretrained weights. Training starting...")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer"
        not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )

    # Instantiate scheduler
    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else
    # creates `args.lr_scheduler_type` Scheduler
    # get train and valid dataset lengths

    num_warmup_steps = 10
    num_steps_per_epoch = math.ceil(train_ds_len / args.batch_size_per_device)
    total_training_steps = (
        num_steps_per_epoch * num_epochs // gradient_accumulation_steps
    )

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler"
        not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * args.num_devices,
            num_training_steps=total_training_steps * args.num_devices,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=num_warmup_steps * args.num_devices,
            total_num_steps=total_training_steps * args.num_devices,
        )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the
    # same order we gave them to the prepare method.
    s = time.time()
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    print(f"Prepare done in {time.time() - s} seconds.")

    # Now we train the model
    if accelerator.is_main_process:
        print("Starting training ...")
        print("Number of batches on main process", train_ds_len // batch_size)

    for epoch in range(num_epochs):
        fwd_time_sum, bwd_time_sum, optim_step_time_sum = 0, 0, 0
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)

        train_dataloader = train_ds.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=collate_partial,
        )

        for step, batch in tqdm.tqdm(
            enumerate(train_dataloader), total=train_ds_len // batch_size + 1
        ):
            # We could avoid this line since we set the accelerator with
            # `device_placement=True`.
            with accelerator.accumulate(model):
                s_fwd = time.time()
                outputs = model(**batch)
                loss = outputs.loss
                loss_sum += loss
                e_fwd = time.time()
                fwd_time = e_fwd - s_fwd
                fwd_time_sum += fwd_time
                s_bwd = time.time()
                accelerator.backward(loss)
                e_bwd = time.time()
                bwd_time = e_bwd - s_bwd
                bwd_time_sum += bwd_time

                s_opt_step = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                e_opt_step = time.time()
                optim_step_time_sum += e_opt_step - s_opt_step

            if accelerator.is_main_process:
                accelerator.print(
                    f"[epoch {epoch} step {step}] "
                    f"loss: {loss.item()} step-time: {e_opt_step - s_fwd}"
                )

            aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()

            # as long as this is not the last step report here
            if step != (train_ds_len // batch_size - 1):
                train.report(
                    {
                        "epoch": epoch,
                        "iteration": step,
                        "train_loss_batch": aggregated_loss,
                        "avg_train_loss_epoch": None,
                        "eval_loss": None,
                        "perplexity": None,
                        "num_iterations": step + 1,
                        "train_time_per_epoch": None,
                        "eval_time_per_epoch": None,
                        "fwd_time": fwd_time,
                        "bwd_time": bwd_time,
                        "avg_fwd_time_per_epoch": None,
                        "avg_bwd_time_per_epoch": None,
                        "learning_rate": lr_scheduler.get_lr()[0],
                    }
                )

        e_epoch = time.time()
        accelerator.print("Train time per epoch: ", e_epoch - s_epoch)

        eval_s_epoch = time.time()
        print("Running evaluation ...")
        perplex, eloss = evaluate(
            model=model,
            eval_ds=valid_ds,
            accelerator=accelerator,
            bsize=config["eval_batch_size"],
            ds_kwargs={"collate_fn": collate_partial},
        )
        accelerator.print("Eval result loss", eloss)
        accelerator.print("Eval perplex", perplex)

        eval_e_epoch = time.time()
        accelerator.print("Eval time per epoch: ", eval_e_epoch - eval_s_epoch)
        accelerator.print("avg fwd time: ", fwd_time_sum / (step + 1))
        accelerator.print("avg bwd time: ", bwd_time_sum / (step + 1))
        accelerator.print(
            "avg opt step time: ", optim_step_time_sum / (step + 1)
        )

        accelerator.print(f"Saving the model in {ckpt_path}")
        accelerator.wait_for_everyone()
        with accelerator.main_process_first():
            ckpt_path_epoch = Path(ckpt_path) / f"epoch-{epoch}"
            ckpt_path_epoch.mkdir(parents=True, exist_ok=True)

        if accelerator.is_main_process:
            print("Saving tokenizer and config.")
            tokenizer.save_pretrained(ckpt_path_epoch)

        accelerator.wait_for_everyone()

        checkpointing_time_s = time.time()
        # This checkpointing method makes deepspeed checkpoints on each node and then
        # Ray Train will aggregate them to a central s3 bucket.
        # It should be done on all processes (not just the Rank 0)
        # checkpoint_model(
        #     checkpoint_folder=ckpt_path_epoch,
        #     ckpt_id=epoch,
        #     model=model,
        #     epoch=epoch,
        #     last_global_step=step
        # )

        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            ckpt_path_epoch,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True,
            state_dict=accelerator.get_state_dict(model),
        )

        accelerator.wait_for_everyone()
        # Note: After the following call, in the case of remote storage, the checkpoint
        # directoy will get synced to the remote storage and then deleted from the
        # local directory. This will open up local disk.
        metrics = {
            "epoch": epoch,
            "iteration": step,
            "train_loss_batch": aggregated_loss,
            "avg_train_loss_epoch": loss_sum.item() / (step + 1),
            "eval_loss": eloss,
            "perplexity": perplex,
            "num_iterations": step + 1,
            "train_time_per_epoch": e_epoch - s_epoch,
            "eval_time_per_epoch": eval_e_epoch - eval_s_epoch,
            "fwd_time": fwd_time,
            "bwd_time": bwd_time,
            "avg_fwd_time_per_epoch": fwd_time_sum / (step + 1),
            "avg_bwd_time_per_epoch": bwd_time_sum / (step + 1),
            "learning_rate": lr_scheduler.get_lr()[0],
        }

        train.report(
            metrics,
            # We do not need to explictly call report(checkpoint).
            # This is because the checkpointing is not on all distributed workers, it's
            # only done on rank_0 which is forced to be co-located with the trainer
            # object. By default the files created by trainer will get synced which
            # will include the checkpoint files created by the Rank_0.
            # Note that this will not delete the checkpoints from the previous
            # iterations.
            checkpoint=train.Checkpoint.from_directory(ckpt_path_epoch),
        )
        print("Checkpointing time: ", time.time() - checkpointing_time_s)

        if perplex < args.stop_perplexity:
            print(
                f"Perplexity reached {perplex} < {args.stop_perplexity}. Stopping."
            )
            break


def parse_args():
    """Arg parser for the fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Llama 2 7B Fine-Tuning Script."
    )
    parser.add_argument(
        "--mx",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--batch-size-per-device",
        "-bs",
        type=int,
        default=16,
        help="Batch size to use per device.",
    )
    parser.add_argument(
        "--stop-perplexity",
        default=0,
        type=float,
        help="Target perplexity to reach after which to stop training. Default is 0. If 0, training will not stop on perplexity.",
    )
    parser.add_argument(
        "--eval-batch-size-per-device",
        "-ebs",
        type=int,
        default=2,
        help="Batch size to use per device (For evaluation).",
    )
    parser.add_argument(
        "--num-devices",
        "-nd",
        type=int,
        default=4,
        help="Number of devices to use.",
    )
    parser.add_argument(
        "--grad-accum",
        "-ga",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="./data/train.jsonl",
        help="Path to dataset train jsonl file.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="./data/test.jsonl",
        help="Path to datset test jsonl file.",
    )
    parser.add_argument(
        "--special-token-path",
        type=str,
        required=False,
        help="Path to dataset special token json file.",
    )
    parser.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="If passed, will not use gradient checkpointing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--model-name",
        "-mn",
        default="meta-llama/Llama-2-7b-hf",
        type=str,
    )
    parser.add_argument(
        "--num-epochs",
        "-ne",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--num-checkpoints-to-keep",
        type=int,
        help=(
            "Number of checkpoints to keep, if None, all checkpoints will be kept, "
            "if set to n>=1, the top n checkpoint with min. evaluation perplexity "
            "will be kept."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--ctx-len", type=int, default=512, help="Input sequence length."
    )
    parser.add_argument(
        "--ds-config",
        type=str,
        default="./ds_7b_13b.json",
        required=True,
        help="Deepspeed config json to use.",
    )
    parser.add_argument(
        "--hf-token",
        "-hft",
        required=True,
        type=str,
        help="Hugginface hub account token.",
    )

    args = parser.parse_args()

    return args


def main():
    """Main method for Fine-Tuning Llama 2 LLM Models"""
    args = parse_args()

    if not args.output_dir:
        raise ValueError("--output_dir must be specified")

    # update the config with args so that we have access to them.
    config = vars(args)
    config.update(
        **{
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "seed": 42,
            "batch_size": args.batch_size_per_device,
            "gradient_accumulation_steps": args.grad_accum,
            "model_name": args.model_name,
            "block_size": args.ctx_len,
            "eval_batch_size": args.eval_batch_size_per_device,
        }
    )

    # Add deepspeed plugin to the config
    ds_plugin = DeepSpeedPlugin(hf_ds_config=config.get("ds_config"))
    config.update(ds_plugin=ds_plugin)

    os.environ["TUNE_RESULT_DIR"] = args.output_dir

    ray.init(
        runtime_env={
            "env_vars": {
                "TUNE_RESULT_DIR": os.environ["TUNE_RESULT_DIR"],
            },
        }
    )

    train_ds = create_ray_dataset(args.train_path)
    if args.test_path is not None:
        valid_ds = create_ray_dataset(args.test_path)
    else:
        valid_ds = None

    # json file
    if args.special_token_path:
        try:
            with open(args.special_token_path, "r") as json_file:
                special_tokens = json.load(json_file)["tokens"]
        except (FileNotFoundError, TypeError):
            special_tokens = []
    else:
        special_tokens = []

    trial_name = f"{args.model_name}".split("/")[-1]

    trainer = TorchTrainer(
        training_function,
        train_loop_config={
            "config": config,
            "args": vars(args),
            "special_tokens": special_tokens,
        },
        run_config=train.RunConfig(
            name=trial_name,
        ),
        scaling_config=train.ScalingConfig(
            num_workers=args.num_devices,
            use_gpu=True,
        ),
        datasets={
            "train": train_ds,
            "valid": valid_ds,
        },
        dataset_config=ray.train.DataConfig(
            datasets_to_split=["train", "valid"],
        ),
    )

    results = trainer.fit()
    best_checkpoint = results.best_checkpoints[0]

    print("Results are stored in:")
    print(results.path)
    print("Best checkpoint is stored in:")
    print(best_checkpoint[0])
    print(f"With perplexity: {best_checkpoint[1]['perplexity']}")


if __name__ == "__main__":
    main()
