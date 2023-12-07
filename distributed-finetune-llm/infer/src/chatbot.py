# Created by scalers.ai for Dell
"""
This file reads the config.yml file, and runs the model
with the given arguments as a chatbot on the specified
port.

To run this file, first change the config.yml, then run
python3 chatbot.py
"""

import functools
import os
import sys
import time
from threading import Thread

import gradio as gr
import yaml
from sft2bin import sft2bin_worker
from torch_dtypes import TORCH_DTYPES
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def parse_config(config_file: str) -> dict:
    """
    Parse the yaml config file, and return as a dictionary
        Parameters:
            config_file: The path to the config file
        Returns:
            The YAML file as a dictionary.
    """

    config_yaml = None
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f.read())
    return config_yaml


def validate_streams_yaml(yaml_file):
    """
    Validates the format of a YAML configuration file.

    Parameters:
    yaml_file (str): File path of the YAML configuration.

    Returns:
    bool: True if the YAML format is valid, otherwise False.
    """
    if not os.path.isfile(yaml_file):
        print(f"The file {yaml_file} does not exist.")
        sys.exit(1)

    try:
        with open(yaml_file, "r") as stream:
            data = yaml.safe_load(stream)

        if (
            not isinstance(data, dict)
            or "chatbot_args" not in data
            or "other_args" not in data
        ):
            return False

        chatbot_args = data["chatbot_args"]
        other_args = data["other_args"]
        if not isinstance(chatbot_args, dict) or not isinstance(
            other_args, dict
        ):
            return False

        return True

    except yaml.YAMLError:
        return False


def chat_stream(
    model,
    tokenizer,
    max_new_tokens,
    message,
    history,
):
    # Tokenize input
    encoded_input = tokenizer(message, return_tensors="pt")
    encoded_input = encoded_input.to("cuda")

    start_time = time.time()
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_special_tokens=True, skip_prompt=True
    )
    generate_kwargs = dict(
        **encoded_input, max_new_tokens=max_new_tokens, streamer=streamer
    )

    # Start the TextIteratorStreamer in a different thread.
    gen_thread = Thread(target=model.generate, kwargs=generate_kwargs)
    gen_thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

    gen_thread.join(timeout=1)

    inference_time = time.time() - start_time
    num_tokens = len(
        tokenizer(partial_text, return_tensors="pt")["input_ids"]
        .numpy()
        .flatten()
    )
    token_per_sec = num_tokens / inference_time
    with open("app.log", "w") as file:
        file.write(f"Question: {message}\n")
        file.write(f"Answer: {partial_text}\n")
        file.write(f"Inference latency: {inference_time} sec\n")
        file.write(f"Token per sec: {token_per_sec}\n")
        file.write(f"\n\n\n")


def start_chatbot(
    model_path="meta-llama/Llama-2-7b-hf",
    data_type="torch.bfloat16",
    max_new_tokens=500,
    max_padding_length=4096,
    port=7860,
) -> None:
    # Convert safetensors files to bin files, and get path to bin files
    model_path = sft2bin_worker(model_path)

    # Initialize and load tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            device_map="auto",
            max_length=max_padding_length,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=TORCH_DTYPES[data_type],
        )
    except OSError as exception:
        sys.exit(
            "An error occured when loading the model. Please verify that the inputs in the config.yml file are correct"
        )

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    demo = gr.ChatInterface(
        fn=functools.partial(chat_stream, model, tokenizer, max_new_tokens),
        title="Llama 2 Chatbot",
    )
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    if not validate_streams_yaml("config.yml"):
        sys.exit("config.yml is not a valid YAML file")

    args = parse_config("config.yml")

    # Set HF Token if it is not empty
    if args["other_args"]["hf_token"] != "":
        from huggingface_hub.hf_api import HfFolder

        HfFolder.save_token(args["other_args"]["hf_token"])

    start_chatbot(**(args["chatbot_args"]))
