# Created by Scalers AI for Dell Inc.

import time
import fire
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

def infer(
    model_name: str,
):
    input_text = (
        "Discuss the history and evolution of artificial intelligence in 80 words"
    )
    max_new_tokens = 100

    # Initialize and load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OVModelForCausalLM.from_pretrained(model_name)

    # Initialize HF pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors=True,
    )
    # Inference
    start_time = time.time()
    output = text_generator(input_text, max_new_tokens=max_new_tokens)
    _ = tokenizer.decode(output[0]["generated_token_ids"])
    end_time = time.time()

    # Calculate number of tokens generated
    num_tokens = len(output[0]["generated_token_ids"])

    inference_time = end_time - start_time
    token_per_sec = num_tokens / inference_time
    print(f"Inference time: {inference_time} sec")
    print(f"Token per sec: {token_per_sec}")

if __name__ == "__main__":
    fire.Fire(infer)