# Created by scalers.ai for Dell
"""
This script builds a Gradio ChatInterface demo for interacting with a
conversational model deployed as a REST API endpoint.
"""

import argparse
import json

import gradio as gr
import requests


def send_request(message, history, tokens, temperature):
    """Send a request to the model and yield partial generated messages.

    :param message: The input message.
    :type message: str
    :param history: The conversation history.
    :type history: list
    :param tokens: The maximum number of tokens for the response.
    :type tokens: str
    :param temperature: The temperature for response generation.
    :type temperature: str
    :yields: Partially generated messages.
    """
    input_prompt = message
    prompt = f"system_prompt: You are a teacher. prompt:{input_prompt}"

    history_transformer_format = history + [[prompt, ""]]

    messages = "".join(
        [
            "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
            for item in history_transformer_format
        ]
    )

    if "gpu" in args.url.lower():
        pload = {
            "prompt": messages,
            "stream": True,
            "max_tokens": int(tokens),
            "temperature": float(temperature),
        }
    else:
        pload = {"prompt": prompt}

    response = requests.post(args.url, json=pload, stream=True)

    partial_message = ""
    for chunk in response.iter_content(
        chunk_size=None,
        decode_unicode=False,
    ):
        if chunk:
            try:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"]
                partial_message = partial_message + output
                yield partial_message
            except json.JSONDecodeError:
                data = chunk.decode("utf-8")
                output = data
                partial_message = partial_message + output
                yield partial_message


def build_demo():
    """Build and return a Gradio ChatInterface demo.

    :returns: Gradio ChatInterface demo.
    :rtype: gr.Blocks
    """
    with gr.Blocks() as demo:
        tokens = gr.Textbox(label="Max Tokens", value=128, placeholder="0")
        temperature = gr.Textbox(
            label="Temperature", value=0.5, placeholder="temp"
        )
        gr.ChatInterface(
            send_request,
            additional_inputs=[tokens, temperature],
            concurrency_limit=100,
        )
    return demo


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add command-line arguments
    parser.add_argument(
        "--host", type=str, default="localhost"
    )  # Hostname argument
    parser.add_argument(
        "--port", type=int, default=7860
    )  # Port number argument
    parser.add_argument(
        "--url", type=str, default="http://localhost:30800/gpu"
    )  # URL argument

    # Parse the arguments
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the demo
    demo = build_demo()

    # Launch the demo server
    demo.launch(server_name=args.host, server_port=args.port)
