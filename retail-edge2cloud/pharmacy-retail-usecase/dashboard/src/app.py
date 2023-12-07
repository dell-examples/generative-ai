# Created by Scalers AI for Dell Inc.
import json
import time
import os
import gradio as gr
import requests


class UI:
    """Class to start UI and perform postprocessing tasks."""

    def __init__(self):
        """Initialize the class variables."""
        self.llama_endpoint = "http://llama:8000/generate"
        self.whisper_endpoint = "http://whisper:8080/transcribe"
        self.host_ip = os.environ['HOST_IP']
        self.demo = self.load_ui()

    def get_transcription(self, audio):
        """Get transcription from whisper model."""
        data = {"audio_path": audio}
        json_data = json.dumps(data)
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.whisper_endpoint, data=json_data, headers=headers
        )
        result = response.json()
        return result.get("transcription")

    def add_text(self, history, text):
        """Add chat history and user input to ui"""
        history = history + [(text, None)]
        return history, gr.Textbox(value="", interactive=False)

    def bot(self, history):
        """Inference call."""
        endpoint = self.llama_endpoint
        params = {"query": history[-1][0]}
        response = requests.get(url=endpoint, params=params)

        if response.status_code == 200:
            response_data = response.json()
            if response_data.strip() == "":
                return "Please provide more details."

            history[-1][1] = ""
            for character in response_data:
                history[-1][1] += character
                time.sleep(0.01)
                yield history
        elif response.status_code >= 400:
            if response.status_code < 500:
                error_message = f"Client error: {response.status_code}"
            else:
                error_message = (
                    f"Server Error: {response.status_code}. "
                    "Might be an issue with downloading/loading the model. "
                    "Check server logs for details."
                )
            raise gr.Error(error_message)

    def load_ui(self):
        """UI components."""
        with gr.Blocks(css="app.css") as demo:
            with gr.Tab("Pharmacy Drive-Thru Dashboard", elem_id="tab"):
                with gr.Row():
                    with gr.Column():
                        video = gr.HTML(
                            f"<iframe src='http://{self.host_ip}:8888/mystream/' width='725' height='360' frameborder='0' allowfullscreen></iframe>"
                        )

                    with gr.Column():
                        video = gr.HTML(
                            f"<iframe src='http://{self.host_ip}:8888/mystream/' width='725' height='360' frameborder='0' allowfullscreen></iframe>"
                        )

                with gr.Box(elem_id="box"):
                    chatbot = gr.Chatbot(
                        label="Chat Window",
                        avatar_images=(
                            "images/user_logo.png",
                            "images/scalers_logo.png",
                        ),
                        elem_id="chatbot",
                    )
                    with gr.Row():
                        with gr.Column(scale=9):
                            msg = gr.Textbox(
                                label="Llama2 Prompt", elem_id="textbox"
                            )
                        with gr.Column(scale=3):
                            audio = gr.Audio(
                                source="microphone",
                                type="filepath",
                                elem_id="audio",
                            )
                        with gr.Column(scale=1):
                            go = gr.Button("", elem_id="send")

                audio.stop_recording(self.get_transcription, audio, msg)
                go.click(
                    self.add_text, [chatbot, msg], [chatbot, msg], queue=False
                ).then(self.bot, chatbot, chatbot)
        return demo


if __name__ == "__main__":
    ui = UI()
    ui.demo.queue().launch(server_name="0.0.0.0")
