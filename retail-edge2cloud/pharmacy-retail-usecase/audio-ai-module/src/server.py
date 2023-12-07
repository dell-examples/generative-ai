# Created by Scalers AI for Dell Inc.
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import whisper
from fastapi import FastAPI, File, UploadFile
from helper import (
    OpenVINOAudioEncoder,
    OpenVINOTextDecoder,
    patch_whisper_for_ov_inference,
)
from pydantic import BaseModel

app = FastAPI()

# Loads the model
model = whisper.load_model("base").to("cpu").eval()

WHISPER_ENCODER_OV = Path("whisper-base-ov-model/whisper_encoder.xml")
WHISPER_DECODER_OV = Path("whisper-base-ov-model/whisper_decoder.xml")

# Init OpenVINO runtime
core = ov.Core()

# Replaces base model encoder and decoder with OpenVINO encoder and decoder
patch_whisper_for_ov_inference(model)
model.encoder = OpenVINOAudioEncoder(core, WHISPER_ENCODER_OV, device="AUTO")
model.decoder = OpenVINOTextDecoder(core, WHISPER_DECODER_OV, device="AUTO")


class AudioPathInput(BaseModel):
    audio_path: str


def get_transcription(audio):
    """Transcribe the audio file into text."""
    task = "transcribe"
    start = time.time()
    # Inference
    transcription = model.transcribe(audio, task=task)
    print(time.time() - start)
    return transcription["text"]


@app.post("/transcribe/")
async def transcribe_audio(audio_input: AudioPathInput):
    """API for audio transcription using whisper."""
    print(audio_input.audio_path)
    transcribed_text = get_transcription(audio_input.audio_path)
    return {"transcription": transcribed_text}
