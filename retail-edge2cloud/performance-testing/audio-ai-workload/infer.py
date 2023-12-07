# Created by Scalers AI for Dell Inc.
import time
import fire
import openvino as ov
import whisper
from helper import (
    OpenVINOAudioEncoder,
    OpenVINOTextDecoder,
    patch_whisper_for_ov_inference,
)

def infer(
    model_name: str,
    ovmodel_path: str,
    device: str,
    audio: str,
):
    # Loads the model
    model = whisper.load_model(model_name)

    WHISPER_ENCODER_OV = f"{ovmodel_path}/whisper_encoder.xml"
    WHISPER_DECODER_OV = f"{ovmodel_path}/whisper_decoder.xml"

    core = ov.Core()

    # Replaces base model encoder and decoder with OpenVINO encoder and decoder
    patch_whisper_for_ov_inference(model)
    model.encoder = OpenVINOAudioEncoder(core, WHISPER_ENCODER_OV, device=device)
    model.decoder = OpenVINOTextDecoder(core, WHISPER_DECODER_OV, device=device)

    start_time = time.time()
    # Transcribe the audio file into text
    _ = model.transcribe(audio=audio, task="transcribe")
    end_time = time.time()

    print(f"Inference time: {end_time-start_time}")

if __name__ == "__main__":
    fire.Fire(infer)