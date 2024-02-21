# Created by Scalers AI for Dell Inc.

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import fire
import av
import time

class CVModelInfer:
    def __init__(
        self,
        model: str,
        video: str,
        video_format: str,
        codec: str,
        video_out: str,
    ):
        self.model = hub.KerasLayer(model, signature="serving_default", signature_outputs_as_dict=True)
        self.video = video
        self.video_format = video_format
        self.video_out = video_out
        self.codec = codec
        self.count = 0
        self.video_capture()

    def video_capture(self):
        self.input_container = av.open(self.video)
        self.output_container = av.open(f'{self.video_out}.{self.video_format}', mode='w')
        
        if self.codec == 'h264':
            codec_lib = 'libx264'
        elif self.codec == 'h265':
            codec_lib = 'libx265'
        else:
            codec_lib = self.codec

        self.in_stream = self.input_container.streams.video[0]
        # get video fps
        self.fps = int(self.in_stream.average_rate)
        self.out_stream = self.output_container.add_stream(codec_lib, self.fps)
        self.out_stream.pix_fmt = self.in_stream.pix_fmt
        self.out_stream.width = self.in_stream.width
        self.out_stream.height = self.in_stream.height

    def preprocess_image(self, img):
        img_arr = np.expand_dims(img, axis=0)
        return img_arr

    def infer(self, frame):
        _ = self.model(frame)
        self.count += 1


def infer_model(
    model: str,
    video: str = "input.mp4",
    video_format: str = "mkv",
    codec: str = "h264",
    video_out: str = "output",
):

    model = CVModelInfer(model,
                         video, 
                         video_format, 
                         codec, 
                         video_out,
                         )
    start_time = time.time()
    for packet in model.input_container.demux(model.in_stream):
        for frame in packet.decode():
            img = frame.to_image()
            preprocesses_img_arr = model.preprocess_image(img)
            model.infer(preprocesses_img_arr)
            if isinstance(frame, av.VideoFrame):
                model.output_container.mux(model.out_stream.encode(frame))
    # Write the frames that are possibly cached:
    for packet in model.out_stream.encode():
        model.output_container.mux(packet)

    model.input_container.close()    
    model.output_container.close()
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time/model.count} sec")

if __name__ == "__main__":
    fire.Fire(infer_model)