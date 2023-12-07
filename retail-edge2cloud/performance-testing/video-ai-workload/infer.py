# Created by Scalers AI for Dell Inc.
import time
from typing import Tuple
import fire
import av
import cv2
import numpy as np
import openvino as ov


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = False,
    stride: int = 32,
):
    """
    Resize image and padding for detection.

    Takes image as input, resizes image to fit into new shape with saving
    original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
    img (np.ndarray): image for preprocessing
    new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
    color (Tuple(int, int, int)): color for filling padded area
    auto (bool): use dynamic input size, only padding for stride constrins applied
    scale_fill (bool): scale image to fill new_shape
    scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
    stride (int): input padding stride
    Returns:
    img (np.ndarray): image after preprocessing
    ratio (Tuple(float, float)): hight and width scaling ratio
    padding_size (Tuple(int, int)): height and width padding size


    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down, do not scale up (for better test mAP)
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.

    Takes image in np.array format, resizes it to specific size using
    letterbox resize and changes data layout from HWC to CHW.

    Parameters:
    img0 (np.ndarray): image for preprocessing
    Returns:
    img (np.ndarray): image after preprocessing
    """
    # resize
    img = letterbox(img0)[0]

    # Convert HWC to CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def image_to_tensor(image: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.

    Takes image in np.array format, resizes it to specific size using
    letterbox resize and changes data layout from HWC to CHW.

    Parameters:
    img (np.ndarray): image for preprocessing
    Returns:
    input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def infer(compiled_model, image: np.ndarray):
    """
    OpenVINO YOLOv8 model inference function.

    Preprocess image, runs model inference.
    Parameters:
        image (np.ndarray): input image.
    """
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    _ = compiled_model(input_tensor)


def video_capture(video, video_out):
    """Initialize video decode and encode using PyAV."""
    input_container = av.open(video)
    output_container = av.open(video_out, mode="w")

    codec_lib = "libx264"

    in_stream = input_container.streams.video[0]
    # get video fps
    fps = int(in_stream.average_rate)
    out_stream = output_container.add_stream(codec_lib, fps)
    out_stream.pix_fmt = in_stream.pix_fmt
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    return input_container, in_stream, output_container, out_stream

def run_infer(
    model_path: str,
    device: str,
    video: str,
):
    # Loads the model into CPU
    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model, device)

    # Video capture
    video_out = "output.mp4"
    input_container, in_stream, output_container, out_stream = video_capture(
        video, video_out
    )

    start_time = time.time()
    frame_count = 0
    # Starts reading frames from video file
    for packet in input_container.demux(in_stream):
        for frame in packet.decode():
            frame_count += 1
            img_arr = frame.to_ndarray(format="rgb24")
            # Inference
            infer(compiled_model, img_arr)
            if isinstance(frame, av.VideoFrame):
                output_container.mux(out_stream.encode(frame))
    end_time = time.time()

    # Write the frames that are possibly cached:
    for packet in out_stream.encode():
        output_container.mux(packet)

    time_taken = end_time - start_time
    inference_time = time_taken / frame_count
    print(f"Inference time per frame: {inference_time}")


if __name__ == "__main__":
    fire.Fire(run_infer)