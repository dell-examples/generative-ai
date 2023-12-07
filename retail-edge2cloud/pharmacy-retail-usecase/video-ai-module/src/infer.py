# Created by Scalers AI for Dell Inc.
import collections
import concurrent.futures
import functools
import json
import operator
import queue
from typing import Dict, Tuple

import av
import cv2
import numpy as np
import openvino as ov
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics.yolo.utils import ops

app = FastAPI()
frame_queue = queue.Queue()


def do_lines_intersect(line1, line2):
    """Checks intersection between all lines forming the zone and the bbox bottom line."""
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    def orientation(x1, y1, x2, y2, x3, y3):
        val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise

    o1 = orientation(x1, y1, x2, y2, x3, y3)
    o2 = orientation(x1, y1, x2, y2, x4, y4)
    o3 = orientation(x3, y3, x4, y4, x1, y1)
    o4 = orientation(x3, y3, x4, y4, x2, y2)

    if o1 != o2 and o3 != o4:
        return True  # Lines intersect
    return False


def bottom_line_touches_polygon(bottom_line):
    """Check whether cars bounding box bottom line intersects with the defined zones."""
    with open("zone.json", "r") as json_file:
        data = json.load(json_file)

    polygons = []
    for key in data.keys():
        polygon = [(point["x"], point["y"]) for point in data[key]]
        polygons.append(polygon)

    zone_detections = {}  # Create a dictionary to store counts for each zone
    counted_zones = (
        set()
    )  # Create a set to store zones that have already been counted

    for i, polygon in enumerate(polygons, start=1):
        for j in range(len(polygon)):
            k = (j + 1) % len(polygon)
            if do_lines_intersect(bottom_line, [polygon[j], polygon[k]]):
                zone_key = f"zone{i}"
                if zone_key not in counted_zones:
                    if zone_key in zone_detections:
                        zone_detections[zone_key] += 1
                    else:
                        zone_detections[zone_key] = 1
                    counted_zones.add(zone_key)
            else:
                zone_key = f"zone{i}"
                if zone_key not in counted_zones:
                    zone_detections[zone_key] = 0

    return zone_detections, any(
        value == 1 for value in zone_detections.values()
    )


def plot_one_box(
    box: np.ndarray,
    img: np.ndarray,
    frame_num: int,
    color: Tuple[int, int, int] = None,
    mask: np.ndarray = None,
    label: str = None,
    line_thickness: int = 5,
):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
        img (no.ndarray): input image
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness

    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    bottom_line = [(int(box[0]), int(box[3])), (int(box[2]), int(box[3]))]
    count, result = bottom_line_touches_polygon(bottom_line)
    if result:
        color = (
            0,
            255,
            0,
        )  # color or [random.randint(0, 255) for _ in range(3)]
    else:
        color = (255, 165, 0)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img, count


def draw_results(results: Dict, source_image: np.ndarray, frame_num):
    """
    Helper function for drawing bounding boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:

    """
    count = [{"zone1": 0, "zone2": 0}]
    text = "KIOSK 1"
    text2 = "KIOSK 2"
    cv2.putText(
        source_image,
        f"{text}",  # Use the key to access zone counts
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        4,
        3,
    )
    cv2.putText(
        source_image,
        f"{text2}",  # Use the key to access zone counts
        (1220, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        4,
        3,
    )
    boxes = results["det"]
    masks = results.get("segment")
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        if int(lbl) == 2:
            label = f"car {conf:.2f}"
            mask = masks[idx] if masks is not None else None
            source_image, count_ret = plot_one_box(
                xyxy,
                source_image,
                frame_num,
                mask=mask,
                label=label,
                line_thickness=1,
            )
            count.append(count_ret)
    result = dict(
        functools.reduce(operator.add, map(collections.Counter, count))
    )

    if "zone1" in result:
        message1 = "Processing Request"
    else:
        message1 = "Available"
    if "zone2" in result:
        message2 = "Processing Order"
    else:
        message2 = "Available"
    cv2.putText(
        source_image,
        f"{message1}",  # Use the key to access zone counts
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
        3,
    )
    cv2.putText(
        source_image,
        f"{message2}",  # Use the key to access zone counts
        (1220, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
        3,
    )
    return source_image


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
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

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
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

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
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

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


try:
    scale_segments = ops.scale_segments
except AttributeError:
    scale_segments = ops.scale_coords


def postprocess(
    pred_boxes: np.ndarray,
    input_hw: Tuple[int, int],
    orig_img: np.ndarray,
    min_conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.7,
    agnosting_nms: bool = False,
    max_detections: int = 300,
    pred_masks: np.ndarray = None,
    retina_mask: bool = False,
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
        pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
        retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and
                                           segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=80,
        **nms_kwargs,
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        shape = (
            orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        )
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(
                proto[i], pred[:, 6:], pred[:, :4], shape[:2]
            )  # HWC
            segments = [
                scale_segments(input_hw, x, shape, normalize=False)
                for x in ops.masks2segments(masks)
            ]
        else:
            masks = ops.process_mask(
                proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True
            )
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [
                scale_segments(input_hw, x, shape, normalize=False)
                for x in ops.masks2segments(masks)
            ]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results


def detect(image: np.ndarray, model: ov.Model):
    """
    OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
    num_outputs = len(model.outputs)
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    masks = None
    if num_outputs > 1:
        masks = result[model.output(1)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(
        pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks
    )
    return detections


def process_video(video_file, ov_model_path, frame_queue, data):
    """Video processing for inference call and post processing."""
    core = ov.Core()
    ov_model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(ov_model)
    coordinate1 = data["zone1"]
    coordinate2 = data["zone2"]
    input_container = av.open(video_file)
    output_container = av.open(
        "rtsp://rtspsim:8554/mystream", mode="w", format="rtsp"
    )
    in_stream = input_container.streams.video[0]
    out_stream = output_container.add_stream(codec_name="h264", rate=30)
    out_stream.width = 1920
    out_stream.height = 1080
    blue_translucent = (0, 0, 255)
    pts1 = np.array(
        [(point["x"], point["y"]) for point in coordinate1], dtype=np.int32
    )
    pts2 = np.array(
        [(point["x"], point["y"]) for point in coordinate2], dtype=np.int32
    )
    frame_no = 0
    while True:
        for packet in input_container.demux(in_stream):
            for frame in packet.decode():
                input_image = frame.to_ndarray(format="rgb24")
                overlay = input_image.copy()
                cv2.fillPoly(input_image, [pts1], color=blue_translucent)
                cv2.fillPoly(input_image, [pts2], color=blue_translucent)
                cv2.rectangle(
                    input_image, (10, 10), (700, 180), (197, 197, 197), -1
                )
                cv2.rectangle(
                    input_image, (1210, 10), (1900, 180), (197, 197, 197), -1
                )

                alpha = 0.5
                cv2.addWeighted(
                    overlay, alpha, input_image, 1 - alpha, 0, input_image
                )
                detections = detect(input_image, compiled_model)[0]
                image_with_boxes = draw_results(
                    detections, input_image, frame_no
                )

                frame = av.VideoFrame.from_ndarray(
                    image_with_boxes, format="rgb24"
                )

                if output_container:
                    for packet in out_stream.encode(frame):
                        output_container.mux(packet)

        # Seek back to the beginning of the input video to loop
        input_container.seek(0)

    input_container.close()


def stream_optimized(frame_queue):
    """Get the frame queue data and call process video for inference."""
    video_files = ["pharmacy_drivethru.mp4"]  # Add more video files if needed
    with open("zone.json", "r") as json_file:
        data = json.load(json_file)

    ov_model_path = "yolov8n-seg_openvino_model/yolov8n-seg.xml"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_video, video, ov_model_path, frame_queue, data
            )
            for video in video_files
        }

    for future in concurrent.futures.as_completed(futures):
        if future.exception() is not None:
            print(f"Video processing failed: {future.exception()}")


if __name__ == "__main__":
    stream_optimized(frame_queue)
