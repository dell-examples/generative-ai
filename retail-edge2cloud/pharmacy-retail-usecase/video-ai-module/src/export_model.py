# Created by Scalers AI for Dell Inc.
from ultralytics import YOLO

# Loads and export model into OpenVINO format.
model = YOLO("yolov8n-seg.pt")
model.export(format="openvino", dynamic=False, half=False)
