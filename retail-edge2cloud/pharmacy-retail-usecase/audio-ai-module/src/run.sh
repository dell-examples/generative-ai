# Created by Scalers AI for Dell Inc.

# Convert the model into OpenVINO format
python3 export_model.py
# Starts Whisper server
uvicorn server:app --port 8080 --host 0.0.0.0
