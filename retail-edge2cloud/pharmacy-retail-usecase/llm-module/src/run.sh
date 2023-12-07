# Created by Scalers AI for Dell Inc.

# Saves HF Access token
python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HF_TOKEN')"
# Export the model into OpenVINO format
python3 export_model.py
# Create text embedding from sample invoices and saves it into chroma db.
python3 create_embedding.py
# Starts LLM apis server.
uvicorn chat_server:app --port 8000 --host 0.0.0.0
