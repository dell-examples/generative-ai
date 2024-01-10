# Created by scalers.ai for Dell Inc
echo "Creating embedding from docs folder ..."
python3 create_embedding.py -c docs
echo "Created embeddings. Starting model server..."
uvicorn server:app --host 0.0.0.0 --port 8000
