# Created by scalers.ai for Dell
# Bash script for installing requirements for ONNX Model Conversion

#!/bin/bash
# Install base requirements
pip3 install --no-cache-dir -r custom_requirements.txt

# Install PyTorch nightly build for CPU
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/nightly/cpu

# Download requirements files
wget https://raw.githubusercontent.com/microsoft/onnxruntime/5ff27ef02a1b8d8668c6a9f4da2b7a578f4d9f05/onnxruntime/python/tools/transformers/models/llama/requirements-cpu.txt
wget https://raw.githubusercontent.com/microsoft/onnxruntime/5ff27ef02a1b8d8668c6a9f4da2b7a578f4d9f05/onnxruntime/python/tools/transformers/models/llama/requirements.txt

# Install requirements for CPU
pip3 install --no-cache-dir -r requirements-cpu.txt
