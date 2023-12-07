# Edge v/s Cloud Performance Testing

## Introduction
This technical guide will provide an overview of a performance testing comparison between Dell PowerEdge systems and AWS EC2 instances for three specific areas: video AI, audio AI, and large language models (LLMs). The goal of this testing is to provide insights into the performance and cost-effectiveness of each platform for these specific workloads.

Scalers AI™ selected the popular and modern industry leading models for performance testing. Llama 2 7B Chat model for LLM workload, YOLOv8 nano segmentation model for video AI workload and Whisper base model for audio AI workload are being used for performance testing.

## Getting Started
### Pre-requisites
Before starting the performance testing process, ensure you have the following prerequisites installed:

- Ubuntu 22.04
- Python 3.10
- [Docker Engine v23 or latest](https://docs.docker.com/engine/install/ubuntu/#installation-methods)
- [Nvidia CUDA Toolkit v12.2.1 or latest](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-toolkit-release-notes/index.html)
- [Nvidia Container Toolkit v1.14.0 or latest](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.0/install-guide.html)

### System Setup
- Install numactl:
    ```sh
    sudo apt install numactl
    ```
- Install FFMPEG:
    ```sh
    sudo apt install ffmpeg
    ```
- Install OpenGL library:
    ```sh
    sudo apt-get install libgl1
    ```
- For API load testing, install [Apache Bench](https://httpd.apache.org/docs/2.4/programs/ab.html):
    ```sh
    sudo apt install apache2-utils
    ```

## CPU Performance Testing
This section provides instructions for performance testing the OpenVINO based workload on CPU. The original models are first exported into OpenVINO supported format and then used for inference.
The CPU performance testing includes three AI workloads:
- [LLM Workload: Llama 2](#llm-workload--llama-2)
- [Video AI Workload : YOLOv8](#video-ai-workload--yolov8)
- [Audio AI Workload : Whisper](#audio-ai-workload--whisper)

The performance testing is executed in below Systems. For details refer [SUT](./SUT.pdf)

### Dell PowerEdge Servers

| System         | CPU                   | Cores per Socket | Sockets | vCPUs |
|------------------------|-----------------------|------------------|---------|-------|
| Dell PowerEdge XR4520c | Intel(R) Xeon(R) D-2776NT (3rd Gen) | 16               | 1       | 32   |
| Dell PowerEdge R760xa  | Intel(R) Xeon(R) Platinum 8480+ (4th Gen) | 56               | 2       | 224   |

### AWS Instances

Here are the comparable AWS EC2 instances for Dell PowerEdge XR4520c and R760xa respectively:

| Instance Type          | CPU                   | Cores per Socket | Sockets | vCPUs | Pricing (On demand Linux pricing per hour) |
|------------------------|-----------------------|------------------|---------|-------| -------|
| m6i.8xlarge  | Intel(R) Xeon(R) Platinum 8375C  (3rd Gen) | 16               | 1       | 32   | 1.536  USD |
| m7i.48xlarge | Intel Xeon Platinum 8488C (4th Gen) | 48               | 2       | 192   | 9.6768 USD  |


## LLM Workload : Llama 2
This section provides instructions for performance testing the [Llama-2 7B Chat model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). The original model is exported into OpenVINO format and then used for text generation (inference) using the [Optimum](https://github.com/huggingface/optimum) & [Transformers](https://github.com/huggingface/transformers) library.

The `llm-workload` directory contains python script to export the model and run inference using Llama 2 model.

### Python Dependencies
Install the Python packages:

```
pip3 install -r requirements.txt
```
**Note**: Install packages in a python virtual environment to avoid conflict with existing packages.

### Llama-2 Model Access
To access the Llama-2 7B Chat Model from Hugging Face:

1. Request access the Llama-2 7B Chat Model: [Llama-2-7B-Chat-HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Log in to Hugging Face CLI and enter your access token when prompted:
    ```sh
    huggingface-cli login
    ```
### Export Model
Convert Llama-2 7B chat huggingface model into OpenVINO IR format using Optimum.
The reference code snippet is provided below for exporting the model.
```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Export HF Llama 2 7b Chat model into OpenVINO format.
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id, export=True)

# Saves exported model and tokenizer into a local folder.
model.save_pretrained("llama-2-7b-chat-ov")
tokenizer.save_pretrained("llama-2-7b-chat-ov")
```
### Inference
Generate text using Llama 2 7B chat model and prints the inference time and throughput(token per sec).

> NOTE: Update the parameters according with your values while running infer.py

```
python3 infer.py --model_name="llama-2-7b-chat-ov"
```

## Video AI Workload : YOLOv8
This section provides instructions for performance testing the YOLOv8 nano segmentation([YOLOv8n-seg](https://docs.ultralytics.com/tasks/segment/)) model. The original model is exported into OpenVINO format using ultralytics library and then used for object segmentation(inference) using OpenVINO runtime.

The `video-ai-workload` directory contains python script to export the model and run inference using YOLOv8 model.

### Python Dependencies
Install the Python packages:

```
pip3 install -r requirements.txt
```
**Note**: Install packages in a python virtual environment to avoid conflict with existing packages.

### Export Model
Convert YOLOv8 nano segmentation model into OpenVINO IR format using ultralytics library. The reference code snippet is provided below for exporting the model.
```python
from ultralytics import YOLO

# Loads and export model into OpenVINO format.
model = YOLO("yolov8n-seg.pt")
model.export(format="openvino", dynamic=False, half=False)
```

### Inference
Decodes the video frames using PyAV, runs segmentation on video frames, encode the video into a file and prints inference time (per frame).

> NOTE: Update the parameters according with your values while running infer.py
```
python3 infer.py --model_path="yolov8n-seg_openvino_model/yolov8n-seg.xml" --device="CPU" --video="pharmacy_drivethru.mp4"
```
## Audio AI Workload : Whisper
This section provides instructions for performance testing the [Whisper Base](https://github.com/openai/whisper#available-models-and-languages) multilingual model. The original model is exported into OpenVINO format and then used for inference using OpenVINO runtime.

The `audio-ai-workload` directory contains python script to export the model and transcribe(inference) an audio using Whispher model.

### Python Dependencies
Install the Python packages:

```
pip3 install -r requirements.txt
```
**Note**: Install packages in a python virtual environment to avoid conflict with existing packages.

### Export Model
Convert Whisper base model into OpenVINO IR format.

For converting the model into OpenVINO format refer to their official Jupyter Notebook on [github](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/227-whisper-subtitles-generation/227-whisper-convert.ipynb) to generate whisper_encoder and whisper_decoder .bin and .xml files. The notebook can be followed along until the encoder and decoder model is saved and rest of the code can be ignored in this case.


### Inference

Transcribe an audio file into text.

> NOTE: Update the parameters according with your values while running infer.py

```
python3 infer.py --model_name="base" --ovmodel_path="whisper-base-ov-model" --device="CPU" --audio="sample.mp3"
```

## Core Pinning
Binds a process or thread to specific physical CPU cores.

`OMP_NUM_THREADS=<num_of_cores> numactl --localalloc --physcpubind=<cpus> python3 <script>`

Ref:  [Numactl manual](https://linux.die.net/man/8/numactl)

## GPU Performance Testing
This section provides instructions for performance testing the AI workloads on Nvidia GPU. The GPU performance test includes below workloads:
- [LLM Workload: Llama 2](#llm-workload--llama-2-1)

The performance testing is executed in below Systems. For details refer [SUT](./SUT.pdf)

### Dell PowerEdge Servers

| System         | CPU                   | Cores per Socket | Sockets | vCPUs | GPU | GPU Memory | No of GPUs |
|------------------------|-----------------------|------------------|---------|-------| ----- | ----- | ----- |
| Dell PowerEdge XR4520c | Intel(R) Xeon(R) D-2776NT (3rd Gen) | 16               | 1       | 32   | Nvidia A30 | 24 GB | 1 |

### AWS Instances

Here are the comparable AWS EC2 instances for Dell PowerEdge XR4520c:

| Instance Type          | CPU                   | Cores per Socket | Sockets | vCPUs | GPU | GPU Memory | No of GPUs |  Pricing (On demand Linux pricing per hour) |
|------------------------|-----------------------|------------------|---------|-------| -------| ----- | ----- | -------|
| g5.8xlarge | AMD EPYC 7R32  | 16        | 1      | 32 | Nvidia A10 | 24 GB | 1  | 2.448 USD  |

## LLM Workload : Llama 2
This section provides instructions for performance testing [Llama 2 7B chat HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model served on [Text Generation Inference](https://github.com/huggingface/text-generation-inference) server. The model is loaded into GPU by the TGI server to run inference. [Apache Bench](https://httpd.apache.org/docs/2.4/programs/ab.html) is being used for loading testing.

### Start Text Generation Inference (TGI) Server
1. Set environment variables:
    ```sh
    export MODEL=meta-llama/Llama-2-7b-chat-hf
    export VOLUME=$PWD/data
    export TOKEN=[UPDATE HF TOKEN HERE]
    export MAX_CONCURRENT_REQUESTS=2000
    export DTYPE=bfloat16
    ```

    **Note**: HF access token is required to serve gated models in HuggingFace.
2. Start TGI server using Docker container:
    ```
    sudo docker run -d --name tgi --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$TOKEN -p 8080:80 -v $VOLUME:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $MODEL --max-concurrent-requests $MAX_CONCURRENT_REQUESTS --dtype $DTYPE
    ```
Refer [TGI](https://github.com/huggingface/text-generation-inference) github for more details.

### Intiate Concurrent Request
Concurrent requests are initiated using Apache Bench tool:
1. Save below json data into `data.json` file:
    ```json
    {"inputs":"Discuss the history and evolution of artificial intelligence in 80 words","parameters":{"max_new_tokens":100}}
    ```
2. Run concurrent request using Apache Bench:
    ```
    ab -n [total no of requests] -c [no of concurrent requests per test] -T application/json -p data.json http://127.0.0.1:8080/generate
    ```

Refer [Apache Bench](https://httpd.apache.org/docs/2.4/programs/ab.html) manual for more details.

### Getting Performance Testing Data
1. Apache Bench will provide a report once all the requests are completed. Request per second and total time taken for test can be collected from the report.
2. The inference time (s) and time per token (ms) are found in TGI server logs.
To view TGI server logs:
    ```sh
    sudo docker logs tgi
    ```
