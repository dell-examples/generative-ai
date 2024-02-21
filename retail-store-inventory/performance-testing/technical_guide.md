# Retail Performance Testing

## Introduction
This technical guide will provide an overview of a performance comparison between [ZenDNN TensorFlow with tuning guide settings](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58205-epyc-9004-tg-aiml.pdf) vs default TensorFlow for Mobilenet-SSD detection model.

## Getting Started
### Pre-requisites

#### Software
Before starting the performance testing process, ensure you have the following prerequisites installed:

- Ubuntu 22.04
- Python 3.11

#### Hardware
- System - Dell PowerEdge R7615
- CPU - AMD EPYC 9354
- NumberOfSockets - 2

## Performance Testing TF Mobilenet-SSD detection model
This section provides instructions for performance testing the Mobilenet-SSD detection model workload on CPU for ZenDNN TensorFlow(Node pinning) vs default TensorFlow.

### Test environment and components for ZenDNN TensorFlow(Node pinning) vs default TensorFlow
| Name | Default | ZenDNN |
| --- | ---- | ---- |
| Python | 3.11 | 3.11 |
| Tensorflow | [Tensorflow - 2.13](https://pypi.org/project/tensorflow/) | [TF_v2.12_ZenDNN_v4.1_Python_v3.11](https://download.amd.com/developer/eula/zendnn/zendnn-4-1/tensorflow/TF_v2.12_ZenDNN_v4.1_Python_v3.11.zip) |
| BIOS Tuning Recommendations | NUMA Node per socket (NPS) NPS4 | NUMA Node per socket (NPS) NPS4 |
| Transparent Huge Pages | default: madvice | always |
| ENV config | - | export TF_ENABLE_ZENDNN_OPTS=0
| ENV config | - | export ZENDNN_CONV_ALGO=3
| ENV config | - | export ZENDNN_TF_CONV_ADD_FUSION_SAFE=0
| ENV config | - | export ZENDNN_TENSOR_POOL_LIMIT=512
| ENV config | - | export OMP_NUM_THREADS=32
| ENV config | - | export GOMP_CPU_AFFINITY=0-31 |

### ZenDNN TensorFlow mode
This section provides instructions for performance testing with ZenDNN TensorFlow.

#### Pre-requisites
1) Install NUMACTL
`sudo apt install numactl`

2) Python Dependencies
Install the following Python dependencies:

    ```
    pip3 install -r ./zendnn_workload/requirements.txt
    ```
3) Download [TF_v2.12_ZenDNN_v4.1_Python_v3.11.zip](https://download.amd.com/developer/eula/zendnn/zendnn-4-1/tensorflow/TF_v2.12_ZenDNN_v4.1_Python_v3.11.zip)
4) ZenDNN TensorFlow installation
    Refer this [ZenDNN technical guide](https://www.amd.com/content/dam/amd/en/documents/developer/tensorflow-zendnn-user-guide-4.0.pdf) to install ZenDNN TensorFlow 4.1

#### Configure Environment settings as per ZenDNN tuning guide
1) Configure NUMA Node per socket to 4 in BIOS settings

2) Configure ENV variables
    ```
    export TF_ENABLE_ZENDNN_OPTS=0
    export ZENDNN_CONV_ALGO=3
    export ZENDNN_TF_CONV_ADD_FUSION_SAFE=0
    export ZENDNN_TENSOR_POOL_LIMIT=512
    export OMP_NUM_THREADS=32
    ```
3) Configure Transparent huge pages settings
    ```
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    ```

#### Run inference
1) Execute below command to run inference and get inference time
    ```
    python3 infer.py --model <repo home>/retail-usecase/pipeline/retail/src/saved_model
    ```
#### Node Pinning
1) Follow below step to run inference with node pinning. This shall bind a process or thread to specific physical CPU cores.

    `numactl --cpunodebind=<0-3> --membind=<0-3> python3 <script>`

### Default TensorFlow mode
This section provides instructions for performance testing with Default TensorFlow.

#### Pre-requisites
1) Python Dependencies
    Install the following Python dependencies:

    ```
    pip3 install -r ./default_workload/requirements.txt
    ```

2) Configure Transparent huge pages settings
    ```
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    ```

#### Run inference
1) Execute below command to run inference and get inference time
    ```
    python3 infer.py --model <repo home>/retail-usecase/pipeline/retail/src/saved_model
    ```
