# Medical Performance Testing

## Introduction
This technical guide will provide an overview of a performance comparison between [ZenDNN TensorFlow with tuning guide settings](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58205-epyc-9004-tg-aiml.pdf) vs default TensorFlow for Resnet50 Pneumonia classification model.

## Getting Started
### Pre-requisites

#### Software
Before starting the performance testing process, ensure you have the following prerequisites installed:

- Ubuntu 22.04
- Python 3.11

#### Hardware
- System - Dell PowerEdge R7625
- CPU - AMD EPYC 9554 64-Core Processor
- NumberOfSockets - 2

## Performance Testing TF Resnet50 Pneumonia classification model
This section provides instructions for performance testing the Resnet50 Pneumonia classification model workload on CPU for ZenDNN TensorFlow(Node pinning) vs default TensorFlow.

Note: Please ensure that you run the following two modes in separate Python virtual environments.

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
3) Download [TF_v2.12_ZenDNN_v4.1_Python_v3.11.zip](https://www.amd.com/en/developer/zendnn/eula/tensorflow-4-1-notices-and-licenses.html?filename=TF_v2.12_ZenDNN_v4.1_Python_v3.11.zip)
3) ZenDNN TensorFlow installation
    Refer this [ZenDNN technical guide](https://www.amd.com/content/dam/amd/en/documents/developer/tensorflow-zendnn-user-guide-4.0.pdf) to install ZenDNN TensorFlow 4.1

#### Configure Environment settings as per ZenDNN tuning guide
1) Configure NUMA Node per socket to 4 in BIOS settings

2) Configure ENV variables
    ```
    export TF_ENABLE_ZENDNN_OPTS=0
    export ZENDNN_CONV_ALGO=3
    export ZENDNN_TF_CONV_ADD_FUSION_SAFE=1
    export ZENDNN_TENSOR_POOL_LIMIT=512
    export OMP_NUM_THREADS=128
    ```
3) Configure Transparent huge pages settings
    ```
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    ```

#### Run inference
1) Execute below command to run inference and get inference time
    ```
    python3 infer.py --model <repo home>/medical-usecase/model_repository/pneumonia/1/pneumonia_model.pb
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
    python3 infer.py --model <repo home>/medical-usecase/model_repository/pneumonia/1/pneumonia_model.pb
    ```
