# ONNX Model Conversion

This document states the steps to convert Llama 2 7B/13B model into ONNX format.

## Table of Contents:

* [Steps for ONNX model conversion](#steps-for-onnx-model-conversion)
* [NFS Setup](#nfs-setup)

## Steps for ONNX model Conversion

1.  Create a virtual environment and activate the virtual environment

2. Install the requirements
    ```bash
    bash ./install_requirements.sh
    ```

3. Run the python script
    ```bash
    python3 model_conversion.py
    ```

The following are the configurations options
that are currently accepted for argument parser in `model_conversion.py`

| Name | Description | Accepted values | Default value |
| --- | --- | --- | --- |
| model_name | The name of the Llama 2 7B/13B model | `str` | `meta-llama/Llama-2-7b-hf` |
| hf_token | The Hugging Face token for accessing the Llama models | `str` |  |
| data_type | The data type for model precision | `str` | `int 8` |

These arguments can be passed as necessary when running the `model_conversion.py` script

The converted model will be created and stored in the directory `onnx_<model_name>-<data_type>` depending on the model name and data type.
E.g. `onnx_Llama-2-7b-hf-int8`

## NFS Setup
Once the model is converted and stored in the `onnx_<model_name>-<data_type>`directory, copy this directory to your NFS directory setup [here](../../../docs/cluster_setup.md#network-file-system-nfs-setup)

[Back to Detailed setup guide](../../../README.md)
