# Serving LLM Models

In this section, we explore the intricacies of distributed inference with the LLAMA 2 models.

## Table of Contents

* [Inference Serving Optimizations](#inference-serving-optimizations)
* [Prerequisites](#prerequisites)
* [Serving Llama 2 70B Chat on GPUs](#serving-llama-2-70b-chat-on-gpus)
    * [Building Serving Docker Images](#building-serving-docker-images)
    * [Deployment Configuration](#deployment-configuration)
        * [Ray Cluster Configuration](#ray-cluster-configuration-1)
        * [Serve application Configuration](#serve-application-configuration)
    * [Deploying LLama 2 70B Serving](#deploying-llama-2-70b-serving)
* [Serving Llama 2 7B Chat on CPUs](#serving-llama-2-7b-chat-on-cpus)
    * [Building Serving Docker Images](#building-serving-docker-images-1)
    * [Deployment Configuration](#deployment-configuration-1)
        * [Ray Cluster Configuration](#ray-cluster-configuration-2)
        * [Serve application Configuration](#serve-application-configuration-1)
    * [Deploying Llama 2 7B Serving](#deploying-llama-2-7b-serving)


## Inference Serving Optimizations

This table showcases hardware and corresponding libraries utilized to provide optimized inference serving.

| Backend Hardware | Library | Version |
| --- | --- | --- |
| NVIDIA® GPU | [vLLM](https://github.com/vllm-project/vllm) | `0.2.6` |
| AMD GPU | [vLLM-ROCm](https://github.com/vllm-project/vllm) | `0.2.6` |
| Intel® CPU | [BigDL-LLM](https://github.com/intel-analytics/BigDL) | `2.4.0` |
| AMD CPU | [Optimum ONNX Runtime](https://huggingface.co/docs/optimum/main/en/onnxruntime/overview) | `1.16.1` |

> [!TIP]
> Refer [Infer Backends](./infer_backends.md) to know more about the libraries.

## Prerequisites
* [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
* [Access to Llama 2 Models](https://huggingface.co/meta-llama)


## Serving Llama 2 70B Chat on GPUs

To deploy the Llama 2 70B Chat models on GPUs(NVIDIA®/AMD) refer this section.


### Building Serving Docker Images

Follow the below steps to build and push the docker images to a container registry for inferencing.

1. Build the docker image.

    i. The below example is for building NVIDIA® GPU docker image

      ```sh
      cd serving/gpu
      sudo docker build -t infer:latest -f Dockerfile.nvidia .
      ```
      <details>
      <summary>Build AMD Docker Image</summary>
      To build an AMD GPU Docker image, replace the `Dockerfile.nvidia` name to `Dockerfile.amd`.

      ```sh
      sudo docker build -t infer:latest -f Dockerfile.amd .
      ```
      </details>

2. Tag the docker images with container registry URI.
    > [!NOTE]
    > The tag should be based on your container registry name

    > [!NOTE]
    > The tag should align with the image name.

    Update the `<container registry>` with your container registry URI.

    ```sh
    sudo docker tag infer:latest <container registry>/infer:latest
    ```

3. Push the docker images to container registry.

    Update the `<container registry>` with your container registry URI.

    ```
    sudo docker push <container registry>/infer:latest
    ```

### Deployment Configuration

The Kubernetes deployment configuration has two parts

* [Ray Cluster Configuration](#ray-cluster-configuration)
* [Serve application Configuration](#serve-application-configuration)

#### Ray Cluster Configuration

The Ray cluster needs to configured based on the hardware availability for the serving deployment.

Refer the [Ray Cluster Configuration](./ray_cluster.md#ray-cluster-configuration) for more details.

The [cluster.nvidia.yml](../serving/gpu/cluster.nvidia.yml) available on the repository follows the below configuration for the ray cluster.

| Node ID | <b>Type of Node | | Allocations<b> |  | |
| -- | --- | -- | -- | -- | -- |
|  | | <b>CPU<b> | <b>Memory</b> | <b>Disk</b> | <b>GPU</b> |
| Node 1 | Head |  160 | 300GB | 2TB | 8 |
| Node 2 | Worker |  160 | 300GB | 1TB | 4 |
| Node 3 | Worker |  160 | 300GB | 1TB | 4 |

#### Serve application Configuration

The LLM serving is defined by the Ray Serve configuration.

To serve the Llama 2 70B model on your cluster, `serveConfigV2` section under `spec` with the below details.

| Configuration | Details | Default |
| --- | --- | --- |
| `port` | Serving Port | 8000 |
| `route_prefix` | Serving endpoint route. | `/` |
| `model_name` |Provide HuggingFace model name or the full path to the model mounted on the [cluster NFS](./cluster_setup.md#network-file-system-nfs-setup). | |
| `hf_token` | HuggingFace token |  |
| `gpu_count` | Tensor Parallelism to load the LLM Model. <br> It is recommended to set `4` as tensor parallelism for Llama 2 70B model.  | `4` |
| `data_type` | Inference data type. <br> Supported are `bfloat16`, `float16`, `float32`, `auto`, `half`, `float` | `bfloat16` |
| `batch_size` | Batch size of handling concurrent requests. <br> Set the batch size based on your latency and throughput requirements. | `256` |
| `num_replicas` | Number of serving replicas to deploy. <br> Each replica will use number of gpus set for `gpu_count`. | `4` |
| `max_concurrent_queries` | Set the same value as `batch_size` | `256` |


<details>
<summary>More on our test cluster serve configuration</summary>

Our test cluster consists of a total of 16 NVIDIA® GPUs (12xNVIDIA® A100 Tensor Cores 80GB SXM and 4xNVIDIA® H100 Tensor Cores 80GB PCIe).

The [serve configuration - serving/gpu/cluster.nvidia.yml ](../serving/gpu/cluster.nvidia.yml#L11) we deployed is described below.

```yaml
proxy_location: HeadOnly
http_options:
    host: 0.0.0.0
    port: 8000
applications:
    - name: llama2 # serving application name
    import_path: inference.typed_app_builder # inference script import path
    route_prefix: / # serve endpoint route
    args: # serve deployment arguments
        gpu_count: 4 # number of gpus allocated for each deployment
        model_name: "meta-llama/Llama-2-70b-chat-hf" # LLM model name or path
        data_type: bfloat16 # Inference datatype
        batch_size: 256 # inference batch size
        hf_token: !!!!! # HuggingFace token for model like LLama 2
    deployments:
    - name: VLLMPredictDeployment # Inference class
        num_replicas: 4 # number of deployment replicas
        max_concurrent_queries: 256 # maximum concurrent queries
        ray_actor_options: # ray actor option for the each replica
            num_cpus: 20 # number of CPU core allocated for each replicas
```


</details>


> [!TIP]
> For more information refer [Ray Serve Config Page](https://docs.ray.io/en/latest/serve/production-guide/config.html)


### Deploying LLama 2 70B Serving

1. Make sure that you have updated the [cluster.nvidia.yml](../serving/gpu/cluster.nvidia.yml) according to your requirements.
1. Once the deployment configuration is ready, deploy to the k3s cluster

    ```sh
    cd serve/gpu
    kubectl apply -f cluster.nvidia.yml
    ```

    > [!NOTE]
    > The pods might take more than 10 mins to start running based on the system's network speed.
3. Verify the inferencing pods running on all of the k3s nodes.

    ```sh
    kubectl get po -o wide
    ```
    Wait until all the pods are at `Running` state before continuing.

    ```sh
    NAME                                   READY   STATUS    RESTARTS       AGE   IP           NODE     NOMINATED NODE   READINESS GATES
    kuberay-operator                       1/1     Running   0               1h   10.42.0.15   xe9680   <none>           <none>
    raycluster-head-5lbx4                  1/1     Running   0              10m   10.42.0.32   xe9680   <none>           <none>
    raycluster-worker-gpu1-dsd23           1/1     Running   0              10m   10.42.1.21   xe8545   <none>           <none>
    raycluster-worker-gpu2-df4sa           1/1     Running   0              10m   10.42.1.10   r760xa   <none>           <none>
    ```

> [!TIP]
> To know more about the Llama 2 70B distributed inference benchmarks, refer the [Performance Testing on GPU page](../performance_testing/technical_guide.md#2-performance-testing-on-gpu)

## Serving Llama 2 7B Chat on CPUs

To deploy the Llama 2 7B Chat model on CPUs(Intel®/AMD) refer this section.

### Building Serving Docker Images

Follow the below steps to build and push the docker images to a container registry for inferencing.

1. Build the docker image.

    i. The Intel® CPU Docker Image

      ```sh
      cd serving/cpu/intel
      sudo docker build -t infer_cpu:latest -f Dockerfile.intel .
      ```
    ii. The AMD CPU Docker image

    * Convert the Llama 2 7B model to ONNX format.
        ```sh
        cd serving/amd/model_conversion
        bash install_requirements.txt
        python3 model_conversion.py
        ```
    * Once the model is converted, copy the Llama 2 int8 model to your [NFS directory](./cluster_setup.md#network-file-system-nfs-setup)
    * Build the image
        ```sh
        cd serving/cpu/amd
        sudo docker build -t infer_cpu:latest -f Dockerfile.amd .
        ```
2. Tag the docker images with container registry URI.
    > [!NOTE]
    > The tag should be based on your container registry name

    > [!NOTE]
    > The tag should align with the image name

    Update the `<container registry>` with your container registry URI.

    ```sh
    sudo docker tag infer:latest <container registry>/infer:latest
    ```
3. Push the docker images to container registry.

    Update the `<container registry>` with your container registry URI.

    ```
    sudo docker push <container registry>/infer:latest
    ```
### Deployment Configuration

The Kubernetes deployment configuration has two parts

* Ray Cluster Configuration
* Serve application Configuration

#### Ray Cluster Configuration

The Ray cluster needs to configured based on the hardware availability for the serving deployment.

Refer the [Ray Cluster Configuration](./ray_cluster.md) for more details.

The [cluster.cpu.yml](../serving/cpu/cluster.cpu.yml) available on the repository follows the below configuration for the ray cluster.

| Node ID | <b>Type of Node | | Allocations<b> |  | |
| -- | --- | -- | -- | -- | -- |
|  | | <b>CPU<b> | <b>Memory</b> | <b>Disk</b> | <b>Manufacturer</b> |
| Node 1 | Head |  224 | 1TB | 2TB |  Intel® |
| Node 2 | Worker | 256   | 1TB | 1TB | AMD |
| Node 3 | Worker | 224  | 1TB | 1TB | Intel® |
| Node 4 | Worker |  128 | 1TB | 1TB | AMD |


#### Serve application Configuration

The LLM serving is defined by the Ray Serve configuration.

To serve the Llama 2 7B Chat model on your cluster, `serveConfigV2` section under `spec` with the below details.

| Configuration | Details | Default |
| --- | --- | --- |
| `port` | Serving Port | 8001 |
| `route_prefix` | Serving endpoint route. | `/cpu` for Intel and `/acpu` for AMD |
| `model_name` |Provide HuggingFace model name or the full path to the model mounted on the [cluster NFS](./cluster_setup.md#network-file-system-nfs-setup). | |
| `target_device` | Defines the optimized library to use for inference. <br> `INTEL` uses BigDL-LLM and `AMD` uses Hugging Optimum ONNX Runtime. | `INTEL` for Intel and `AMD` for AMD |
| `hf_token` | HuggingFace token |  |
| `data_type` | Inference data type. <br> Only `int8` is supported. | `int8` |
| `batch_size` | Batch size of handling concurrent requests. <br> Set the batch size based on your latency and throughput requirements. | `1` |
| `max_new_token` | Maximum number of generated tokens for the generation. | `256` |
| `temperature` | LLM generation temperature. | `1.0` |
| `num_cpus` | The number of CPU cores allocated for each replicas. <br> Each replica will use the `num_cpus` cores set. | `224` for Intel® and `128` for AMD. |
| `num_replicas` | Number of serving replicas to deploy. <br> Each replica  | `4` |
| `amd_cpu` | Set this 1, if the deployment targets AMD CPU nodes. | `1` |
| `max_concurrent_queries` | Set the same value as `batch_size` | `1` |


<details>
<summary>More on our test CPU cluster serve configuration</summary>

The [serve configuration - serving/cpu/cluster.cpu.yml](../serving/cpu/cluster.cpu.yml#L11) used in deploying Llama 2 7B Chat model on Intel® and AMD CPUs is as follows

```yaml
proxy_location: HeadOnly
http_options:
    host: 0.0.0.0
    port: 8001
applications:
    - name: llama2-cpu-intel  #serving application name
      import_path: inference:typed_app_builder # inference script import path
      route_prefix: /cpu # inference endpoint route for Intel CPUs
      args:
        model_name: "<Model Path>" # add model path here
        data_type: int8 # inference datatype
        max_new_tokens: 256 # maximum number of generated tokens
        temperature: 1.0 # LLM generation temperature
        batch_timeout: 0.1 # batch wait timeout
        batch_size: 1 # batch size to handle concurrent request.
        hf_token: <HF_TOKEN> # huggingface token here
        target_device: "INTEL" # Targets the BigDL-LLM inference for Intel CPUs
      deployments:
        - name: CPUDeployment
          num_replicas: 2 # number of deployment replicas
          max_concurrent_queries: 1
          ray_actor_options:
          num_cpus: 224 # number of CPU cores allocated for each CPUs
    - name: llama2-amd #serving application name
      import_path: inference:typed_app_builder  # inference script import path
      route_prefix: /acpu # inference endpoint route for AMD CPUs
      args:
        model_name: "<Model Path>" # add model path here
        data_type: int8 # inference datatype
        max_new_tokens: 256 # maximum number of generated tokens
        temperature: 1.0 # LLM generation temperature
        batch_timeout: 0.1 # batch wait timeout
        batch_size: 1 # batch size to handle concurrent request.
        hf_token: <HF_TOKEN> # huggingface token here
        target_device: "AMD" # Targets the HuggingFace Optimum ONNX Runtime inference for AMD CPUs
      deployments:
      - name: CPUDeployment
        num_replicas: 3 # number of deployment replicas
        max_concurrent_queries: 1
        ray_actor_options:
          num_cpus: 128 # number of CPU cores allocated for each CPUs
          amd_cpu: 1 # targets the deployment to the AMD CPU nodes
```
</details>

### Deploying LLama 2 7B Serving

1. Make sure that you have update the [cluster.cpu.yml](../serving/cpu/cluster.cpu.yml) according to your requirements.
1. Once the cluster is configured, deploy to the k3s cluster

    ```sh
    cd serve/cpu
    kubectl apply -f cluster.cpu.yml
    ```
    > [!NOTE]
    > The pods might take more than 10 mins to start running based on the system's network speed.*
3. Verify the inferencing pods running on all of the k3s nodes.

    ```sh
    kubectl get po -o wide
    ```
    Wait until all the pods are at `Running` state before continuing.

    ```sh
    NAME                                READY   STATUS    RESTARTS       AGE   IP           NODE     NOMINATED NODE   READINESS GATES
    kuberay-operator                    1/1     Running   0               1h   10.42.0.15   xe9680   <none>           <none>
    raycluster-head-5lbx4               1/1     Running   0              10m   10.42.0.32   xe9680   <none>           <none>
    raycluster-worker-cpu-amd-1-dsd23   1/1     Running   0              10m   10.42.1.21   xe8545   <none>           <none>
    raycluster-worker-cpu-amd-2-dsd23   1/1     Running   0              10m   10.42.1.23   7625-amd <none>           <none>
    raycluster-worker-intel-2-df4sa     1/1     Running   0              10m   10.42.1.10   r760xa   <none>           <none>
    ```

> [!TIP]
> To know more about the Llama 2 7B distributed inference benchmarks, refer the [Performance Testing on CPU page](../performance_testing/technical_guide.md#3-performance-testing-on-cpu)

[Back to Deployment Guide](../README.md#deployment-guide)
