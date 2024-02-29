# v1.0 Release

## Fixes
* Restructured the flow of user guides.
* Improvement in technical contents.
* Code improvements and clean up.

# v0.1 Release

## Highlights
* Generating developer assets for Dell`s enterprise customers to fasttrack distributed inferencing of GenAI(LLAMA2) on heterogenous Dell PowerEdge product portfolio networked using Broadcom ethernet.
* Unlocks the ability to configure distributed inferencing to use combinations of Nvidia GPUs, AMD GPUs, Intel CPUs ,AMD CPUs  or a heterogeneous combination of above.
* Handling LLama 2 70b: LLMs like LLama2 70B have billions of parameters, which makes them requires ~260 GB of GPU memory to load the model and are computationally intensive. Distributed inferencing enables spreading the computational load across multiple GPUs, thereby improving efficiency and speed.

# Components

| Software | Version |
| --- | ---- |
| [K3s](https://github.com/k3s-io/k3s/) | `v1.28.5+k3s1` |
| [Helm](https://github.com/helm/helm) | `v3.12.3` |
| [KubeRay](https://github.com/ray-project/kuberay) | `1.0.0` |
| [NVIDIA GPU Operator Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html#operator-install-guide) | `v23.9.1` |
| [Ray](https://github.com/ray-project/ray) | `2.8.1` |
| [Ray Serve](https://github.com/ray-project/ray) | `2.8.1` |
| [NVIDIA NGC PyTorch Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) | `23.10-py3` |
| [vLLM](https://github.com/vllm-project/vllm) | `0.2.6` |
| [GPUtil](https://github.com/anderskm/gputil) | `1.4.0` |
| [Locust](https://github.com/locustio/locust) | `2.20.0` |
