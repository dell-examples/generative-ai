# KubeRay Installation and Ray Cluster Configuration

**Estimated Time: 40 mins ⏱️**


In this section, we streamline the integration of KubeRay and configuring our Ray Clusters for fine-tuning.

## Table of Contents

* [KubeRay Installation](#kuberay-installation)
* [Ray Cluster Configuration](#ray-cluster-configuration)
    * [Creating Fine-Tuning Docker Images and Kubernetes Secrets](#creating-fine-tuning-docker-images-and-kubernetes-secrets)
    * [Configuring Ray Cluster](#configuring-the-cluster)
    * [Deploying Ray Cluster](#deploying-ray-cluster)
* [Accessing Ray Dashboard](#accessing-the-ray-dashboard)


## KubeRay Installation

Follow the below steps to deploy the [KubeRay Operator v1.0.0-rc.0](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) on your k3s cluster.

1. Add the kuberay repo to helm

    ```sh
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    ```
2. Install the CRDs and KubeRay operators

    ```sh
    helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0
    ```
3. Verify the kuberay operator pod deployment

    ```sh
    kubectl get pods
    ```

    You should see an output similar to this

    ```sh
    NAME                                READY   STATUS    RESTARTS   AGE
    kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          27s
    ```

## Ray Cluster Configuration

### Creating Fine-Tuning Docker Images and Kubernetes Secrets

#### Building Docker Images

Follow the below steps to build and push the docker images for training to a container registry.

1. Build the docker image

    ```sh
    cd training
    sudo docker build -t train:latest .
    ```

2. Tag the docker images with container registry URI.
    > *Note: The tag should be based on your container registry name.*
    ```sh
    sudo docker tag train:latest <container registry>/train:latest
    ```

    Update the `<container registry>` with your container registry URI.

3. Push the docker image to container registry.

    ```
    sudo docker push <container registry>/train:latest
    ```

    Update the `<container registry>` with your container registry URI.

#### Creating Image Pull Secret for Kubernetes

> *Note: The step may vary based on different container registries.*

The Image Pull Secret should be created for the cluster to pull images from the Azure container registry.

1. Collect the below details from your container registry

    | Name | Details | Example |
    | --- | --- | --- |
    | `CR Server` | Container registry server URI. | `train.cr.io` |
    | `CR User Name` | Container registry user name | `user` |
    | `CR Password` | Container registry password | `password123` |

2. Create the secret on your k3s cluster.

    ```sh
    kubectl create secret docker-registry cr-login --docker-server=<CR Server> --docker-username=<CR User Name> --docker-password=<CR Password>
    ```

### Configuring the cluster

The [Ray cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html#ray-cluster) is defined by the cluster deployment yaml [cluster.yml](../training/cluster.yml).
Refer to the detailed configuration below to customize the cluster settings according to your hardware availability.

> *Note: To know more about the Ray Cluster Configurations on Kubernetes, refer [RayCluster Configuration page](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html#kuberay-config).*


```yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: raycluster
spec:
  rayVersion: "2.3.0"
     ...
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      # setting `num-gpus` on the rayStartParams enables
      # head node to be used as a worker node
      num-gpus: "8"
      ...
    template: # Pod template
        metadata: # Pod metadata
        labels:
          name: ray-head # label for adding node port service
        spec: # Pod spec
            runtimeClassName: nvidia # Runtime to enable NVIDIA GPU access
            containers:
            - name: ray-head
              image: train.cr.io/train:latest # Training docker image
              resources:
                limits:
                  cpu: 50 # Number of CPU cores allocated
                  memory: 500Gi # Memory allocated
                  nvidia.com/gpu: 8 # Number of NVIDIA GPUs allocated
                  ...
                requests:
                  cpu: 14
                  memory: 500Gi
                  nvidia.com/gpu: 8
                  ...
              # Keep this preStop hook in each Ray container config.
              lifecycle:
                preStop:
                  exec:
                    command: ["/bin/sh","-c","ray stop"]
              volumeMounts:
                - name: nfs-pv
                mountPath: /train # NFS mount path
        volumes:
        - name: nfs-pv
          persistentVolumeClaim:
            claimName: nfs-pvc
        imagePullSecrets:
        - name: cr-login # Image pull secret
        ...
  workerGroupSpecs:
  - groupName: gpu-group
    replicas: 2 # Available agent nodes
    minReplicas: 2 # Available agent nodes
    maxReplicas: 5
    rayStartParams:
        num-gpus: "4"
        ...
    template: # Pod template, same as head group
      spec:
        ...
  # Another workerGroup
  - groupName: medium-group
    ...
---
# The ray dashboard is configured as node port on 30265
# Ray dashboard port(8265) service
apiVersion: v1
kind: Service
metadata:
  name: ray-head-dashboard-port
spec:
  selector:
    name: ray-head
  type: NodePort
  ports:
  - port: 8265
    targetPort: 8265
    nodePort: 30265 # the ray dashboard is accessible at this port
```

The [cluster.yml](../training/cluster.yml) available on the repository follows the below configuration for the ray cluster.

| Node ID | <b>Type of Node | | Allocations<b> |  | |
| -- | --- | -- | -- | -- | -- |
|  | | <b>CPU<b> | <b>Memory</b> | <b>Disk</b> | <b>GPU</b> |
| Node 1 | Head + Worker |  100 | 1TB | 2TB | 8 |
| Node 2 | Worker |  100 | 1TB | 1TB | 4 |
| Node 3 | Worker |  100 | 1TB | 1TB | 4 |


### Deploying Ray Cluster

1. Update the [cluster.yml](../training/cluster.yml) by following the configurations described [above](#configuring-the-cluster).
1. Once the cluster is configured, deploy to the k3s cluster

    ```sh
    kubectl apply -f cluster.yml
    ```
    >*Note: The pods might take more than 10 mins to start running based on systems network speed.*
3. Verify the training pods running on all of the k3s nodes.

    ```sh
    kubectl get po -o wide
    ```
    Wait until all the pods are at `Running` state before continuing.

    ```sh
    NAME                                READY   STATUS    RESTARTS       AGE   IP           NODE     NOMINATED NODE   READINESS GATES
    kuberay-operator-58c98b495b-xk6qz   1/1     Running   0              20m   10.42.0.15   xe9680   <none>           <none>
    raycluster-head-5lbx4               1/1     Running   0              15m   10.42.0.47   xe9680   <none>           <none>
    raycluster-worker-gpu-group-8cj75   1/1     Running   0              15m   10.42.1.39   xe8545   <none>           <none>
    raycluster-worker-gpu-group-8cj75   1/1     Running   0              15m   10.42.1.39   r760xa   <none>           <none>
    ```

## Accessing the Ray Dashboard

Ray provides a web-based [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html) for monitoring and debugging Ray Cluster. The visual representation of the system state, allows users to track the performance of the cluster and troubleshoot issues.

1. Once all the cluster pods are running, the ray cluster dashboard is accessible at the [localhost:30265](http://localhost:30265).

2. The **Cluster** section on provides the details on of all nodes connected to the cluster.

    ![Ray Dashboard](../assets/ray_dashboard.png)

## Our Test Cluster for Llama 2 7B Fine-Tuning

Below are the details of the cluster we configured for fine-tuning Llama 2 7B LLM model.

| Server Name | CPU | Memory | OS | Disk | GPUs | NICs |
| -- | -- | --- | -- | -- | -- | -- |
| [Dell PowerEdge XE9680](https://www.dell.com/en-in/work/shop/ipovw/poweredge-xe9680) | Intel(R) Xeon(R) Platinum 8480+ <br> 56 Cores <br> 2 Sockets | 2 TB | Ubuntu 22.04.1 LTS <br> 5.15.0-86-generic | 2 TB | 8x[NVIDIA A100 80 GB SXM](https://www.nvidia.com/en-in/data-center/a100/) GPUs | 1x[BCM57508 NetXtreme-E 100 GbE](https://www.broadcom.com/products/ethernet-connectivity/network-adapters/p2100g) |
| [Dell PowerEdge XE8545](https://www.dell.com/en-us/shop/ipovw/poweredge-xe8545) | AMD EPYC 7763 64-Core Processor <br> 64 Cores <br> 2 Sockets | 1 TB | Ubuntu 22.04.1 LTS <br> 5.15.0-84-generic | 1 TB | 4x[NVIDIA A100 80 GB SXM](https://www.nvidia.com/en-in/data-center/a100/) GPUs | 1x[BCM57508 NetXtreme-E 100 GbE](https://www.broadcom.com/products/ethernet-connectivity/network-adapters/p2100g) |
| [Dell PowerEdge R760xa](https://www.dell.com/en-us/shop/dell-poweredge-servers/poweredge-r760xa-rack-server/spd/poweredge-r760xa/pe_r760xa_16902_vi_vp) | Intel(R) Xeon(R) Platinum 8480+ <br> 56 Cores <br> 2 Sockets | 1 TB | Ubuntu 22.04.1 LTS <br> 5.15.0-84-generic | 1 TB | 4x[NVIDIA H100 80 GB PCIe](https://www.nvidia.com/en-in/data-center/h100/) GPUs | 1x[BCM57508 NetXtreme-E 100 GbE](https://www.broadcom.com/products/ethernet-connectivity/network-adapters/p2100g) |
