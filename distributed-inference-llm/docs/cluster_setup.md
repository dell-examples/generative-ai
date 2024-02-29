# Setting up a Distributed Cluster

**Estimated Time: 40 mins ⏱️**

This section acts as the foundation step. Learn how to configure your cluster using lightweight [K3S](https://docs.k3s.io/) on both server and agent nodes. Explore [Helm](https://helm.sh/), the package manager for Kubernetes, that helps streamline installations. Understand the intricate integration of [NVIDIA® device plugins](https://github.com/NVIDIA/k8s-device-plugin) and [AMD K8S device plugin](https://github.com/ROCm/k8s-device-plugin).


## Table of Contents
* [Setting up Azure Container Registry](#setting-up-azure-container-registryoptional)
* [K3S setup on Server and Agent Nodes](#k3s-setup-on-server-and-agent-nodes)
    * [Server Node Setup](#server-node-setup)
    * [Agent Node Setup](#agent-node-setup)
* [Installing Helm](#installing-helm)
* [NVIDIA® Device Plugins for Kubernetes](#nvidia®-device-plugins-for-kubernetes)
* [AMD Device Plugins for Kubernetes](#amd-device-plugins-for-kubernetes)
* [Network File System (NFS) Setup](#network-file-system-nfs-setup)
    * [Setting up NFS Server](#setting-up-nfs-server)
    * [Creating the Kubernetes Persistent Volume and Claims for NFS](#creating-the-kubernetes-persistent-volume-and-claims-for-nfs)
        * [Create a Persistent Volume](#create-a-persistent-volume)
        * [Create a Persistent Volume Claim](#create-a-persistent-volume-claim)


## Setting up Azure Container Registry[Optional]

> [!NOTE]
> You can setup a container registry of your choice.

Follow the below steps to setup the Azure Container Registry (ACR) on development machine.

1. Follow the [Microsoft Azure documentation](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-portal?tabs=azure-cli) for creating an Azure Container Registry on your azure portal.
2. Sign in to the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli) and login to your container registry.



## K3S setup on Server and Agent Nodes

### Server Node Setup

> [!NOTE]
> The following setup is for the Server/Head node.

Select one of the node as the Server/Head node for the K3S cluster and follow the below steps.

1. Select the ethernet interface for the node if you have multiple NICs available on the system.

2. Collect and set the ethernet interface name of the selected interface.

    You can use `ifconfig` command to get the ethernet details.
    ```sh
    export ETH_IFACE=<Ethernet Interface Name>
    ```
    Replace the `<Ethernet Interface Name>` with selected interface.

3. Setup the k3s service as the server node.

    ```sh
    export INSTALL_K3S_EXEC=" --write-kubeconfig ~/.kube/config --flannel-iface $ETH_IFACE --flannel-backend=host-gw"

    curl -sfL https://get.k3s.io | sh -s -
    ```
4. Change permission of the kube config to get non root access.

    ```sh
    sudo chown -R <user>:<group> ~/.kube
    ```

    Replace the `<user>` and `<group>` with your machine user and group names.

    Example:

    ```sh
    sudo chown -R user:user ~/.kube
    ```

5. Collect the node token of the server for setting up the agent/worker nodes.

    ```sh
    sudo cat /var/lib/rancher/k3s/server/node-token
    ```
6. After completing the k3s server setup, verify the running status of the k3s service.
    ```sh
    sudo systemctl status k3s
    ```

### Agent Node Setup

> [!NOTE]
> Complete the [Server Node Setup](#server-node-setup) before starting agent node setup.

Follow the below steps to add nodes to the k3s cluster as agent/worker nodes.

1. Collect the ethernet interface name of the selected interface of your choice.

     You can use `ifconfig` command to get the ethernet details.
    ```sh
    export ETH_IFACE=<Ethernet Interface Name>
    ```
    Replace the `<Ethernet Interface Name>` with selected interface.

2. Set the k3s server node token.

    ```sh
    export K3S_TOKEN=<Server Node Token>
    ```
    * The node token is accessible at `/var/lib/rancher/k3s/server/node-token` on your server node.

3. Set the k3s server node URL.
    > *Note: Use the IP/Hostname of the Ethernet Interface you selected for the server/head node.*

    ```sh
    export K3S_URL="https://<Server Node IP>:6443"
    ```

    Replace the `<Server Node IP>` with IP/hostname of the server node.

4. Setup the k3s service as the agent node.

    ```sh
    export INSTALL_K3S_EXEC=" --flannel-iface $ETH_IFACE"

    curl -sfL https://get.k3s.io | sh -s -
    ```
5. Once the k3s agent setup is completed. Verify that the k3s service is running by checking the status.
    ```sh
    sudo systemctl status k3s-agent
    ```

## Installing Helm

Install the [Helm v3.12.3 or higher](https://helm.sh/) CLI on the server/head node by following the below steps.

1. Follow the [Helm Documentation](https://helm.sh/docs/intro/install/) to install the helm cli.
2. Update the `KUBECONFIG` environment variable to provide helm access to k3s cluster.

    * Add the below line on the `~/.bashrc` file.
        ```sh
        KUBECONFIG="/home/<user>/.kube/config"
        ```

        update the `<user>` with your machine non root user.
3. Verify the helm cli installation by checking its version

    ```sh
    helm version
    ```

## NVIDIA® Device Plugins for Kubernetes

Once the k3s cluster is setup successfully, the NVIDIA® plugins for Kubernetes needs to be installed to provide access to NVIDIA® GPUs for the cluster pods.

<details>
<summary>Prerequisites</summary>
Before deploying the plugin, make sure that all NVIDIA® GPU drivers are properly installed.

You can verify the GPU driver installation by running `nvidia-smi` on the server.

You should see a similar output as below for the command.

```sh
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:1B:00.0 Off |                  Off |
| N/A   33C    P0              62W / 500W |      4MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

</details>


Follow the below steps on the server/head node to setup.

1. Add the NVIDIA® Helm repository.

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update
    ```
2. Install the NVIDIA® GPU operator plugins

    ```sh
    helm install gpu-operator -n gpu-operator --create-namespace \
    nvidia/gpu-operator $HELM_OPTIONS \
    --set toolkit.env[0].name=CONTAINERD_CONFIG \
    --set toolkit.env[0].value=/var/lib/rancher/rke2/agent/etc/containerd/config.toml.tmpl \
    --set toolkit.env[1].name=CONTAINERD_SOCKET \
    --set toolkit.env[1].value=/run/k3s/containerd/containerd.sock \
    --set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
    --set toolkit.env[2].value=nvidia \
    --set toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT \
    --set-string toolkit.env[3].value=true
    ```

<details>
<summary>Verify the NVIDIA® GPU access on nodes</summary>

* List the nodes on your cluster.

    ```sh
    kubectl get nodes
    ```
* Get the details of the node with NVIDIA® GPU.

    ```sh
    kubectl describe nodes xe9680
    ```
    You should see the details of NVIDIA® GPUs under Capacity section on the output similar to the below for each nodes.

    ```yaml
    ...
    Capacity:
    cpu:                224
    ephemeral-storage:  3072261880Ki
    hugepages-1Gi:      0
    hugepages-2Mi:      0
    memory:             2113250420Ki
    nvidia.com/gpu:     8 # Available GPU on the node
    pods:               110
    Allocatable:
    ...
    ```

> [!NOTE]
> This installation method is for k3s, refer  [NVIDIA® GPU Operator Installation page](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) to know more about the NVIDIA® GPU operator plugins on Kubernetes.

</details>


## AMD Device Plugins for Kubernetes

The AMD plugins for Kubernetes needs to be installed to provide access to AMD GPUs for the cluster pods.

<details>
<summary>Prerequisites</summary>
Before deploying the plugin, make sure that all AMD GPU drivers are properly installed.

You can verify the GPU driver installation by running `rocm-smi` on the server.
</details>

1. Installing AMD GPU Operator

    ```sh
    helm repo add amd-gpu-helm https://rocm.github.io/k8s-device-plugin/ && helm repo update
    ```

2. Install the AMD GPU operator plugins

    ```sh
    helm install amd-gpu amd-gpu-helm/amd-gpu --version 0.10.0
    ```

<details>
<summary>Verify the AMD GPU access on nodes</summary>

* List the nodes on your cluster.

    ```sh
    kubectl get nodes
    ```
* Get the details of the node with AMD GPU.

    ```sh
    kubectl describe nodes r7526
    ```
    You should see the details of AMD GPUs under Capacity section on the output similar to the below for each nodes.

    ```yaml
    ...
    Capacity:
    cpu:                224
    ephemeral-storage:  3072261880Ki
    hugepages-1Gi:      0
    hugepages-2Mi:      0
    memory:             2113250420Ki
    amd.com/gpu:        4 # Available GPU on the node
    pods:               110
    Allocatable:
    ...
    ```

> [!NOTE]
> To know more about the AMD GPU Plugin refer [AMD ROCm™/k8s-device-plugin | Github](https://github.com/ROCm/k8s-device-plugin)

</details>

## Network File System (NFS) Setup

### Setting up NFS Server

> [!NOTE]
> Skip this step if a NFS Server like a [Dell™ PowerScale devices](https://www.dell.com/en-in/work/shop/powerscale-family/sf/powerscale)  already available

The Network File System is used to store the models for the serving. You can save the the serving start time with NFS. The below steps create a NFS server on one of your nodes.

1. Install the nfs server

    ```sh
    sudo apt-get update
    sudo apt-get install nfs-kernel-server
    ```
2. Create a directory `<Directory Path>` to use as the mount directory.
3. Update the file `/etc/exports` file with directory details as root user.

    ```sh
    <Directory Path> *(rw,sync,no_subtree_check,no_root_squash)
    ```
    > [!NOTE]
    > This provides NFS access to all machine on the network.*

4. Restart the nfs service to apply the changes.

    ```sh
    sudo systemctl restart nfs-kernel-server
    ```
5. Verify that the nfs server is running by checking its status

    ```sh
    sudo systemctl status nfs-kernel-server
    ```


### Creating the Kubernetes Persistent Volume and Claims for NFS

#### Create a Persistent Volume

1. Create a yaml file (`nfs_pv.yml`) for the persistent volume.

    ```yaml
    apiVersion: v1
    kind: PersistentVolume
    metadata:
        name: nfs-pv
    spec:
        storageClassName: ""
        capacity:
            storage: <Storage Size>
        volumeMode: Filesystem
        accessModes:
            - ReadWriteMany
        nfs:
            path: <Directory Path>
            server: <Server IP>
    ```

    Update the below parameters accordingly.

    | Name |  Description | Example |
    | -- | --- | --- |
    | `Storage Size` | Storage space allocated to the NFS volume. | `500Gi` |
    | `Directory Path` | Directory path of the NFS Volume on the machine. <br> Use the directory configured for [Network File System](#setting-up-nfs-server). | `/home/user/nfs_mount` |
    | `Server IP` | IP/Hostname of the machine with NFS Server is setup. | `192.10.1.101` |

2. Apply the Persistent Volume on the cluster

    ```sh
    kubectl apply -f nfs_pv.yml
    ```
3. Verify that the `nfs-pv` persistent volume is created.

    ```sh
    kubectl get pv
    ```

    You should see the details of the `nfs-pv` persistent volume similar to below

    ```sh
    NAME     CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS         CLAIM             STORAGECLASS   REASON   AGE
    nfs-pv   500Gi      RWX            Retain           Available                                                5m
    ```

#### Create a Persistent Volume Claim

1. Create a yaml file (`nfs_pvc.yml`) for the persistent volume claim for the persistent volume created on the previous steps.

    ```yaml
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
        name: nfs-pvc
    spec:
        storageClassName: ""
        volumeName: nfs-pv
        accessModes:
            - ReadWriteMany
        resources:
            requests:
                storage: <Storage Size>
    ```

    Update the below parameters accordingly.

    | Name |  Description | Example |
    | -- | --- | --- |
    | `Storage Size` | Storage space allocated to the NFS volume. <br> The storage size for the claim shoul be less than or equal to the Persistent Volume size. | `500Gi` |
2. Apply the persistent volume claim on the cluster

    ```sh
    kubectl apply -f nfs_pvc.yml
    ```
3. Verify that the `nfs-pvc` Persistent Volume Claim is created successfully

    ```sh
    kubectl get pvc
    ```

    You should see the details of the `nfs-pvc` as below

    ```sh
    NAME      STATUS   VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   AGE
    nfs-pvc   Bound    nfs-pv   500Gi      RWX                           5m
    ```

[Back to Deployment Guide](../README.md#deployment-guide)
