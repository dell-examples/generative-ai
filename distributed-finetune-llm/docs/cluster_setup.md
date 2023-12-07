# Setting up a Distributed Cluster

**Estimated Time: 40 mins ⏱️**

This section acts as the foundation. Learn how to configure your cluster using lightweight [K3S](https://docs.k3s.io/) on both server and agent nodes. Dive into [Helm](https://helm.sh/), the package manager for Kubernetes, streamlining installations. Understand the intricate integration of [NVIDIA device plugins](https://github.com/NVIDIA/k8s-device-plugin), essential for leveraging NVIDIA GPU acceleration. Network File System (NFS) setup adds a layer of data accessibility, explored further in sub-sections.

You can leverage the capabilities of [Dell PowerScale devices](https://www.dell.com/en-in/work/shop/powerscale-family/sf/powerscale) for setting up the NFS for your cluster.

## Table of Contents
* [Setting up Azure Container Registry](#setting-up-azure-container-registryoptional)
* [K3S setup on Server and Agent Nodes](#k3s-setup-on-server-and-agent-nodes)
    * [Server Node Setup](#server-node-setup)
    * [Agent Node Setup](#agent-node-setup)
* [Installing Helm](#installing-helm)
* [NVIDIA Device Plugins for Kubernetes](#nvidia-device-plugins-for-kubernetes)
* [Network File System (NFS) Setup](#network-file-system-nfs-setup)
    * [Setting up NFS Server](#setting-up-nfs-server)
    * [Creating the Kubernetes Persistent Volume and Claims for NFS](#creating-the-kubernetes-persistent-volume-and-claims-for-nfs)
        * [Create a Persistent Volume](#create-a-persistent-volume)
        * [Create a Persistent Volume Claim](#create-a-persistent-volume-claim)


## Setting up Azure Container Registry[Optional]

> *Note: You can setup container registry of your choice.*

Follow the below steps to setup the Azure Container Registry (ACR) on development machine.

1. Follow the [Microsoft Azure documentation](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-portal?tabs=azure-cli) for creating an Azure Container Registry on your azure portal.
2. Sign in to the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli) and login to your container registry.


## K3S setup on Server and Agent Nodes

### Server Node Setup

> *Note: The following setup is for the Server/Head node.*

> *Note: The k3s cluster will be setup with `containerd` as runtime.*

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

> *Note: Complete the [Server Node Setup](#server-node-setup) before starting agent node setup.*

Follow the below steps to add nodes to the k3s cluster as agent/worker nodes.

1. Collect the ethernet interface name of the selected interface of your choice.

     You can use `ifconfig` command to get the ethernet details.
    ```sh
    export ETH_IFACE=<Ethernet Interface Name>
    ```
    Replace the `<Ethernet Interface Name>` with selected interface.

2. set the k3s server node token.

    ```sh
    export K3S_TOKEN=<Server Node Token>
    ```
    * The node token is accessible at `/var/lib/rancher/k3s/server/node-token` on your server node.
3. set the k3s server node URL.
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

Install the [Helm v3.12.3 or higher](https://helm.sh/) CLI by on the server/head node by following the below steps.

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

## NVIDIA Device Plugins for Kubernetes

Once the k3s cluster is setup successfully, the NVIDIA plugins for Kubernetes needs to be installed to provide access to NVIDIA GPUs to the cluster pods. Follow the below steps on the server/head node to setup.

1. Verify that all nodes are added to the cluster.

    ```sh
    kubectl get nodes -o wide
    ```

    You should see all the nodes listed and its status as ready.

    ```sh
    NAME     STATUS   ROLES                  AGE   VERSION        INTERNAL-IP     EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION      CONTAINER-RUNTIME
    xe9680   Ready    control-plane,master   40s   v1.27.6+k3s1   192.168.1.110   <none>        Ubuntu 22.04.1 LTS   5.15.0-86-generic   containerd://1.7.6-k3s1.27
    xe8545   Ready    <none>                 20h   v1.27.6+k3s1   192.168.1.101   <none>        Ubuntu 22.04.3 LTS   5.15.0-84-generic   containerd://1.7.6-k3s1.27
    r760xa   Ready    <none>                 20h   v1.27.6+k3s1   192.168.1.105   <none>        Ubuntu 22.04.3 LTS   5.15.0-84-generic   containerd://1.7.6-k3s1.27
    ```
2. Install the NVIDIA device plugins repo using helm

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update
    ```
3. Install the NVIDIA GPU operator plugins

    ```sh
    helm install --wait nvidiagpu \
     -n gpu-operator --create-namespace \
    --set toolkit.env[0].name=CONTAINERD_CONFIG \
    --set toolkit.env[0].value=/var/lib/rancher/k3s/agent/etc/containerd/config.toml \
    --set toolkit.env[1].name=CONTAINERD_SOCKET \
    --set toolkit.env[1].value=/run/k3s/containerd/containerd.sock \
    --set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
    --set toolkit.env[2].value=nvidia \
    --set toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT \
    --set-string toolkit.env[3].value=true \
     nvidia/gpu-operator
    ```

    The above command creates multiple pods on `gpu-operator` namespace across all nodes in the cluster.

4. Verify the operator pods status.

    ```sh
    kubectl get po -n gpu-operator -o wide
    ```

    You should see an output similar to below

    ```sh
    NAME                                                       READY   STATUS      RESTARTS   AGE   IP           NODE     NOMINATED NODE   READINESS GATES
    nvidiagpu-node-feature-discovery-worker-tdppn              1/1     Running     0          40s   10.42.1.3    xe8545   <none>           <none>
    nvidiagpu-node-feature-discovery-worker-fdqkw              1/1     Running     0          40s   10.42.0.9    xe9680   <none>           <none>
    gpu-feature-discovery-45xcs                                1/1     Running     0          40s   10.42.0.16   xe9680   <none>           <none>
    nvidia-device-plugin-daemonset-qxtf2                       1/1     Running     0          40s   10.42.1.7    xe8545   <none>           <none>
    gpu-feature-discovery-ftfbd                                1/1     Running     0          40s   10.42.1.9    xe8545   <none>           <none>
    nvidia-container-toolkit-daemonset-q4mjx                   1/1     Running     0          40s   10.42.0.12   xe9680   <none>           <none>
    nvidia-device-plugin-daemonset-br4mx                       1/1     Running     0          40s   10.42.0.14   xe9680   <none>           <none>
    nvidia-container-toolkit-daemonset-rnf6d                   1/1     Running     0          40s   10.42.1.5    xe8545   <none>           <none>
    nvidia-mig-manager-rk2vx                                   1/1     Running     0          40s   10.42.1.10   xe8545   <none>           <none>
    nvidia-mig-manager-rhz4h                                   1/1     Running     0          40s   10.42.0.17   xe9680   <none>           <none>
    gpu-operator-7d46fd8c68-lk9fb                              1/1     Running     0          40s   10.42.0.10   xe9680   <none>           <none>
    nvidia-cuda-validator-4s2qb                                0/1     Completed   0          40s   10.42.0.19   xe9680   <none>           <none>
    ```
5. Verify the NVIDIA GPU access to the nodes by describing each node.

    ```sh
    kubectl get nodes
    ```

    ```sh
    kubectl describe nodes xe9680
    ```

    You should see the details of NVIDIA GPUs under **Capacity** section on the output similar to the below for each nodes.
    ```sh
    ...
    Capacity:
    cpu:                224
    ephemeral-storage:  3072261880Ki
    hugepages-1Gi:      0
    hugepages-2Mi:      0
    memory:             2113250420Ki
    nvidia.com/gpu:     8
    pods:               110
    Allocatable:
    ...
    ```

## Network File System (NFS) Setup

### Setting up NFS Server

> *Note: Skip this step if a NFS Server like a [Dell PowerScale devices](https://www.dell.com/en-in/work/shop/powerscale-family/sf/powerscale)  already available*

The Network File System is used to store the dataset and the training checkpoint. The below steps creates a NFS server on one of your nodes.

1. Install the nfs server

    ```sh
    sudo apt-get update
    sudo apt-get install nfs-kernel-server
    ```
2. Create an directory `<Directory Path>` to use as the mount directory.
3. Update the file `/etc/exports` file with directory details as root user.

    ```sh
    <Directory Path> *(rw,sync,no_subtree_check,no_root_squash)
    ```
    > *Note: This provides NFS access to all machine on the network.*

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
