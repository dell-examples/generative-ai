# KubeRay Installation and Kubernetes Secrets

**Estimated Time: 10 mins ⏱️**


In this section, we will be streamlining the integration of KubeRay and will create Kubernetes secrets for handling docker images.

## Table of Contents

* [KubeRay Installation](#kuberay-installation)
* [Creating Kubernetes Secrets](#creating-image-pull-secret-for-kubernetes)

## KubeRay Installation

Follow the below steps to deploy the [KubeRay Operator v1.0.0](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) on your k3s cluster.

1. Add the kuberay repo to helm.

    ```sh
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    ```
2. Install both CRDs and KubeRay operator v1.0.0.

    ```sh
    helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0
    ```
3. Verify the kuberay operator pod deployment.

    ```sh
    kubectl get pods
    ```

    You should see an output similar to this.

    ```sh
    NAME                                READY   STATUS    RESTARTS   AGE
    kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          27s
    ```


## Creating Image Pull Secret for Kubernetes

> [!NOTE]
> This step may vary based on different container registries.

The Image Pull Secret should be created for the cluster to pull images from the Azure container registry.

1. Collect the below details from your container registry

    | Name | Details | Example |
    | --- | --- | --- |
    | `CR Server` | Container registry server URI. | `infer.cr.io` |
    | `CR User Name` | Container registry user name | `user` |
    | `CR Password` | Container registry password | `password123` |

2. Create the secret on your k3s cluster.

    ```sh
    kubectl create secret docker-registry cr-login --docker-server=<CR Server> --docker-username=<CR User Name> --docker-password=<CR Password>
    ```

[Back to Deployment Guide](../README.md#deployment-guide)
