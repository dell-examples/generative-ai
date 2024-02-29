# Ray Cluster Configuration

The [Ray cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html#ray-cluster) is defined by the cluster deployment yaml (e.g. [cluster.nvidia.yaml](../serving/gpu/cluster.nvidia.yml)).

This section provides details into the configurations available to customize the cluster deployment according to your hardware availability.

The cluster configuration can be divided into two sections.
1. Cluster deployments
2. Cluster Services

## Cluster Deployment

The cluster configuration utilizes custom CRDs to enable ray clusters. The custom CRDs will be applied to your Kubernetes cluster during the [KubeRay installation](./kuberay_setup.md#kuberay-installation).

The cluster consists of two group pod configurations
1. Head Group (`headGroupSpec`)
2. Worker Group (`workerGroupSpecs`)

The head group defines the head node on the deployed ray cluster. All other nodes are deployed under one or more worker groups.

Some of the parameters required under a group is covered below.

```yaml
rayStartParams:
    dashboard-host: "0.0.0.0" # dashboard host IP
    ...
template: # Pod template
    metadata: # Pod metadata
    labels:
        name: ray-head # label for adding node port service
    spec: # Pod spec
        containers:
        - name: ray-head
            image: infer.cr.io/infer:latest # Serving docker image
            resources:
                limits:
                    cpu: 50 # Number of CPU cores allocated
                    memory: 500Gi # Memory allocated
                    ...
                requests:
                    cpu: 14
                    memory: 500Gi
                    ...
            # Keep this preStop hook in each Ray container config.
            lifecycle:
                preStop:
                    exec:
                    command: ["/bin/sh","-c","ray stop"]
            volumeMounts: # volume mounts for model files
                - name: nfs-pv
            mountPath: /models
    volumes: # volumes to mount the model files
        - name: nfs-pv
            persistentVolumeClaim:
            claimName: nfs-pvc
    imagePullSecrets:
        - name: cr-login # Image pull secret
```


> [!TIP]
> To know more about the Ray Cluster Configurations on Kubernetes, refer [RayCluster Configuration page](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html#kuberay-config)


<details>
<summary>More on the NVIDIA® + AMD GPU Cluster</summary>

The below cluster configuration consists of a head group (NVIDIA® GPUs access) and 3 workers (NVIDIA® + AMD GPUs access).

The head node is configured to be utilized as a worker pod.

```yaml
rayClusterConfig:
    rayVersion: '2.8.1' # ray version
    # Ray head pod template.
    headGroupSpec: # head group spec
      rayStartParams:
        dashboard-host: '0.0.0.0'
        # setting `num-gpus` on the rayStartParams enables
        # head node to be used as a worker node
        num-gpus: "8" # using head as the worker node
      # Pod template
      template: # pod template
        metadata: # pod metadata
          labels:
            name: ray-gpu-head # label for adding node port service
        spec:
          runtimeClassName: nvidia # runtime class to enable NVIDIA GPU access
          containers:
          - name: ray-gpu-head # head pod name
            image: infer.cr.io/nvidia-gpu:latest # docker image for head pod
            ports: # Ray and Ray Serve configurable ports
            - containerPort: 6379
              name: gcs
            - containerPort: 8265
              name: dashboard
            - containerPort: 10001
              name: client
            - containerPort: 8000
              name: serve
            resources:
              limits:
                cpu: "160" # Number of CPU cores allocated
                memory: "300G" # System Memory allocated
                nvidia.com/gpu: 8 # NVIDIA GPUs allocated
              requests:
                cpu: "160"
                memory: "300G"
                nvidia.com/gpu: 8
            volumeMounts:
              - name: nfs-pv-demo  # volumes to mount the model files
                mountPath: /models
              - mountPath: /tmp/ray # volume for ray logs
                name: ray-logs
          volumes:
            - name: ray-logs
              emptyDir: {}
            - name: nfs-pv-demo
              persistentVolumeClaim:
                claimName: nfs-pvc-demo
          imagePullSecrets: # docker image pull secret
            - name: cr-login
          nodeSelector:
              kubernetes.io/hostname: xe9680 # select xe9680 as head node
    workerGroupSpecs: # worker groups specs
    - replicas: 2 # number of replicas to be deployed
      minReplicas: 2
      maxReplicas: 2
      groupName: gpu-nvidia # nvidia worker group
      rayStartParams:
        num-gpus: "4" # number of GPUs available
      template:
        spec:
          runtimeClassName: nvidia # runtime class to enable NVIDIA GPU access
          containers:
          - name: ray-worker
            image: infer.cr.io/nvidia-gpu:latest
            resources:
              limits:
                cpu: "160"
                memory: "300G"
                nvidia.com/gpu: "4" # number of NVIDIA GPUs allocated
              requests:
                cpu: "160"
                memory: "300G"
                nvidia.com/gpu: "4"
            volumeMounts:
              - name: nfs-pv-demo
                mountPath: /models
          volumes:
            - name: nfs-pv-demo
              persistentVolumeClaim:
                claimName: nfs-pvc-demo
          imagePullSecrets:
            - name: cr-login
    - replicas: 1
      minReplicas: 1
      maxReplicas: 1
      groupName: gpu-amd # AMD GPU group
      rayStartParams:
        num-gpus: "1" # number of gpus available
      template:
        spec:
          containers:
          - name: ray-worker
            image:infer.cr.io/amd-gpu:latest
            resources:
              limits:
                cpu: "10"
                memory: "100G"
                amd.com/gpu: 1 # number of AMD GPUs allocated
              requests:
                cpu: "10"
                memory: "100G"
                amd.com/gpu: 1 # number of AMD GPUs allocated
            volumeMounts:
              - name: nfs-pv-demo
                mountPath: /models
          volumes:
            - name: nfs-pv-demo
              persistentVolumeClaim:
                claimName: nfs-pvc-demo
          imagePullSecrets:
            - name: cr-login
```

</details>


<details>
<summary>More on the Intel® CPU + AMD CPU Cluster</summary>

The below cluster configuration consists of a head group (4th Gen Intel® Xeon® Scalable Processors server) and 3 workers (One 4th Gen Intel® Xeon® Scalable Processors servers and two 4th Generation AMD EPYC™ Processors servers).

The head node is configured to be utilized as a worker pod.

```yaml
rayClusterConfig:
    rayVersion: '2.8.1'
    # Ray head pod template.
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
        num-cpus: "220" # using head as the worker node
      # Pod template
      template:
        metadata:
          labels:
            name: ray-cpu-head
        spec:
          containers:
          - name: ray-cpu-head
            image: infer.cr.io/cpu_intel:latest
            volumeMounts:
              - mountPath: /tmp/ray
                name: ray-logs
              - name: nfs-pv-demo
                mountPath: /models
            resources:
              limits:
                cpu: "220"
                memory: "800G"
              requests:
                cpu: "220"
                memory: "800G"
          volumes:
            - name: ray-logs
              emptyDir: {}
            - name: nfs-pv-demo
              persistentVolumeClaim:
                claimName: nfs-pvc-demo
          imagePullSecrets:
            - name: cr-login
          nodeSelector:
              feature.node.kubernetes.io/cpu-model.vendor_id: Intel # Targetting the head node as Intel CPU server
              # kubernetes.io/hostname: xe9680
    workerGroupSpecs:
      - replicas: 1
        minReplicas: 1
        maxReplicas: 1
        groupName: cpu-amd
        rayStartParams:
          num-cpus: "256"
          resources: '"{\"amd_cpu\": 10}"' # custom ray resources to enable targetting serve deployment to AMD CPU servers
        template:
          spec:
            containers:
            - name: cpu-amd
              image: infer.cr.io/cpu_amd:latest
              resources:
                limits:
                  cpu: "256"
                  memory: "800G"
                requests:
                  cpu: "256"
                  memory: "800G"
              volumeMounts:
                - name: nfs-pv-demo
                  mountPath: /models
            volumes:
              - name: nfs-pv-demo
                persistentVolumeClaim:
                  claimName: nfs-pvc-demo
            imagePullSecrets:
              - name: cr-login
            nodeSelector:
              feature.node.kubernetes.io/cpu-model.vendor_id: AMD # Targetting the head node as AMD CPU server
              # kubernetes.io/hostname: xe8545
      - replicas: 1
        minReplicas: 1
        maxReplicas: 1
        groupName: cpu-intel
        rayStartParams:
          num-cpus: "224"
        template:
          spec:
            containers:
            - name: cpu-intel
              image: infer.cr.io/cpu_intel:latest
              resources:
                limits:
                  cpu: "224"
                  memory: "800G"
                requests:
                  cpu: "224"
                  memory: "800G"
              volumeMounts:
                - name: nfs-pv-demo
                  mountPath: /models
            volumes:
              - name: nfs-pv-demo
                persistentVolumeClaim:
                  claimName: nfs-pvc-demo
            imagePullSecrets:
              - name: cr-login
            nodeSelector:
              feature.node.kubernetes.io/cpu-model.vendor_id: Intel # Targetting the head node as Intel CPU server
              # kubernetes.io/hostname: user
      - replicas: 1
        minReplicas: 1
        maxReplicas: 1
        groupName: cpu-amd-2
        rayStartParams:
          num-cpus: "128"
          resources: '"{\"amd_cpu\": 10}"' # custom ray resources to enable targetting serve deployment to AMD CPU servers
        template:
          spec:
            containers:
            - name: cpu-amd-2
              image: infer.cr.io/cpus:t1
              resources:
                limits:
                  cpu: "128"
                  memory: "800G"
                requests:
                  cpu: "128"
                  memory: "800G"
              volumeMounts:
                - name: nfs-pv-demo
                  mountPath: /models
            volumes:
              - name: nfs-pv-demo
                persistentVolumeClaim:
                  claimName: nfs-pvc-demo
            imagePullSecrets:
              - name: cr-login
            nodeSelector:
              feature.node.kubernetes.io/cpu-model.vendor_id: AMD # Targetting the head node as AMD CPU server
              # kubernetes.io/hostname: 7625-amd
```
</details>

## Cluster Services

We will be enabling access to cluster dashboards and serving endpoints ports using Kubernetes Service.

The Ray Dashboard will be enabled through mapping head pod `ContainerPort` `8265` to `NodePort` 30265. Similarly the Serving endpoint is mapped from `8000` to `30800`.

```yaml
apiVersion: v1
kind: Service # Kubernetes service definition
metadata:
  name: ray-head-dashboard-port
spec:
  selector:
    name: ray-gpu-head # selecting the head pod to enable node port
  type: NodePort
  ports:
  - port: 8265
    name: "dashboard"
    targetPort: 8265
    nodePort: 30265 # update the ray dashboard port here
  - port: 8000
    name: "endpoint"
    targetPort: 8000
    nodePort: 30800 # update the inferene endpoint port here
```

These ports (`NodePort`) can be configured from ports ranging from `30000` to `32767`.

> [!TIP]
> Refer [Kubernetes Service](https://kubernetes.io/docs/concepts/services-networking/service/) to know more about creating and configuring services.

[Back to Deployment Guide](../README.md#deployment-guide)
