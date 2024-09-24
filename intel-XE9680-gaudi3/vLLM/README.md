### To run this sample example of vLLM-benchmark:

•   Steps To build and push docker images to the private registry for vLLM Backend and Benchmark: 

    ```bash
    # 1. build vLLM Backend Docker Images
    # Replace dockerregistryIP with docker private registry IP address
    cp Dockerfile-vllm-backend Dockerfile
    docker build -t dockerregistryIP:5500/vllm_backend .
    docker push dockerregistryIP:5500/vllm_backend 
    
    # 2. build vLLM Benchmark Docker Images
    # Replace dockerregistryIP with docker private registry IP address
    cp Dockerfile-vllm-benchmark Dockerfile
    docker build -t dockerregistryIP:5500/vllm_benchmark .
    docker push dockerregistryIP:5500/vllm_benchmark:latest
    
    # 3. Validate images are in the private registry
    docker images
    ```

•   For this example of running Llama-3-8B model, create a HuggingFace token with access to the [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository. 

•   Using the token created above, setup a kubernetes secret, which will be used by the configmap and job yaml files.  
    ``` bash 
    kubectl create secret generic hf-token --from-literal=HF_TOKEN=hf_your_hf_token
    ```

•   Check and edit path of volumes in configmap, deployment and job.yaml files. Additionally, model and resources can be tuned accordingly. 

•   Replace dockerregistryIP in deployment and job YAML files to match images built above.

•   Create Kubernetes resources:
    ```bash
    kubectl create –f configmap.yaml
    kubectl create –f service.yaml
    kubectl create –f deployment.yaml
    kubectl create –f job.yaml
    ```

•   Observe the status of the deployment and job:
    ```bash
    kubectl logs deployments/vllm-f
    kubectl logs jobs/vllm-benchmark -f
    ```

•   Clean up environment:
    ```bash
    kubectl delete –f deployment.yaml
    kubectl delete –f service.yaml
    kubectl delete –f configmap.yaml
    kubectl delete –f job.yaml
    ```
    
•   Delete token (if needed) when testing is complete:
    ```bash
    kubectl delete secret hf-token
    ```
