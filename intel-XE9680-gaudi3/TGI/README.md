### To run this sample example of tgi-benchmark:

•   For this example of running Llama-3-8B model, create a HuggingFace token with access to the [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository. 

•   Using the token created above, setup a kubernetes secret, which will be used by the configmap and job yaml files.  
    ``` bash 
    kubectl create secret generic hf-token --from-literal=HF_TOKEN=hf_your_hf_token
    ```

•   Check and edit path of volumes in configmap, deployment and job.yaml files. Additionally, model and resources can be tuned accordingly. 

•   Create Kubernetes resources:
    ```bash
    kubectl create –f configmap.yaml
    kubectl create –f service.yaml
    kubectl create –f deployment.yaml
    kubectl create –f job.yaml
    ```

•   Observe the status of the deployment and job:
    ```bash
    kubectl logs deployments/tgi -f
    kubectl logs jobs/tgi-benchmark -f
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
