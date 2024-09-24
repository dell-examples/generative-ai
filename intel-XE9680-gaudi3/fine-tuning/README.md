### To run the example of fine-tuning & inference

For this example of running Llama-3-8B or Llama-3-70B model, create a HuggingFace token with access to the [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository. One token will suffice to access both models.

•   Using the token created above, setup a kubernetes secret, which will be used by the configmap and job yaml files.  
    ``` bash 
    kubectl create secret generic hf-token --from-literal=HF_TOKEN=hf_your_hf_token
    ```
•   There are 2 examples for fine-tuning, one with Llama3-8B model, and other with Llama-3-70B model. Check and edit path of volumes in configmap and job.yaml files. Additionally, model and resources can be tuned accordingly. 

Below steps apply to running both models: 

•   Create Kubernetes resources:
    ```bash
    kubectl create –f configmap.yaml
    kubectl create –f job.yaml
    ```
•   Observe the status of the job:
    ```bash
    kubectl logs jobs/fine-tuning-llama -f
    ```
•   Clean up environment:
    ```bash
    kubectl delete –f configmap.yaml
    kubectl delete –f job.yaml
    ```
•   Delete token (if needed) when testing is complete:
    ```bash
    kubectl delete secret hf-token
    ```
