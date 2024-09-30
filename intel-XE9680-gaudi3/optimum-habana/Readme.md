### To run this sample example of habana-optimum:

• For this example of running Llama-3-8B model, create a HuggingFace token with access to the [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository. 

• Using the token created above, setup a kubernetes secret, which will be used by the configmap and job yaml files.  

    kubectl create secret generic hf-token --from-literal=HF_TOKEN=hf_your_hf_token

• Check and edit path of volumes in configmap and job.yaml files. Additionally, model and resources can be tuned accordingly. 

•   Create Kubernetes resources:
    kubectl create –f configmap.yaml
    kubectl create –f job.yaml

•   Observe the status of the job:
    kubectl logs jobs/optimum-benchmark -f


•   Clean up environment:
    kubectl delete –f configmap.yaml
    kubectl delete –f job.yaml

    
•   Delete token (if needed) when testing is complete:
    kubectl delete secret hf-token
