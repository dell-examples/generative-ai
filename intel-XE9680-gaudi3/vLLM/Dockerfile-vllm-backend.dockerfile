FROM vault.habana.ai/gaudi-docker/1.17.1/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
ENV VLLM_TARGET_DEVICE="hpu"
ENV VLLM_RUNTIME=habana

# Install git and other dependencies
RUN apt-get update && apt-get install -y git
 
RUN pip install transformers huggingface_hub ray openai
# Clone llmperf repository
RUN git clone https://github.com/ray-project/llmperf.git /llmperf
 
# Clone Habana's vLLM fork
RUN git clone https://github.com/HabanaAI/vllm-fork.git /vllm-fork
 
# Install Habana's vLLM fork
WORKDIR /vllm-fork
RUN git checkout habana_main
#RUN pip install -e .
RUN python setup.py develop
 
# Install llmperf
WORKDIR /llmperf
RUN pip install -e .
