# Use the base image
#FROM vault.habana.ai/gaudi-docker/1.16.2/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
#FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
FROM vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest

# Set Proxy for git to work
#ENV https_proxy="http://proxy-chain.intel.com:912"
#ENV http_proxy="http://proxy-chain.intel.com:911"
#ENV no_proxy="localhost,127.0.0.1,intel.com"
ENV VLLM_TARGET_DEVICE="hpu"
ENV VLLM_RUNTIME=habana
# Copy apt proxy for apt-get to work
#COPY apt.conf /etc/apt/apt.conf

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
RUN pip install -e .
#RUN python setup.py develop

# Install llmperf
WORKDIR /llmperf
RUN pip install -e .
