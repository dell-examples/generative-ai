FROM python:3.10

ENV PIP_DEFAULT_TIMEOUT=100
ARG CUDA_VISIBLE_DEVICES="0"
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
ARG CUDA_DEVICE_ORDER="PCI_BUS_ID"
ENV CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}
ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
EXPOSE ${GRADIO_SERVER_PORT}

WORKDIR /workspace

# Install dependencies
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ /workspace
COPY images/ /workspace/images/ 

# Populate local environment variables
# COPY ./.env .
# RUN source .env && printenv # To debug, uncomment this line to see env vars in the build logs
# ENV $(cat .env)

CMD ["python", "/workspace/rag-mistral/main.py"]

