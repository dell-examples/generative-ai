# Created by scalers.ai for Dell
# Llama 2 Fine-Tuning Image

FROM nvcr.io/nvidia/pytorch:23.08-py3

USER root

RUN apt-get update

WORKDIR /code

COPY requirements.txt .

RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src /code
