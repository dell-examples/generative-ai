# Created by scalers.ai for Dell
# Container for running the chatbot
FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /code

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT python3 chatbot.py
