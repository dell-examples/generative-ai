# Created by Scalers AI for Dell Inc.

FROM python:3.10.6-slim

RUN apt-get update && apt-get install -y python3-opencv libgl1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /src

WORKDIR /src

CMD python3 dashboard.py
