# Created by Scalers AI for Dell Inc.
# Dockerfile for sensor data analytics service

FROM telegraf:1.29.4

ENV DEBIAN_FRONTEND noninteractive
USER root

RUN apt-get update && \
    apt-get install -y -q --no-install-recommends python3 \
    python3-pip libgl1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --break-system-packages

COPY ./model /model
COPY ./src /app/
COPY ./telegraf.conf /etc/telegraf/

ENTRYPOINT ["/bin/bash", "run.sh"]
