# Created by Scalers AI for Dell Inc.

FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

RUN python3 -m pip install opencv-python==4.9.0.80

WORKDIR /src

COPY src /src

ENTRYPOINT ["python3", "publisher.py"]
