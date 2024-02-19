# Created by Scalers AI for Dell Inc.

FROM python:3.10.13
RUN apt-get -y update && apt-get install -y libgl1 ffmpeg
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY src /src
WORKDIR /src
CMD python3 receive_streams.py --config config/config.yaml
