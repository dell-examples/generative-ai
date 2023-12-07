# Created by Scalers AI for Dell Inc.

FROM python:3.10.13
COPY src /src
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
WORKDIR /src
CMD python3 core.py 