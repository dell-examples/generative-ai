# Created by Scalers AI for Dell Inc.

FROM python:3.10.13
COPY /src/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY src /src
WORKDIR /src
RUN chmod -R 777 /src
EXPOSE 6100/tcp

ENTRYPOINT python3 runner.py
