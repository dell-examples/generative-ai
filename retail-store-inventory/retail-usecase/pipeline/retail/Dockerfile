# Created by Scalers AI for Dell Inc.

FROM python:3.11
RUN apt-get -y update && apt-get install -y libgl1 zip


COPY install_zendnn.sh install_zendnn.sh
RUN bash install_zendnn.sh

COPY requirements.txt requirements.txt
RUN pip3 install --default-timeout=100 -r requirements.txt

COPY src /src
WORKDIR /src
CMD bash run.sh
