# Created by Scalers AI for Dell Inc.

FROM python:3.10.6-slim

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src /src

WORKDIR /src

CMD python3 app.py & python3 dashboard.py