# Created by Scalers AI for Dell Inc.

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/simulator.py .

CMD [ "python", "simulator.py" ]
