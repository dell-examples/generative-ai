# `Telegraf, InfluxDB, GRAFANA(TIG) Pipeline` - Solution for monitoring GPU/CPU while training AI/ML workloads 

Quick solution for benchmarking GPU/CPU while training heavy models, without that big of a hardware utilization footprint

### Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Prerequisites

Make sure you have the following packages:

```bash
Docker version 19.03 or greater
Nvidia container toolkit
Docker compose
```


## Running Solution

For a user:

```bash
sudo docker compose up -d
```

## Opening influxdb

In order to log in to influxdb, the username and password can be found in the .env file

## Running on WSL2:

It is extremely important that you have docker desktop on your system, and enable it for your WSL Distro to get the telegraf container to recognize the nvidia drivers. 
