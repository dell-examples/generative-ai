# Industrial Metaverse | Broadcom NetXtreme® E-Series BCM57508 Bandwidth Performance

## Table of Contents
* [Introduction](#1-introduction)
* [Test Environment](#2-test-environment)
    * [Hardware Configuration](#21-hardware-configuration)
    * [Software Configuration](#22-software-configuration)
* [Test Scenario](#3-test-scenario)
* [Test Workload Configuration](#4-test-workload-configuration)
* [Test Metrics](#5-test-metrics)
* [Performance Reports](#6-performance-reports)

## 1. Introduction
Scalers AI has developed an Industrial metaverse solution targeting manufacturing use case.

### Objective

* Evaluate the scalability of the solution by segregating the visualization on the Dell™ PowerEdge™ R760xa(without GPUs) server and inference+metaverse services on the Dell™ PowerEdge™ R760xa(with NVIDIA® L40S GPUs) server. Utilize the Broadcom® Ethernet Adapter - Broadcom® NetXtreme® E-Series BCM57508 to transmit the uncompressed video stream for encoding and streaming video WebRTC on the browser.
* Gauge performance and resource utilization across varying numbers of concurrent streams.

## 2. Test Environment

### 2.1 Hardware Configuration

The hardware configuration used for the bandwidth testing is as shown below

<img src="../assets/test_architecture.png" height=400px>

The cluster consists of two Dell PowerEdge servers

| Server | CPU | RAM | Disk | GPUs |
| --- | --- | --- | --- | ---- |
| Dell™ PowerEdge™ R760xa | Intel® Xeon® Platinum 8592+ | 500 GB | 500 GB | 4xNVIDIA® L40S GPUs |
| Dell™ PowerEdge™ R760xa | Intel® Xeon® Platinum 8470Q | 500 GB | 1 TB |  |

Each server is networked to a Dell™  PowerSwitch Z9664F-ON through Broadcom® BCM57508 NICs with 100 Gb/s bandwidth.

### 2.2 Software Configuration

**2.2.1 Docker and NVIDIA® CUDA®**

| Software | Version |
| --- | --- |
| Docker | `24.0.7` |
| NVIDIA® CUDA® | `v12.2` |

**2.2.2 Software Components**

| Software | Version |
| --- | --- |
| NVIDIA®  Isaac Sim | `2023.1.0` |
| DeepStream | `6.4 `|
| Zenoh | `0.10.1rc0` |
| OPC UA | `1.0.6` |
| InfluxDB | `1.7.10` |
| Telegraf | `1.22.3` |


**2.2.3 Model**
| Model | Type |
| --- | --- |
| YOLOv8s | CV-Segmentation |

## 3. Test Scenario

* Test scenario executed with varying numbers of concurrent streams.
* NVIDIA Isaac Sim (metaverse), sensor simulator modules and OPC UA servers deployed in Dell™ PowerEdge™ R760xa - 4x NVIDIA L40S GPUs.
* Visualization services including encoding setup on the Dell™ PowerEdge™ R760xa.
* Each scenario runs for 5 minutes.
* Captured metrics represent an average of 5 minute duration.
* FPS per stream includes decoding, inference, post processing and publishing uncompressed streams over network.

## 4. Test Workload Configuration

Out of 4 NVIDIA® L40S GPUs available on the Dell™ PowerEdge™ R760xa
* NVIDIA® Isaac Sim is deployed on 1 NVIDIA® L40S GPU.
* The remaining 3  NVIDIA® L40S GPUs available are utilized for the performance test.


For the performance testing, the inference pipeline is configured to take input from an RTSP video simulator. The video simulator is configured to stream 6 different video streams recorded from the NVIDIA® Isaac Sim. The video streams simulate output video captured from AMRs deployed on the NVIDIA® Isaac Sim metaverse or those that are deployed in the real world.

Each input stream is 30 FPS with 1080p resolution.

## 5. Test Metrics

The below are the metrics measured for each tests

| Metric | Explanation |
| --- | ------ |
| Average GPU Utilization <br> (Memory, Compute) | Measure GPU memory and compute utilization. |
| Network Bandwidth <br> (Average, Maximum) | Measure efficiency in data transfer with average and maximum network bandwidth. |

## 6. Performance Reports

| GPU SKU | Number of Streams | AVG FPS per Stream | Throughput <br> (Tokens/s) | Avg Bandwidth Util (Gbits/s) | Max Bandwidth Util (Gbits/s) | Avg GPU Util (%) | Avg GPU Memory Util (%) |
| ------ | --- | --- | --- | --- | --- | ----- | ---- |
| 1x NVIDIA® L40S GPU | 1 | 29.98 | 29.98 | 1.56 | 3.14 | 4.9 | 7.76 |
| 1x NVIDIA® L40S GPU | 2 | 29.65 | 59.3 | 3.08 | 6.16 | 5.34 | 7.53 |
| 1x NVIDIA® L40S GPU | 4 | 29.78 | 119.12 | 6.17 | 12.6 | 6.21 | 11.6 |
| 1x NVIDIA® L40S GPU | 6 | 29.65 | 177.9 | 9.25 | 19.5 | 7.04 | 16.6 |
| 2x NVIDIA® L40S GPU | 8 | 29 | 232 | 12.4 | 25.5 | 7.95 | 16.7 |
| 2x NVIDIA® L40S GPU | 10 | 29.7 | 297 | 15.5 | 32.1 | 8.8 | 20.2 |
| 2x NVIDIA® L40S GPU | 12 | 29.8 | 357.6 | 18.5 | 37.8 | 9.62 | 24.8 |
| 3x NVIDIA® L40S GPU | 18 | 29.5 | 531 | 27.7 | 58 | 12.1 | 33.4 |
| 3x NVIDIA® L40S GPU | 24 |  27.4 | 657.6 | 34.3 | 83.9 | 14.69 | 41.8 |

For more detailed report refer [performance_report.xlsx](./performance_report.xlsx)


*Performance varies by use case, model, application, hardware & software configurations, the quality of the resolution of the input data, and other factors. This performance testing is intended for informational purposes and not intended to be a guarantee of actual performance of the application.*
