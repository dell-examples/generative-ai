# Created by Scalers AI for Dell

FROM nvcr.io/nvidia/deepstream:6.3-gc-triton-devel

RUN apt-get update && \
    apt-get install -y -q --no-install-recommends python3-pip \
    ffmpeg wget \
    python3-gi python3-gst-1.0 python-gi-dev git \
    python3  cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev \
    libavcodec-dev libavformat-dev libavdevice-dev ffmpeg unzip \
    libglib2.0  \
    libjansson4 libjansson-dev gobject-introspection

RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_x86_64.whl && \
    python3 -m pip install pyds-1.1.8-py3-none-linux_x86_64.whl

RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY src /src
WORKDIR /src

ENTRYPOINT ["python3", "pipeline.py", "--config", "/config/metaverse_config.yaml"]
