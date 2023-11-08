FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

#get deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
    libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev \
    python3-setuptools vim libgtk2.0-dev libgtk-3-dev build-essential cmake mysql-client

# Downgrade pip to ~=19.0
RUN pip3 install pip~=19.0

# First install pytorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install openmmlab packages

WORKDIR /openmmlab

# Install mim (mim is the package manager for openmmlab)
RUN pip3 install -U openmim

RUN mim install mmengine

# install mmcv
RUN mim install mmcv

# install mmdetection
RUN mim install mmdet

# RUN mim install mmpretrain
RUN pip3 install motmetrics


# ####### Now set up PosePipe itself
WORKDIR /PosePipe
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY setup.py README.md /PosePipe/
COPY pose_pipeline /PosePipe/pose_pipeline
RUN pip3 install -e /PosePipe

COPY datajoint_config.json /root/.datajoint_config.json
COPY scripts/*.py /PosePipe/
COPY demo /PosePipe/demo_det/

# Creating directory for pose projects
WORKDIR /
RUN mkdir -p /projects/pose/
RUN mkdir -p /videos

# CMD ["python"]
# ENTRYPOINT ["/bin/bash"]
# ENTRYPOINT ["/usr/bin/python3"]
CMD ["sh", "-c", "while true; do sleep 1000; done"]
