FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

#get deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
    libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev \
    python3-setuptools vim libgtk2.0-dev libgtk-3-dev build-essential cmake mysql-client

####### If using OpenPose need these next sections

# get openpose
WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

# build it
WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=ON -DUSE_CUDNN=ON  .. && make -j `nproc`
WORKDIR /openpose

# install Openpose python. note the python version should match the base image
WORKDIR /openpose/build/python/openpose
RUN make install
RUN ls -l /openpose/build/python/openpose
RUN cp ./pyopenpose.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.10/dist-packages
WORKDIR /usr/local/lib/python3.8/dist-packages
RUN ln -s pyopenpose.cpython-310m-x86_64-linux-gnu.so pyopenpose
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages:${LD_LIBRARY_PATH}"

# download models
WORKDIR /openpose/models
RUN sh getModels.sh

# no path needed since installed in the LD_LIBRARY_PATH
ARG OPENPOSE_PATH=""
RUN mv /openpose/models /usr/local/lib/python3.10/dist-packages/../../../models/

# ####### Now set up PosePipe itself
WORKDIR /PosePipe
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY setup.py README.md /PosePipe/
COPY pose_pipeline /PosePipe/pose_pipeline
RUN pip3 install -e /PosePipe

COPY datajoint_config.json /root/.datajoint_config.json
COPY scripts/*.py /PosePipe/

# Creating directory for pose projects
WORKDIR /
RUN mkdir -p /projects/pose/

# CMD ["python"]
# ENTRYPOINT ["/bin/bash"]
# ENTRYPOINT ["/usr/bin/python3"]
CMD ["sh", "-c", "while true; do sleep 1000; done"]
