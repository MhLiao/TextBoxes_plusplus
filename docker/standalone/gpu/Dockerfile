FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libgeos-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        python-opencv && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

# Set this with `docker build --build-arg CLONE_REPO=$(git remote get-url --all origin) .`
# Note that will only work for https urls since ssh is not installed in the image

ARG CLONE_REPO

RUN git clone -b ${CLONE_TAG} --depth 1 $CLONE_REPO .

# CUDA_ARCH_NAME=Manual is a workaround the lack of compute_60 or higher in cuda7.5's cuda
# Required for recent GPUs
RUN for req in $(cat python/requirements.txt); do pip install $req; done && \
    mkdir build && cd build && \
    cmake .. -DCUDA_ARCH_NAME=Manual && \
    make -j"$(nproc)"

# HACK: OpenCV can be confused by (the lack of) this driver in some systems
RUN ln /dev/null /dev/raw1394

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /opt/caffe
