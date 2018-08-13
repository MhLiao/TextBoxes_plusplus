FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
LABEL version="0.1.0"
LABEL version_convention="Semantic Versioning (see www.semver.org)"
LABEL title="Dockerfile for TextBoxes++ with CRNN"
LABEL description="This Dockerfile sets up TextBoxes++\
and CRNN with all their respective dependencies.\
Make sure to run with nvidia-docker"

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        autoconf-archive \
        automake \
        binutils-dev \
        bison \
        build-essential \
        cmake \
        curl \
        flex \
        g++ \
        git \
        libatlas-base-dev \
        libboost-all-dev \
        libdouble-conversion-dev \
        libedit-dev \
        libevent-dev \
        libgeos-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libjemalloc-dev \
        libkrb5-dev \
        libleveldb-dev \
        liblmdb-dev \
        liblz4-dev \
        liblzma-dev \
        libmatio-dev \
        libnuma-dev \
        libopencv-dev \
        libprotobuf-dev \
        libpython3-dev \
        libpython-dev \
        libsasl2-dev \
        libsnappy-dev \
        libssl-dev \
        libtool \
        pkg-config \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-opencv \
        python-pip \
        python-scipy \
        wget \
        zlib1g-dev && \
        rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT


########################################
#### Install CRNN Dependencies #########
########################################

# Install pytorch
WORKDIR /root
RUN git clone https://github.com/torch/distro.git /root/torch --recursive
RUN cd /root/torch && \
    bash install-deps;
RUN cd /root/torch && \
    ./install.sh

# Update the PATHs
# This is normally done by calling
#     source ~/torch/install/bin/torch-activate
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install Folly
RUN git clone -b v0.35.0  --depth 1 https://github.com/facebook/folly
RUN cd /root/folly/folly && \
    autoreconf -ivf && \
    ./configure && \
    make && \
    make install && \
    ldconfig 

# Install fbthrift
RUN git clone -b v0.24.0  --depth 1 https://github.com/facebook/fbthrift
COPY patches/fbthrift.diff ${CAFFE_ROOT}/patches/fbthrift.diff
RUN cd /root/fbthrift && \
    git apply ${CAFFE_ROOT}/patches/fbthrift.diff
RUN cd /root/fbthrift/thrift && \
    autoreconf -ivf && \
    ./configure && \
    make && \
    make install

# Install thpp
RUN git clone -b v1.0 https://github.com/facebook/thpp
COPY patches/thpp.diff ${CAFFE_ROOT}/patches/thpp.diff
RUN cd /root/thpp && \
    git apply ${CAFFE_ROOT}/patches/thpp.diff
RUN cd /root/thpp/thpp && \
    ./build.sh

# Install fblualib
RUN git clone -b v1.0 https://github.com/facebook/fblualib
RUN cd /root/fblualib/fblualib && \
    ./build.sh

########################################
#### Install CRNN ######################
########################################
COPY crnn/src ${CAFFE_ROOT}/crnn/src

# Install CRNN
RUN cd ${CAFFE_ROOT}/crnn/src && \
    ./build_cpp.sh

########################################
#### Install Textboxes++ Dependencies ##
########################################
WORKDIR $CAFFE_ROOT

# Cython needs to be installed seperately
Copy python ${CAFFE_ROOT}/python
RUN pip install Cython==0.28.5
RUN pip install -r python/requirements.txt

########################################
#### Install Textboxes++ ###############
########################################
WORKDIR $CAFFE_ROOT
# Include Build context
COPY caffe.cloc ${CAFFE_ROOT}/caffe.cloc
Copy cmake ${CAFFE_ROOT}/cmake
Copy CMakeLists.txt ${CAFFE_ROOT}/CMakeLists.txt
Copy include ${CAFFE_ROOT}/include 
Copy matlab ${CAFFE_ROOT}/matlab
Copy src ${CAFFE_ROOT}/src
COPY tools ${CAFFE_ROOT}/tools
COPY LICENSE ${CAFFE_ROOT}/LICENSE
COPY scripts ${CAFFE_ROOT}/scripts

# CUDA_ARCH_NAME=Manual is a workaround the lack of compute_60 or higher in cuda7.5's cuda
# Required for recent GPUs
RUN mkdir build && \
    cd build && \
    cmake .. -DCUDA_ARCH_NAME=Manual && \
    make -j"$(nproc)"

# HACK: OpenCV can be confused by (the lack of) this driver in some systems
RUN ln /dev/null /dev/raw1394

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig


########################################
#### COPY Context required at runtime ##
########################################
COPY crnn/data ${CAFFE_ROOT}/crnn/data
COPY data ${CAFFE_ROOT}/data
COPY demo_images ${CAFFE_ROOT}/demo_images
COPY examples ${CAFFE_ROOT}/examples
COPY README.md ${CAFFE_ROOT}/README.md


WORKDIR /opt/caffe
