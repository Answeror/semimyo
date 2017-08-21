FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER answeror <answeror@gmail.com>

RUN echo "deb http://ftp.sjtu.edu.cn/ubuntu/ trusty main restricted universe multiverse" > /etc/apt/sources.list && \
  echo "deb http://ftp.sjtu.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://ftp.sjtu.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://ftp.sjtu.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://ftp.sjtu.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://ftp.sjtu.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://ftp.sjtu.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://ftp.sjtu.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://ftp.sjtu.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://ftp.sjtu.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
  apt-get -qqy update

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  libopenblas-dev \
  libopencv-dev \
  python-numpy \
  wget \
  unzip
RUN git clone --recursive https://github.com/dmlc/mxnet/ && cd mxnet && \
  git checkout v0.9.3 && \
  git submodule update --init && \
  cp make/config.mk . && \
  echo "USE_CUDA=1" >>config.mk && \
  echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk && \
  echo "USE_CUDNN=1" >>config.mk && \
  echo "USE_BLAS=openblas" >>config.mk
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH 
ADD tensor_gpu-inl.cuh /mxnet/mshadow/mshadow/cuda/tensor_gpu-inl.cuh
RUN cd /mxnet && make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs

ENV PYTHONPATH /mxnet/python
