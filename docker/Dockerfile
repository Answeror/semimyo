FROM answeror/mxnet:0.9.3-cuda8
MAINTAINER answeror <answeror@gmail.com>

RUN apt-get -qqy update && apt-get install -y python-pip python-scipy
RUN mkdir ~/.pip && \
  echo "[global]" > ~/.pip/pip.conf && \
  echo "index-url = https://pypi.mirrors.ustc.edu.cn/simple" >> ~/.pip/pip.conf
RUN pip install click==6.6 logbook==1.0.0 joblib==0.10.3 nose==1.3.7 lru-dict==1.1.6 scikit-learn==0.18.1

ADD mxnet/tensor_gpu-inl.cuh /mxnet/mshadow/mshadow/cuda/tensor_gpu-inl.cuh
RUN cd /mxnet && make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs

ADD mxnet/dropout-inl.h /mxnet/src/operator/dropout-inl.h
RUN cd /mxnet && make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs

ADD mxnet/resource.cc /mxnet/src/resource.cc
RUN cd /mxnet && make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs

WORKDIR /code
