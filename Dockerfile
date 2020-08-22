FROM ubuntu:18.04

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=0.12.1 
ARG TENSORFLOW_ARCH=cpu
ARG KERAS_VERSION=1.2.0
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest
ARG CAFFE_VERSION=master

# Install some dependencies
# libjasper-dev \
# libopenjpeg2 \
# libpng12-dev \
RUN apt-get update && apt-get install -y \
    bc \
    build-essential \
    cmake \
    curl \
    g++ \
    gfortran \
    git \
    libffi-dev \
    libfreetype6-dev \
    libhdf5-dev \
    libjpeg-dev \
    liblcms2-dev \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libtiff5-dev \
    libwebp-dev \
    libzmq3-dev \
    nano \
    pkg-config \
    python-dev \
    software-properties-common \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    qt5-default \
    libvtk6-dev \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libgdal-dev \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev \
    libtbb-dev \
    libeigen3-dev \
    python-dev \
    python-tk \
    python-numpy \
    python3-dev \
    python3-tk \
    python3-numpy \
    ant \
    default-jdk \
    doxygen \
    && \
  apt-get clean && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
  update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
    pyopenssl \
    ndg-httpsclient \
    pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
    python-numpy \
    python-h5py \
    python-matplotlib \
    && \
  apt-get clean && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/*


# Install TensorFlow
RUN pip --no-cache-dir install \
  https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl


# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}


# Expose Ports for TensorBoard (6006)
EXPOSE 6006


# Adding the project files in the container
COPY . /


WORKDIR "/root"


CMD ["python", "./TensorGUI"]