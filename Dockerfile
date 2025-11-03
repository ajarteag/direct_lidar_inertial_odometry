FROM ros:humble

RUN apt-get update \
   # Install C++ dependencies
   && apt-get install -y --no-install-recommends libomp-dev libpcl-dev libeigen3-dev \
   # Install ROS dependencies
   && apt-get install -y --no-install-recommends ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-nav-msgs ros-humble-pcl-ros \
   # Install Nvidia CUDA Toolkit 12.6 (same as JetPack 6.2)
   && apt-get install gnupg2 curl ca-certificates \
   && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
   && dpkg -i cuda-keyring_1.1-1_all.deb \
   && apt-get update \
   && apt-get install -y --no-install-recommends \
   	cuda-cudart-12-6 \
   	cuda-compat-12-6 \
   	cuda-cudart-dev-12-6 \
    	cuda-command-line-tools-12-6 \
    	cuda-minimal-build-12-6 \
    	cuda-libraries-dev-12-6 \
    	cuda-nvml-dev-12-6 \
    	libnpp-dev-12-6 \
    	libcusparse-dev-12-6 \
    	libcublas-dev-12-6 \
