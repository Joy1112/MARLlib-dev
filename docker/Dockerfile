FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY ./ /workingspaces/marllib

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-key adv --keyserver hkp://pgp.mit.edu:80 --recv-keys A4B469963BF863CC && \
    apt-get update --fix-missing
    
RUN apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils ca-certificates nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

RUN python3 -m pip install pip==24.0.0 && pip install marllib

RUN pip install protobuf==3.17 gym==0.20.0 Cython==3.0.0a10

# mamujoco installation
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev unzip
RUN pip install patchelf-wrapper
RUN mkdir /root/.mujoco && cd /root/.mujoco && \
    wget https://www.roboti.us/file/mjkey.txt && \
    wget https://roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && mv mujoco200_linux mujoco200 && \
    echo "LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin;" > ~/.bashrc && \
    pip install mujoco-py==2.0.2.8 && \
    cd /root && git clone https://github.com/schroederdewitt/multiagent_mujoco && \
    cd multiagent_mujoco && mv multiagent_mujoco /workingspaces/marllib/multiagent_mujoco

WORKDIR /workingspaces/marllib
