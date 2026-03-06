ARG BASE_IMAGE
FROM $BASE_IMAGE

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-geographic-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install the python package
COPY . /opt/NavigationScaling
WORKDIR /opt/NavigationScaling
RUN pip3 install -e .

# Install python deps
RUN pip3 install hydra-core \
    transformers \
    einops \
    diffusers \
    torchcfm
    
RUN pip3 install git+https://github.com/bdaiinstitute/theia.git --no-deps

COPY docker/ros_entrypoint.sh /opt/ros_entrypoint.sh
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/opt/ros_entrypoint.sh", "bash"]