#!/bin/bash

DOCKER_IMAGE_NAME="earthrovers_deployment"
TAG="0.0.1"

DOCKER_IMAGE_ID="${DOCKER_IMAGE_NAME}:${TAG}"
echo "Using $DOCKER_IMAGE_ID"

SCRIPT=$(readlink -f "$0")
CWD=$(dirname "$SCRIPT")

xhost +local:
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

docker run \
    -it --rm \
    --privileged \
    --net=host \
    --pid=host \
    --ipc=host \
    -e SDL_VIDEODRIVER='x11' \
    -e DISPLAY=$DISPLAY \
    -e HF_HUB_CACHE="/opt/NavigationScaling/earthrovers/deployment/hf_cache" \
    --mount "type=bind,src=$CWD/../,dst=/opt/NavigationScaling" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/localtime:/etc/localtime \
    -v /dev:/dev \
    -v /dev/shm:/dev/shm \
    --shm-size=8gb \
    --gpus 'all,"capabilities=graphics,utility,display,video,compute"' \
    "$DOCKER_IMAGE_ID" "$@" 
xhost -local: