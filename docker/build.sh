#!/bin/sh

IMAGE_NAME="earthrovers_deployment"
TAG="0.0.1"


SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

# The base image tag keeps changing, if this does not work, check the latest ros2 humble tag from
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/containers/ros
BASE_IMAGE="nvcr.io/nvidia/isaac/ros:x86_64-ros2_humble_6f2a6bddf70fcd928f08e874635efe43"

# GPU capabilites require Nvidia container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    -t $IMAGE_NAME:$TAG \
    -f Dockerfile ..