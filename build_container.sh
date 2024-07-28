#!/bin/bash

# Script to build the Docker container for the htm_flow project.

# Set the image name
IMAGE_NAME="htm_flow:latest"

# Set the Dockerfile name, if it's not the default Dockerfile
DOCKERFILE_NAME="Dockerfile"

# Check if the Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not running. Please start the Docker daemon and try again."
    exit 1
fi

# Build the Docker image
echo "Building Docker image '${IMAGE_NAME}'..."
if docker build -t ${IMAGE_NAME} -f ${DOCKERFILE_NAME} .; then
    echo "Docker image '${IMAGE_NAME}' built successfully."
else
    echo "Failed to build Docker image '${IMAGE_NAME}'."
    exit 1
fi

# Provide option to run the container after building
echo "Do you want to run the Docker container now? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo "Running Docker container '${IMAGE_NAME}'..."
    docker run --gpus all ${IMAGE_NAME}
else
    echo "Build completed. Not running the container."
fi
