# Use an official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set a label to describe the image
LABEL description="Docker image for building and running the htm_flow project with optional GPU support."

# Avoid user interaction with tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Clone and set up the external libraries
# Google Test and Taskflow
RUN git clone https://github.com/taskflow/taskflow.git && \
    mkdir -p include/taskflow && \
    cp -r taskflow/taskflow/* include/taskflow && \
    rm -rf taskflow

RUN mkdir -p lib && \
    cd lib && \
    git clone https://github.com/google/googletest/ && \
    cd ..

# Copy the local project directory into the Docker image
COPY . .

# Make the build script executable
RUN chmod +x build.sh

# Run the build script with parameters 'Release' and 'GPU'
RUN ./build.sh Release GPU > /build_logfile 2>&1; cat /build_logfile; test $(cat /build_logfile | grep -c "Error") -eq 0


# Set the default command to run the application
CMD ["./build/htm_flow"]
