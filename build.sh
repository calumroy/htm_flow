#!/bin/sh

# A Bash Script to build the project or clean the build directory.
# You can provide a release type parameter: Release, Debug, or RelWithDebInfo
# If no parameter is provided, the default is Release.
# You can also provide a GPU option: GPU 
# Example: ./build.sh Debug GPU

# Go to library source tree root and run the usual:
# make this an executable with chmod +x ./build.sh
# Use ./build.sh clean to delete the build directory.

# Function to display help information
display_help() {
    echo "Usage: ./build.sh [RELEASE_TYPE] [GPU_OPTION]"
    echo
    echo "RELEASE_TYPE:"
    echo "  Release        - Build with optimization (default)."
    echo "  Debug          - Build with debugging symbols."
    echo "  RelWithDebInfo - Build with optimization and debugging symbols."
    echo
    echo "GPU_OPTION:"
    echo "  GPU            - Enable GPU support."
    echo
    echo "Clean:"
    echo "  clean          - Deletes the build folder."
    echo
    echo "Example:"
    echo "  ./build.sh Debug GPU"
    echo "  ./build.sh clean"
    exit 0
}

# Check if -h parameter is provided
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    display_help
fi

# Check if clean parameter is provided
if [ "$1" = "clean" ]; then
    echo "Cleaning the build directory..."
    rm -rf build
    echo "Build directory cleaned."
    exit 0
fi

# Make the build dir if it doesn't exist.
mkdir -p build
# Move into the build dir.
cd build

# Set default release type if no parameter is provided.
if [ -z "$1" ]; then
  RELEASE_TYPE="Release"
else
  RELEASE_TYPE=$1
fi

# Check if the provided parameter is valid
if [ "$RELEASE_TYPE" != "Release" ] && [ "$RELEASE_TYPE" != "Debug" ] && [ "$RELEASE_TYPE" != "RelWithDebInfo" ]; then
  echo "Invalid release type parameter. Please choose from Release, Debug, or RelWithDebInfo."
  exit 1
fi

# Check for GPU option (case-insensitive)
if [ -z "$2" ]; then
  # Default behavior when $2 is not provided
  GPU_OPTION="-DUSE_GPU=OFF"
elif [ "$(echo "$2" | tr '[:upper:]' '[:lower:]')" = "gpu" ]; then
  GPU_OPTION="-DUSE_GPU=ON"
else
  GPU_OPTION="-DUSE_GPU=OFF"
fi

# Build the project with the specified release type parameter and GPU option.
# which nvcc should equal something like /usr/local/cuda-12.2/bin/nvcc
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=$RELEASE_TYPE -DCMAKE_CUDA_COMPILER=$(which nvcc) $GPU_OPTION ..

cmake --build .
