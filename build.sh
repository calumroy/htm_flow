#!/bin/sh

# A shell script to build the project or clean the build directory.
# You can provide a release type parameter: Release, Debug, or RelWithDebInfo
# If no parameter is provided, the default is Release.
# You can optionally enable GPU and/or GUI:
# - GPU enables CUDA bits (requires nvcc)
# - GUI enables Qt6 debugger build (requires Qt6 installed locally)
# Example: ./build.sh Debug GPU
# Example: ./build.sh Release GUI
# Example: ./build.sh RelWithDebInfo GPU GUI

# Go to library source tree root and run the usual:
# make this an executable with chmod +x ./build.sh
# Use ./build.sh clean to delete the build directory.

# Function to display help information
display_help() {
    echo "Usage: ./build.sh [RELEASE_TYPE] [GPU] [GUI]"
    echo
    echo "RELEASE_TYPE:"
    echo "  Release        - Build with optimization (default)."
    echo "  Debug          - Build with debugging symbols."
    echo "  RelWithDebInfo - Build with optimization and debugging symbols."
    echo
    echo "OPTIONS:"
    echo "  GPU            - Enable GPU support."
    echo "  GUI            - Enable Qt6 GUI debugger build."
    echo
    echo "Clean:"
    echo "  clean          - Deletes the build folder."
    echo
    echo "Example:"
    echo "  ./build.sh Debug GPU"
    echo "  ./build.sh Release GUI"
    echo "  ./build.sh RelWithDebInfo GPU GUI"
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

# Optional flags (order-independent): GPU / GUI
GPU_OPTION="-DUSE_GPU=OFF"
GUI_OPTION="-DHTM_FLOW_WITH_GUI=OFF"

for arg in "$@"; do
  lower=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
  if [ "$lower" = "gpu" ]; then
    GPU_OPTION="-DUSE_GPU=ON"
  elif [ "$lower" = "gui" ]; then
    GUI_OPTION="-DHTM_FLOW_WITH_GUI=ON"
  fi
done

# Build the project with the specified release type parameter and GPU option.
# which nvcc should equal something like /usr/local/cuda-12.2/bin/nvcc
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=$RELEASE_TYPE -DCMAKE_CUDA_COMPILER=$(which nvcc) $GPU_OPTION $GUI_OPTION ..

cmake --build .
