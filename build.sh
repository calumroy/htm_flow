#!/bin/sh

# A Bash Script to build the project.
# You can provide a release type parameter: Release, Debug, or RelWithDebInfo
# If no parameter is provided, the default is Release.
# You can also provide a GPU option: GPU 
# Example: ./build.sh Debug GPU

# Go to library source tree root and run the usual:
# make this an executable with chmod +x ./build.sh

# A bash script to create a build dir if it doesn't exist and then move into it.
# If the build dir exists, it will be deleted and recreated.

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
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=$RELEASE_TYPE -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.0/bin/nvcc $GPU_OPTION ..

cmake --build .
