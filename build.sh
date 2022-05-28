#!/bin/sh
# Go to library source tree root and run the usual:
# make this an executable with chmod +x ./build.sh
mkdir build && cd $_
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .