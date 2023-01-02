#!/bin/sh

# Downloads the latest taskflow source code from github.
git clone https://github.com/taskflow/taskflow.git
# Copies the header only files to the include directory if they don't exist already.
mkdir -p include/taskflow
cp -r taskflow/taskflow/* include/taskflow

# Downloads the latest google test source code from github.
mkdir -p lib
cd ./lib
git clone https://github.com/google/googletest/