#!/bin/sh

# Downloads the latest taskflow source code from github.
git clone https://github.com/taskflow/taskflow.git
# Copies the header only files to the include directory if they don't exist already.
mkdir inlcude/taskflow
cp -r taskflow/taskflow/* include/taskflow