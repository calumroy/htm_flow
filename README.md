# htm_flow
A pattern recognition algorithm using the cpp library cpp taskflow. 
```
setup.sh
```
Installs taskflow locally and then copies the header only files for task flow to include directory.

```
build.sh
```  
builds the project using cmake and gcc > 10.2.1 
Creates the executable in a out of source build directory.
It also builds all the unit tests including the cuda GPU tests.

## Task flow with GPU support
To run the task flow with GPU support, you need to have CUDA installed on your system.
This requires installing CUDA compiler nvcc.   
See https://taskflow.github.io/taskflow/CompileTaskflowWithCUDA.html  

To build a gpu example use nvcc e.g
`/usr/local/cuda-11.7/bin/nvcc -std=c++17 -I ./include/ --extended-lambda ./cuda/task_gpu_test.cu -o gpu_test`

Info on CUDA cmake building
https://developer.nvidia.com/blog/building-cuda-applications-cmake/


## Unit tests
Uses the google test framework.
To run the unit tests, you need to have the google test framework installed.
This is done automatically by the setup.sh script.
```
setup.sh 
```  

To run the unit tests, use the executable `./build/htm_flow_tests`  
To list all unit tests use googletest flags `./build/htm_flow_tests --gtest_list_tests`  
To run specifc test e.g "parallel_Images2Neibs.test2_wrap" use googletest flags `./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test2_wrap`  

## Task Flow Profiler
To run the tests and output a task flow graph showing the thread profile use the flag
`TF_ENABLE_PROFILER=simple.json`  e.g `TF_ENABLE_PROFILER=simple.json ./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test2_wrap`  
This will output a json file with the task flow profile.
Paste this into https://taskflow.github.io/tfprof/ to get a nice view of the profile.

## Valgrind callgrind profiler
```
valgrind --tool=callgrind ./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test5_large
```
This will output a callgrind.out file.
To view the profile use kcachegrind `sudo apt install kcachegrind`  
e.g `kcachegrind callgrind.out.1234`

## Memory leak check
heaptrack `sudo apt install heaptrack` and `sudo apt install heaptrack-gui`
```
heaptrack ./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test5_large
```
```
heaptrack_gui ./heaptrack.htm_flow_tests.425833.zst
```