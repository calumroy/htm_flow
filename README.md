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
It also builds all the unit tests (include GPU option for including the cuda GPU tests, see below).

You can build a debug or release CPU only version of the project.
```
build.sh Debug
```
```
build.sh Release
```
```
build.sh RelWithDebInfo
```

You may need cmake installed to build the project.
```
sudo apt install cmake
```



## Task flow with GPU support
To run the task flow with GPU support, you need to have CUDA installed on your system.
This requires installing CUDA compiler nvcc.   
See https://taskflow.github.io/taskflow/CompileTaskflowWithCUDA.html  

once you have CUDA installed, you can build the project with GPU support using the build script.
```
build.sh Release GPU
```
or
```
./build.sh Debug GPU
```
This will build the project with GPU support and also build the GPU unit tests.

To build a gpu example using nvcc e.g
`/usr/local/cuda-11.7/bin/nvcc -std=c++17 -I ./include/ --extended-lambda ./cuda/task_gpu_test.cu -o gpu_test`
This is not needed however as the build script will automatically build the cuda examples.

Info on CUDA cmake building
https://developer.nvidia.com/blog/building-cuda-applications-cmake/


## Unit tests
Uses the google test framework.
To run the unit tests, you need to have the google test framework installed.
This is done automatically by the setup.sh script.
```
setup.sh 
```  

To run the unit tests, use the executable   
`./build/htm_flow_tests`  
To list all unit tests use googletest flags   
`./build/htm_flow_tests --gtest_list_tests`  
To run specifc test e.g "parallel_Images2Neibs.test2_wrap" use googletest flags  
`./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test2_wrap`  

To run the GPU unit tests you must build with the GPU option `build.sh Debug GPU`
Then run specific GPU unit tests with e.g  
`./build/htm_flow_tests --gtest_filter=gpu_Images2Neibs.test1_wrap`

# Visualize Taskflow Graphs
You can dump a taskflow graph to a DOT format and visualize it using a number of free GraphViz tools such as [GraphViz Online](https://dreampuf.github.io/GraphvizOnline/).

e.g add the code 
```
// dump the graph to a DOT file through std::cout
taskflow.dump(std::cout); 
```

After the graph has been dumped to the console, copy the output to the GraphViz Online tool and click the "Generate Graph" button.

## Task Flow Profiler
To run the tests and output a task flow graph showing the thread profile use the flag
`TF_ENABLE_PROFILER=simple.json`  e.g `TF_ENABLE_PROFILER=simple.json ./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test2_wrap`  
This will output a json file with the task flow profile.
Paste this into https://taskflow.github.io/tfprof/ to get a nice view of the profile.

Top get just a terminal output of the profiling data use the TF_ENABLE_PROFILER flag with no file name.  
```
TF_ENABLE_PROFILER= ./build/htm_flow_tests
```

For very large files the online profiler may not work.
use a local version of the profiler. Rename the output file to simple.tfp
```
git clone https://github.com/taskflow/taskflow.git
cd taskflow
mkdir build
cd build
cmake ../ -DTF_BUILD_PROFILER=ON
./tfprof/server/tfprof --mount ../tfprof/ --input ../../htm_flow/simple.tfp
```
Now go to http://localhost:8080/ to view the profile.

## Valgrind callgrind profiler
```
valgrind --tool=callgrind ./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test5_large
```
This will output a callgrind.out file.
To view the profile use kcachegrind `sudo apt install kcachegrind`  
e.g `kcachegrind callgrind.out.1234`
Valgrind gives the best profiling data when compiled with debug symbols `./build.sh Debug`  

## Memory leak check
heaptrack `sudo apt install heaptrack` and `sudo apt install heaptrack-gui`
```
heaptrack ./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test5_large
```
```
heaptrack_gui ./heaptrack.htm_flow_tests.425833.zst
```

## Nvidia nsight-systems profiler
To profile the cpu and gpu code using the Nvidia nsight-systems profiler.
Install this deb package https://developer.nvidia.com/nsight-systems
run `sudo nsys-ui`  
documentation at https://docs.nvidia.com/nsight-systems/UserGuide/index.html

For example to profile one the the gpu unit test cases in the nsys-ui gui use the following two commands as the command line arguments and the working directory.
`htm_flow_tests --gtest_filter=gpu_Images2Neibs.test3_very_large`  
`/home/calum/Documents/projects/htm_flow/build`  

## Using cuda-gdb to debug GPU code
To debug the GPU code you can use cuda-gdb tool
E.g to run the test case gpu_Images2Neibs.test4_large_2_step
and then set a conditional breakpoint in the kernel code on line 128:  

```
cuda-gdb --args ./build/htm_flow_tests --gtest_filter=gpu_Images2Neibs.test4_large_2_step
(cuda-gdb) break gpu_overlap.cu:128 if jj==19
(cuda-gdb) run
```

# Generate Doxygen code documentation
To generate the code documentation, you need to have doxygen installed.
```
sudo apt install doxygen
```
Then run the doxygen command in the root directory of the project.
```
doxygen Doxyfile
```
This will generate the documentation in the `htm_flow/docs` directory.
Open the `htm_flow/docs/html/index.html` file in a web browser to view the documentation.

# Run in a docker container
To run the project in a docker container, you need to have docker installed with nvidia container toolkit.
Install dockler with the following command:
```
 sudo apt-get update
 sudo apt-get install ./docker-desktop-<version>-<arch>.deb
```
See here for installing the nvidia container toolkit:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt

Then build the docker image with the following command:
```
./build_container.sh
```
This will build the docker image with the name `htm_flow:latest`.
