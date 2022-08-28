# htm_flow
A pattern recognition algorithm using the cpp library cpp taskflow. 

setup.sh
Installs taskflow locally and then copies the header only files for task flow to include directory.

build.sh
builds the project using cmake and gcc > 10.2.1
Creates the executable in a out of source build directory.

## Task flow with GPU support
To run the task flow with GPU support, you need to have CUDA installed on your system.
This requires installing CUDA compiler nvcc.   
See https://taskflow.github.io/taskflow/CompileTaskflowWithCUDA.html  
The installation instructions from nvidia to install cuda toolkit for debian 11 x86_64 where (on the 28/08/22) 
```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo dpkg -i cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo cp /var/cuda-repo-debian11-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```

To build a gpu example use nvcc e.g
`/usr/local/cuda-11.7/bin/nvcc -std=c++17 -I ./include/ --extended-lambda ./cuda/task_gpu_test.cu -o gpu_test`

