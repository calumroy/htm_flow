# htm_flow

A C++ implementation of Hierarchical Temporal Memory (HTM) using the [Taskflow](https://taskflow.github.io/) parallel programming library.

## Overview

HTM is a biologically-inspired machine learning algorithm that models the neocortex. This implementation provides:

- **HTMLayer**: A single HTM layer implementing the full pipeline (overlap → inhibition → spatial learning → sequence memory → temporal pooling)
- **HTMRegion**: A stack of layers forming a cortical hierarchy, where each layer learns increasingly abstract representations
- **HTMRegionRuntime**: GUI-compatible wrapper for visualization and debugging

The architecture allows flexible configuration and testing of single layers, multi-layer regions, or complete networks.

## Quick Start

```bash
# Install dependencies (Taskflow, GoogleTest)
./setup.sh

# Build (CPU-only, Release mode)
./build.sh Release

# Run headless (3 steps by default)
./build/htm_flow

# Run with more steps
./build/htm_flow --steps 100
```

## Building

### Prerequisites
```bash
sudo apt install cmake g++  # GCC >= 10.2.1 required
```

### Build Options

```bash
./build.sh [Release|Debug|RelWithDebInfo] [GPU]
```

**Examples:**
```bash
./build.sh Debug              # Debug build, CPU only
./build.sh Release            # Release build, CPU only
./build.sh Debug GPU          # Debug build with CUDA support
./build.sh Release GPU        # Release build with CUDA support
./build.sh clean              # Delete build directory
```

> **Note:** For GUI builds, use the container scripts in `htm_gui/` (see "With GUI" section below).

## Running the Main Executable

### Headless Mode (no GUI)
```bash
./build/htm_flow                    # Run 3 steps (default)
./build/htm_flow --steps 100        # Run 100 steps
./build/htm_flow --steps 50 --log   # Run with timing output
```

### With GUI (Container)

The GUI uses a container with Qt6 pre-installed, so you don't need to install Qt6 locally.

```bash
# Install podman if needed
sudo apt install podman

# Build the container image (once)
cd htm_gui
./build_image.sh

# Run the main GUI
./run_gui.sh

# Or run debug_region with different configs (see "Manual Debugging" section)
./run_debug.sh 2layer
```

The scripts mount your source code, build inside the container, and launch the GUI with X11/Wayland display forwarding.

**The GUI allows you to:**
- Step through the HTM algorithm one timestep at a time
- Visualize column activations, cell states, and predictions
- Inspect proximal and distal synapses for any column/cell
- Switch between different input patterns
- Select which layer to view (for multi-layer regions)

## Manual Debugging with GUI

The `debug_region` tool lets you run any HTM configuration with the GUI for manual debugging and experimentation.

```bash
cd htm_gui
./build_image.sh              # Build container (once)

# Run different configurations
./run_debug.sh                # Single layer (default)
./run_debug.sh 2layer         # Two-layer hierarchy
./run_debug.sh 3layer         # Three-layer hierarchy
./run_debug.sh temporal       # Temporal pooling config

# Train first, then debug
./run_debug.sh 2layer --train 20

# Show all options
./run_debug.sh --help
```

**Available configurations:**
| Config | Description |
|--------|-------------|
| `1layer` / `single` | Single layer (default) |
| `2layer` | Two-layer hierarchy |
| `3layer` | Three-layer hierarchy |
| `temporal` | Temporal pooling experiment (Python test suite params) |
| `default` | Default config (larger 20x40 grid) |

**Adding custom configurations:** Edit `src/debug_region.cpp` and add to the `create_config()` function:

```cpp
if (name == "my_config") {
  auto layer_cfg = htm_flow::small_test_config();
  layer_cfg.cells_per_column = 8;  // Customize
  return htm_flow::uniform_region_config(2, layer_cfg);
}
```



## Project Structure

```
htm_flow/
├── include/htm_flow/
│   ├── config.hpp           # Configuration structs (HTMLayerConfig, HTMRegionConfig)
│   ├── htm_layer.hpp        # Single HTM layer (full pipeline)
│   ├── htm_region.hpp       # Multi-layer hierarchy
│   ├── region_runtime.hpp   # GUI-compatible wrapper with input generation
│   ├── overlap.hpp          # Overlap calculator
│   ├── inhibition.hpp       # Inhibition calculator
│   ├── spatiallearn.hpp     # Spatial learning calculator
│   ├── sequence_pooler/     # Sequence memory (active cells, predictions, learning)
│   └── temporal_pooler/     # Temporal pooling calculator
├── src/
│   ├── debug_region.cpp     # Debug tool for manual GUI testing
│   └── ...                  # Implementation files
├── test/                    # Unit and integration tests
├── htm_gui/                 # Qt6 debugger GUI
└── cuda/                    # GPU implementations
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `HTMLayer` | Core HTM layer with full pipeline. Use directly or stack into regions. |
| `HTMRegion` | Stack of HTMLayers for hierarchical processing. |
| `HTMRegionRuntime` | GUI-compatible wrapper with input generation and layer selection. |

### Configuration

All HTM parameters are defined in `HTMLayerConfig` (see `include/htm_flow/config.hpp`). Preset configurations are available:

```cpp
auto cfg = htm_flow::default_layer_config();      // Standard defaults
auto cfg = htm_flow::small_test_config();         // Fast testing
auto cfg = htm_flow::temporal_pooling_test_config(); // Temporal pooling experiments
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


## Testing

Uses the [GoogleTest](https://github.com/google/googletest) framework (installed by `setup.sh`).

### Running Tests

```bash
# Run all tests
./build/htm_flow_tests

# List available tests
./build/htm_flow_tests --gtest_list_tests

# Run a specific test
./build/htm_flow_tests --gtest_filter=parallel_Images2Neibs.test2_wrap

# Run tests matching a pattern
./build/htm_flow_tests --gtest_filter=parallel*
```

### Integration Tests

Integration tests verify the complete HTM pipeline (HTMLayer, HTMRegion):

```bash
# Run all HTMLayer integration tests
./build/htm_flow_tests --gtest_filter=HTMLayerIntegration.*

# Run all HTMRegion integration tests
./build/htm_flow_tests --gtest_filter=HTMRegionIntegration.*

# Run all integration tests
./build/htm_flow_tests --gtest_filter=*Integration*
```

**Available integration test suites:**
- `HTMLayerIntegration.*` - Single layer sanity checks (columns activate, predictions form, learning occurs)
- `HTMRegionIntegration.*` - Multi-layer hierarchies (layer stacking, input propagation, temporal pooling)

### Unit Tests

Unit tests verify individual calculators in isolation:

```bash
./build/htm_flow_tests --gtest_filter=parallel_inhibition*   # Inhibition calculator
./build/htm_flow_tests --gtest_filter=parallel_overlap*      # Overlap calculator
./build/htm_flow_tests --gtest_filter=spatiallearn*          # Spatial learning
./build/htm_flow_tests --gtest_filter=TemporalPooler*        # Temporal pooler
```

### GPU Tests

Requires building with GPU support: `./build.sh Debug GPU`

```bash
./build/htm_flow_tests --gtest_filter=gpu_Images2Neibs.*
```

## Visualize Taskflow Graphs
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

## Generate Doxygen Code Documentation
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

## Run in a Docker Container
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

Enter the docker container with the following command:
```
docker run -it --gpus all htm_flow:latest
```
