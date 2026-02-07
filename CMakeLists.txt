cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

# Build the Qt GUI debugger (see htm_gui/).
# Default OFF so headless builds do not require Qt to be installed.
option(HTM_FLOW_WITH_GUI "Build Qt6 GUI debugger" OFF)

# Cmake option to enable building CUDA GPU section or not. Requires CUDA 11.0 or higher.
# To use specify -DUSE_GPU=ON when running cmake.
option(USE_GPU "Enable GPU support" OFF) 

# Check if USE_GPU option is set to ON
if(USE_GPU)
    project("htm_flow" VERSION 0.9 LANGUAGES CXX CUDA DESCRIPTION "A sparse distributed pattern recognizer using cpp taskflow library.")
    set(CUDA_DIR ${PROJECT_SOURCE_DIR}/cuda)
else()
    # If USE_GPU is OFF, then build the project without CUDA
    project("htm_flow" VERSION 0.9 LANGUAGES CXX DESCRIPTION "A sparse distributed pattern recognizer using cpp taskflow library.")
endif()

# Error settings
add_library(error_settings INTERFACE)
add_library(tf::error_settings ALIAS error_settings)
target_compile_options(
  error_settings
  INTERFACE
  $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wfatal-errors>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wfatal-errors>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wfatal-errors>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3 /permissive->
)

# -----------------------------------------------------------------------------
# Find NVCC
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------

if(USE_GPU)
    include(CheckLanguage)
    message(STATUS "Configuring CUDA ...")
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "\nNo CUDA compiler found")
    endif()
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
        message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
        message(FATAL_ERROR "\nTaskflow requires CUDA at least v11")
    endif()
endif()

# Default settings
add_library(default_settings INTERFACE)
add_library(tf::default_settings ALIAS default_settings)
target_link_libraries(
  default_settings 
  INTERFACE 
  tf::error_settings 
)

# Include directories
include_directories("include")

# Include directories for the project source files
include_directories(${PROJECT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# yaml-cpp for config file loading
# -----------------------------------------------------------------------------
if(EXISTS "${PROJECT_SOURCE_DIR}/lib/yaml-cpp/CMakeLists.txt")
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  add_subdirectory(lib/yaml-cpp)
endif()

if(USE_GPU)
    add_subdirectory(cuda)

    # Add test task flow executable (use this to confirm that task flow is working)
    # Test script name
    set(TEST_SCRIPT_NAME task_flow_test)
    add_executable(${TEST_SCRIPT_NAME} ./src/task_flow_test.cpp)

    target_compile_features(${TEST_SCRIPT_NAME} PRIVATE cxx_std_17)

    target_link_libraries(${TEST_SCRIPT_NAME} PRIVATE task_gpu_test)

    install(TARGETS ${TEST_SCRIPT_NAME} DESTINATION bin)
endif()

# Add all the cpp files in the src directory
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/overlap/*.cpp 
                                  ${PROJECT_SOURCE_DIR}/src/inhibition/*.cpp
                                  ${PROJECT_SOURCE_DIR}/src/spatiallearn/*.cpp
                                  ${PROJECT_SOURCE_DIR}/src/sequence_pooler/*/*.cpp
                                  ${PROJECT_SOURCE_DIR}/src/temporal_pooler/*.cpp
                                  ${PROJECT_SOURCE_DIR}/src/utilities/*.cpp)

# -----------------------------------------------------------------------------
# htm_flow_core: static library containing all HTM algorithm, config, and
# runtime code.  External projects can link against this target instead of
# duplicating the source file list.
# -----------------------------------------------------------------------------
add_library(htm_flow_core STATIC
  ./src/gui_runtime.cpp
  ./src/config.cpp
  ./src/config_loader.cpp
  ./src/htm_layer.cpp
  ./src/htm_region.cpp
  ./src/region_runtime.cpp
  ${SRC_FILES}
)

target_compile_features(htm_flow_core PUBLIC cxx_std_17)

# Public include directories so consumers get the headers automatically.
target_include_directories(htm_flow_core PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

# The IHtmRuntime interface lives in htm_gui/include (header-only, no Qt dep).
if(EXISTS "${PROJECT_SOURCE_DIR}/htm_gui/include")
  target_include_directories(htm_flow_core PUBLIC "${PROJECT_SOURCE_DIR}/htm_gui/include")
endif()

# Link yaml-cpp if available
if(TARGET yaml-cpp)
  target_link_libraries(htm_flow_core PUBLIC yaml-cpp)
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# -----------------------------------------------------------------------------
# Build the main htm_flow executable (links against htm_flow_core)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ./src/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE htm_flow_core)

# Required for CUDA debugging in Visual Studio Code. This adds a -G flag to compilation.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

if(USE_GPU)
  target_link_libraries(${PROJECT_NAME} PRIVATE task_gpu_test gpu_overlap)
endif()

if(HTM_FLOW_WITH_GUI)
  if(EXISTS "${PROJECT_SOURCE_DIR}/htm_gui/CMakeLists.txt")
    add_subdirectory(htm_gui)
    target_link_libraries(${PROJECT_NAME} PRIVATE htm_gui)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HTM_FLOW_WITH_GUI)
  else()
    message(WARNING "HTM_FLOW_WITH_GUI is ON but 'htm_gui/' is missing; building without GUI.")
  endif()
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# -----------------------------------------------------------------------------
# Debug region runner - for debugging integration test configurations with GUI
# -----------------------------------------------------------------------------
add_executable(debug_region ./src/debug_region.cpp)

target_link_libraries(debug_region PRIVATE htm_flow_core)

if(USE_GPU)
  target_link_libraries(debug_region PRIVATE task_gpu_test gpu_overlap)
endif()

if(HTM_FLOW_WITH_GUI)
  if(TARGET htm_gui)
    target_link_libraries(debug_region PRIVATE htm_gui)
    target_compile_definitions(debug_region PRIVATE HTM_FLOW_WITH_GUI)
  endif()
endif()

install(TARGETS debug_region DESTINATION bin)

####################################################################
# Build tests if (BUILD_TESTS) is ON 
option(BUILD_TESTS "Build the tests" ON)

if(BUILD_TESTS)
  include_directories(cuda
                      src
                      include/htm_flow
                      include/utilities
                      )
    
  # Add the Google Test subdirectory  
  add_subdirectory(lib/googletest)
  
  # Enable testing for your project
  enable_testing()
  include(GoogleTest)

  # Collect test source files only (no need to list library sources; they come
  # from htm_flow_core).
  file(GLOB_RECURSE TEST_FILES ./test/*.cpp)

  # Filter out the GPU test files if GPU flag is set to FALSE
  if(NOT USE_GPU)
    list(FILTER TEST_FILES EXCLUDE REGEX ".*gpu.*\.cpp$")
  endif()

  # Add your test files
  add_executable(${PROJECT_NAME}_tests ${TEST_FILES})

  # Link against htm_flow_core (brings in headers + all algorithm code)
  target_link_libraries(${PROJECT_NAME}_tests PRIVATE htm_flow_core)

  # Link your test executable to the Google Test library and other libraries
  if(USE_GPU)
    target_link_libraries(${PROJECT_NAME}_tests PRIVATE gtest gtest_main task_gpu_test gpu_overlap)
  else()
    target_link_libraries(${PROJECT_NAME}_tests PRIVATE gtest gtest_main)
  endif()

  # Register individual gtest TEST() cases with CTest so `ctest -R <pattern>` works.
  gtest_discover_tests(${PROJECT_NAME}_tests DISCOVERY_TIMEOUT 30)
endif()
