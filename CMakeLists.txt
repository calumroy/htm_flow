cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

# Build the Qt GUI debugger (see htm_gui/).
option(HTM_FLOW_WITH_GUI "Build Qt6 GUI debugger" ON)

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

# Build the Project executable
add_executable(${PROJECT_NAME} ./src/main.cpp ${SRC_FILES})

# Required for CUDA debugging in Visual Studio Code. This adds a -G flag to compilation.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

if(USE_GPU)
  target_link_libraries(${PROJECT_NAME} PRIVATE task_gpu_test gpu_overlap)
else()
  target_link_libraries(${PROJECT_NAME} PRIVATE)
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

# Add all the cpp files in the test directory and include the necessary source files
file(GLOB_RECURSE TEST_FILES ./test/*.cpp ${PROJECT_SOURCE_DIR}/src/inhibition/inhibition.cpp 
                                  ${PROJECT_SOURCE_DIR}/src/overlap/overlap.cpp 
                                  ${PROJECT_SOURCE_DIR}/src/overlap/overlap_utils.cpp
                                  ${PROJECT_SOURCE_DIR}/src/spatiallearn/spatiallearn.cpp
                                  ${PROJECT_SOURCE_DIR}/src/sequence_pooler/active_cells/active_cells.cpp
                                  ${PROJECT_SOURCE_DIR}/src/sequence_pooler/predict_cells/predict_cells.cpp
                                  ${PROJECT_SOURCE_DIR}/src/sequence_pooler/sequence_learning/sequence_learning.cpp
                                  ${PROJECT_SOURCE_DIR}/src/temporal_pooler/temporal_pooler.cpp
                                  ${PROJECT_SOURCE_DIR}/src/utilities/stopwatch.cpp)


  # Filter out the GPU test files if GPU flag is set to FALSE
  if(NOT USE_GPU)
    list(FILTER TEST_FILES EXCLUDE REGEX ".*gpu.*\.cpp$")
  endif()

  # Add your test files
  add_executable(${PROJECT_NAME}_tests ${TEST_FILES})

  # Link your test executable to the Google Test library and other libraries
  if(USE_GPU)
    target_link_libraries(${PROJECT_NAME}_tests gtest gtest_main task_gpu_test gpu_overlap)
  else()
    target_link_libraries(${PROJECT_NAME}_tests gtest gtest_main)
  endif()

  # Register the test executable with CTest
  add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)
endif()
