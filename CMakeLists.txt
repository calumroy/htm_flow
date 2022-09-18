cmake_minimum_required(VERSION 3.8)

project("htm_flow" VERSION 0.9 DESCRIPTION "An spare distributed pattern recogniser using cpp taskflow library.")

set(CUDA_DIR ${PROJECT_SOURCE_DIR}/cuda)

# error setting
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
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------

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


add_library(default_settings INTERFACE)
add_library(tf::default_settings ALIAS default_settings)
target_link_libraries(
  default_settings 
  INTERFACE 
  tf::error_settings 
  #tf::optimize_settings 
  #tf::features_settings
)

include_directories("include")
                    # "cuda/include")

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(cuda)

add_executable(${PROJECT_NAME} ./src/task_flow_test.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

target_link_libraries(${PROJECT_NAME} PRIVATE task_gpu_test)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)