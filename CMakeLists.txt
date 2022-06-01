cmake_minimum_required(VERSION 3.8)

project("htm_flow" VERSION 0.9 DESCRIPTION "An spare distributed pattern recogniser using cpp taskflow library.")

include_directories("include")

add_executable(${PROJECT_NAME} ./src/task_flow_test.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")