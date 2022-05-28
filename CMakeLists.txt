cmake_minimum_required(VERSION 3.5)

project("htm_flow" VERSION 0.9 DESCRIPTION "An spare distributed pattern recogniser using cpp taskflow library.")

#add_subdirectory(./src/task_flow_test.cpp)

add_executable(${PROJECT_NAME} ./src/task_flow_test.cpp)

target_include_directories(${PROJECT_NAME}
 PRIVATE
  src/External/taskflow/)