


cmake_minimum_required(VERSION 3.8)

project(cuda)

if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "This project has a top-level one called [${CMAKE_PROJECT_NAME}]")
else()
    message(STATUS "This project is a top-level one")
endif()

add_library(task_gpu_test STATIC task_gpu_test.cu)

#add_library(${PROJECT_NAME} SHARED)

target_sources(task_gpu_test
    PRIVATE
      task_gpu_test.cu
)

find_package(Threads REQUIRED)

# target_link_libraries(task_gpu_test ${PROJECT_NAME} 
#        Threads::Threads tf::default_settings)


if(NOT CUDA_ARCHITECTURES)
  set_property(TARGET task_gpu_test PROPERTY CUDA_ARCHITECTURES OFF)
endif()

set_target_properties(task_gpu_test PROPERTIES LINKER_LANGUAGE CXX)

# without it public headers won't get installed
set(public_headers
    include/task_gpu_test.hpp
)
# note that ${public_headers} has to be in quotes
set_target_properties(task_gpu_test PROPERTIES PUBLIC_HEADER "${public_headers}")



# #Not related to public headers, but it might be a good idea to add d suffix to debug binaries - that way you’ll get libSomeLibraryd.d with Debug configuration and libSomeLibrary.a with Release. 
# set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

# install(TARGETS task_gpu_test DESTINATION lib)
# install(FILES task_gpu_test.cu DESTINATION include)

# # Add a gpu example to compile. THis requires CUDA tookit nvcc to be installed.
# list(APPEND TF_CUDA_EXAMPLES 
#   task_gpu_test
# )

# foreach(cuda_example IN LISTS TF_CUDA_EXAMPLES)
#   #add_executable(${cuda_example} ${cuda_example}.cu)
#   #target_link_libraries(${cuda_example} ${PROJECT_NAME} 
#   #    Threads::Threads tf::default_settings)
#   add_library(${cuda_example} SHARED ${cuda_example}.cu)
#   FIND_PACKAGE (Threads)
#   target_link_libraries(${cuda_example} ${PROJECT_NAME} 
#       Threads::Threads tf::default_settings)
#   # avoid cmake 3.18+ warning
#   # we let nvcc to decide the flag if the architecture is not given
#   if(NOT CUDA_ARCHITECTURES)
#     set_property(TARGET ${cuda_example} PROPERTY CUDA_ARCHITECTURES OFF)
#   endif()

#   install(TARGETS ${cuda_example} DESTINATION lib)
#   install(FILES ${cuda_example}.cu DESTINATION include)
# endforeach()   

