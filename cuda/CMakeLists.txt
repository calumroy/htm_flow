


cmake_minimum_required(VERSION 3.8)

project(cuda)

add_library(task_gpu_test SHARED task_gpu_test.cu)

find_package(Threads REQUIRED)

target_link_libraries(task_gpu_test ${PROJECT_NAME} 
       Threads::Threads tf::default_settings)


if(NOT CUDA_ARCHITECTURES)
  set_property(TARGET task_gpu_test PROPERTY CUDA_ARCHITECTURES OFF)
endif()

set_target_properties(task_gpu_test PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS task_gpu_test DESTINATION lib)
install(FILES task_gpu_test.cu DESTINATION include)

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

