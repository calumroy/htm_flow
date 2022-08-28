# Add a gpu example to compile. THis requires CUDA tookit nvcc to be installed.
list(APPEND TF_CUDA_EXAMPLES 
  task_gpu_test
)

foreach(cuda_example IN LISTS TF_CUDA_EXAMPLES)
  add_executable(${cuda_example} ${cuda_example}.cu)
  target_link_libraries(${cuda_example} ${PROJECT_NAME} 
      Threads::Threads tf::default_settings)

  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${cuda_example} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
endforeach()   