cmake_minimum_required(VERSION 3.16)

project(gpu_htm_flow_cuda_lib CUDA CXX C)  # enable cuda language

if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "This project has a top-level one called [${CMAKE_PROJECT_NAME}]")
else()
    message(STATUS "This project is a top-level one")
endif()

############## LIBRARY 1 ####################
# Set the name of library 2
set(LIBRARY1_NAME task_gpu_test)
# Print the value of the variable ${PROJECT_SOURCE_DIR}
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
# Add all the cpp files in the src directory
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/task_gpu_test.cu)

add_library(${LIBRARY1_NAME} STATIC ${SRC_FILES})

target_sources(${LIBRARY1_NAME}
    PRIVATE
      ${SRC_FILES}
)

############## LIBRARY 2 ####################
# Set the name of library 2
set(LIBRARY2_NAME gpu_overlap)
# Add another library to the project.
# Add all the cpp files in the src directory
file(GLOB_RECURSE SRC_FILES_2 ${CMAKE_CURRENT_SOURCE_DIR}/src/overlap/gpu_overlap.cu)
# Print the value of the variable ${SRC_FILES_2}
message(STATUS "SRC_FILES_2: ${SRC_FILES_2}")
add_library(${LIBRARY2_NAME} STATIC ${SRC_FILES_2})

target_sources(${LIBRARY2_NAME}
    PRIVATE
      ${SRC_FILES_2}
)
#############################################

############## EXECUTABLE LIBRARY 2 ####################
# Set the name of the executable
set(EXECUTABLE2_NAME gpu_overlap_exec)
# Add the executable
#add_executable(${EXECUTABLE2_NAME} src/main.cpp src/overlap/gpu_overlap.cu)
# Link the executable with library 2
#target_link_libraries(${EXECUTABLE2_NAME} ${LIBRARY2_NAME})

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 11)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

# set(PUBLIC_HEADERS3
#     include/overlap/gpu_overlap.hpp
# )
include_directories(include/)

link_directories(include/)


add_executable(${EXECUTABLE2_NAME})
target_sources(${EXECUTABLE2_NAME} PRIVATE src/main.cpp)
set_target_properties(${EXECUTABLE2_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${EXECUTABLE2_NAME} PRIVATE gpu_overlap)


#############################################

find_package(Threads REQUIRED)

# target_link_libraries(task_gpu_test ${PROJECT_NAME} 
#        Threads::Threads tf::default_settings)


if(NOT CUDA_ARCHITECTURES)
#   set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
  set_property(TARGET ${LIBRARY1_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
  set_property(TARGET ${LIBRARY2_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
  #set_property(TARGET ${EXECUTABLE2_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
endif()

# set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${LIBRARY1_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${LIBRARY2_NAME} PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(${EXECUTABLE2_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_features(${LIBRARY1_NAME} PRIVATE cxx_std_17)
target_compile_features(${LIBRARY2_NAME} PRIVATE cxx_std_17)
#target_compile_features(${EXECUTABLE2_NAME} PRIVATE cxx_std_17)
# without it public headers won't get installed
set(public_headers1
    include/task_gpu_test.hpp
)
set(public_headers2
    include/overlap/gpu_overlap.hpp
)
# note that ${public_headers} has to be in quotes
# set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")
set_target_properties(${LIBRARY1_NAME} PROPERTIES PUBLIC_HEADER "${public_headers1}")
set_target_properties(${LIBRARY2_NAME} PROPERTIES PUBLIC_HEADER "${public_headers2}")


# target_include_directories(${PROJECT_NAME}
#     PRIVATE
#         # where the library itself will look for its internal headers
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#     PUBLIC
#         # where top-level project will look for the library's public headers
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#         # where external projects will look for the library's public headers
#         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )
target_include_directories(${LIBRARY1_NAME}
    PRIVATE
        # where the library itself will look for its internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(${LIBRARY2_NAME}
    PRIVATE
        # where the library itself will look for its internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

#Not related to public headers, but it might be a good idea to add d suffix to debug binaries - that way you’ll get libSomeLibraryd.d with Debug configuration and libSomeLibrary.a with Release. 
set_target_properties(${LIBRARY1_NAME} PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(${LIBRARY2_NAME} PROPERTIES DEBUG_POSTFIX "d")

# Required for CUDA debugging in Visual Studio Code. THis adds a -G flag to compilation.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${LIBRARY1_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    target_compile_options(${LIBRARY2_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    #target_compile_options(${EXECUTABLE2_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


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

