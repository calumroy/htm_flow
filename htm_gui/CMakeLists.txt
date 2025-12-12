cmake_minimum_required(VERSION 3.16)

# Enable Qt meta-object compilation for Q_OBJECT classes.
set(CMAKE_AUTOMOC ON)

add_library(htm_gui
  src/debugger.cpp
  src/main_window.cpp
  src/views.cpp
)

target_include_directories(htm_gui
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

target_link_libraries(htm_gui
  PUBLIC
    Qt6::Widgets
)

target_compile_features(htm_gui PRIVATE cxx_std_17)
