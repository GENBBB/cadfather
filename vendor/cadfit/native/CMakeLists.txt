# cadfit._native: the C++ part of det (cluster_normals + section). No optimizer.
# Build through native/build.sh: it picks the interpreter, finds pybind11
# and puts the module into src/cadfit/.
cmake_minimum_required(VERSION 3.15)
project(cadfit_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native bindings.cpp section_fit.cpp)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
