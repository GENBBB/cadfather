cmake_minimum_required(VERSION 3.15)
project(cad_grad_optimizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Position-independent code is required for linking the static lib into
# the pybind11 shared module on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include dirs
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find OpenMP (optional, for batched gradient)
find_package(OpenMP)

# Core gradient library
file(GLOB OPS_SRC src/ops/*.cpp)
add_library(cad_grad STATIC
    src/mesh.cpp
    src/optimizer.cpp
    src/csg_tree.cpp
    ${OPS_SRC}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cad_grad PUBLIC OpenMP::OpenMP_CXX)
endif()

# Test executables (the snapshot has no C++ tests/ directory)
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_main.cpp)
    add_executable(cad_grad_tests tests/test_main.cpp)
    target_link_libraries(cad_grad_tests PRIVATE cad_grad)

    add_executable(cad_grad_combos tests/test_combos.cpp)
    target_link_libraries(cad_grad_combos PRIVATE cad_grad)

endif()

# pybind11 module (optional, only if pybind11 found)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_cad_grad src/bindings.cpp)
    target_link_libraries(_cad_grad PRIVATE cad_grad)
endif()
