# Build of `_cad_grad`, the C++ backend of the parameter optimizer.
#
# This is OUR recipe, not the one in the vendored snapshot: our source changes to
# the snapshot are applied as patches to a copy, and the snapshot's own
# `vendor/cad_optimizer/CMakeLists.txt` knows nothing about them. The snapshot must
# not be edited: the next snapshot update would silently drop the change. So the
# recipe lives here and only refers to the snapshot sources without changing them;
# our code changes are patches in `patches/`, applied to a copy in the build
# directory (below).
#
# Exactly one target is built: the Python module. There is no intermediate static
# library; the snapshot recipe needed it to link into both the tests and the module.
#
# Do not run this directly; use `agent/tools/build_cad_grad.sh`: it picks the
# interpreter, finds pybind11 and puts the result where the runtime looks for it.

cmake_minimum_required(VERSION 3.15)
project(cad_grad_module CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# agent/tools/cad_grad -> agent/tools -> agent -> repository root
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(SNAPSHOT "${REPO_ROOT}/vendor/cad_optimizer")

if(NOT EXISTS "${SNAPSHOT}/include/mesh.h")
    message(FATAL_ERROR
        "Snapshot is incomplete: no ${SNAPSHOT}/include. "
        "Headers are taken from the source, see vendor/cad_optimizer/PROVENANCE.md")
endif()

# Our changes to the snapshot sources are patches from `patches/`, applied in name order.
# The snapshot is not touched: `src/` is copied into the build directory and the patches
# are applied to the copy. If any patch fails to apply it is a configuration error, not a
# build without it: otherwise after a snapshot update the module would silently be built
# without the change.
# `-DCAD_GRAD_PATCHES=OFF` builds the snapshot as is, to check equivalence.
option(CAD_GRAD_PATCHES "apply agent/tools/cad_grad/patches to a copy of the snapshot src/" ON)
set(SRC_ROOT "${SNAPSHOT}/src")
if(CAD_GRAD_PATCHES)
    file(GLOB PATCHES "${CMAKE_CURRENT_LIST_DIR}/patches/*.patch")
    list(SORT PATCHES)
    set(PATCHED_ROOT "${CMAKE_BINARY_DIR}/patched")
    file(REMOVE_RECURSE "${PATCHED_ROOT}")
    file(COPY "${SNAPSHOT}/src" DESTINATION "${PATCHED_ROOT}")
    foreach(PATCH_FILE IN LISTS PATCHES)
        execute_process(
            COMMAND patch -p1 --forward --batch --no-backup-if-mismatch -d "${PATCHED_ROOT}" -i "${PATCH_FILE}"
            RESULT_VARIABLE PATCH_RC OUTPUT_VARIABLE PATCH_OUT ERROR_VARIABLE PATCH_OUT)
        if(NOT PATCH_RC EQUAL 0)
            message(FATAL_ERROR "Patch does not apply to the snapshot: ${PATCH_FILE}\n${PATCH_OUT}")
        endif()
        get_filename_component(PATCH_NAME "${PATCH_FILE}" NAME)
        message(STATUS "Patch applied: ${PATCH_NAME}")
    endforeach()
    set(SRC_ROOT "${PATCHED_ROOT}/src")
else()
    message(STATUS "Patches disabled: snapshot as is")
endif()

file(GLOB OPS_SRC "${SRC_ROOT}/ops/*.cpp")
if(NOT OPS_SRC)
    message(FATAL_ERROR "Snapshot is incomplete: no ${SNAPSHOT}/src/ops/*.cpp")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_cad_grad
    "${SRC_ROOT}/bindings.cpp"
    "${SRC_ROOT}/mesh.cpp"
    "${SRC_ROOT}/optimizer.cpp"
    "${SRC_ROOT}/csg_tree.cpp"
    ${OPS_SRC}
)
target_include_directories(_cad_grad PRIVATE "${SNAPSHOT}/include")

# OpenMP is needed only by the batched gradient and is optional in the snapshot
# recipe too: without it the module builds and works, just slower.
if(OpenMP_CXX_FOUND)
    target_link_libraries(_cad_grad PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found - the batched gradient will be single-threaded")
endif()
