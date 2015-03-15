#
# CudaSift library
#
# Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
# Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
#

# CMake requirements
cmake_minimum_required(VERSION 2.6.2 FATAL_ERROR)

# Paths
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/source")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(BINARY_DIR "${CMAKE_BINARY_DIR}")

# Check build directory
if (${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_BINARY_DIR})
    MESSAGE("Please do an out-of-tree build:")
    MESSAGE("rm CMakeCache.txt; mkdir build; cd build; cmake ..; make")
    MESSAGE(FATAL_ERROR "in-tree-build detected")
endif ()

# Project
project(cudasift)

# Version
set(cudasift_VERSION_MAJOR 3)
set(cudasift_VERSION_MINOR 0)
set(cudasift_VERSION_PATCH 0)

# Options
option(PACKAGE "Create package" OFF)

# Packaging
if (PACKAGE)
    set(CPACK_PACKAGE_VERSION_MAJOR "${cudasift_VERSION_MAJOR}")
    set(CPACK_PACKAGE_VERSION_MINOR "${cudasift_VERSION_MINOR}")
    set(CPACK_PACKAGE_VERSION_PATCH "${cudasift_VERSION_PATCH}")
    set(CPACK_GENERATOR "ZIP")
    include(CPack)
endif()

# Tools and libraries
include(cmake-macros)
include(cmake-tools)
include(libs-cuda)
include(libs-opencv)

# Subdirs
add_subdirectory(source)
