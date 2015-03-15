#
# CudaSift library
#
# Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
# Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
#

# Messages
message(STATUS "################################################")
message(STATUS "Checking for CUDA")

# Find
find_package(CUDA 5)
info_tool("CUDA" CUDA_FOUND FATAL_ERROR)
if (VERBOSE)
    message(STATUS "Version: ${CUDA_VERSION_STRING}")
    message(STATUS "Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "SDK: ${CUDA_SDK_ROOT_DIR}")
endif ()
