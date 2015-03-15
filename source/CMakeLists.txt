#
# CudaSift library
#
# Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
# Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
#

# Sources
#file(GLOB CUDA_SOURCE ${SOURCE_DIR}/*.cu)
#file(GLOB CUDA_HEADER ${SOURCE_DIR}/*.h)
set(CUDA_SOURCE cs_device.cu)
set(CUDA_HEADER cs_device.h)

# CUDA compiler flags
append_flags(CUDA_CFLAGS "-Xcompiler;-fPIC;-Xcompiler;-fopenmp")
append_flags(CUDA_CFLAGS "--ftz=true;--prec-div=false;--prec-sqrt=false")
append_flags(CUDA_CFLAGS "-gencode;arch=compute_30,code=sm_30")
append_flags(CUDA_CFLAGS "-gencode;arch=compute_35,code=sm_35")
append_flags(CUDA_CFLAGS "${OPENCV_CFLAGS}")
string(REPLACE " " ";" CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${CUDA_CFLAGS})

# CUDA linker flags
append_flags(CUDA_LFLAGS "${OPENCV_LDFLAGS}")
string(REPLACE " " ";" CUDA_LFLAGS ${CUDA_LFLAGS})

# Include directories
CUDA_INCLUDE_DIRECTORIES(${SOURCE_DIR} ${BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA target
#set(CUDA_VERBOSE_BUILD TRUE)
set(CUDA_EXEC "${PROJECT_NAME}")
CUDA_ADD_LIBRARY(${CUDA_EXEC} ${CUDA_HEADER} ${CUDA_SOURCE} SHARED)
target_link_libraries(${CUDA_EXEC} ${CUDA_LFLAGS})
