#
# CudaSift library
#
# Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
# Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
#

# Messages
message(STATUS "################################################")
message(STATUS "Checking for OpenCV")

# Find
pkg_check_modules(OPENCV opencv)
info_library("OpenCV" OPENCV FATAL_ERROR)
