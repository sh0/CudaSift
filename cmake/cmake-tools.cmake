#
# CudaSift library
#
# Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
# Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
#

# Messages
message(STATUS "################################################")
message(STATUS "Checking for pkg-config")

# Find
find_package(PkgConfig)
info_tool("pkg-config" PKG_CONFIG_FOUND FATAL_ERROR)
if (VERBOSE)
    message(STATUS "Path: ${PKG_CONFIG_EXECUTABLE}")
endif ()
