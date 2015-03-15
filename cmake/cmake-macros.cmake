#
# CudaSift library
#
# Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
# Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
#

# Tool info
macro (info_tool tool_name tool_found)
    if (${tool_found})

        message(STATUS "Found ${tool_name}")

    else ()

        if (${ARGV2})
            message(${ARGV2} "Could not find ${tool_name}")
        else (${ARGV2})
            message(STATUS "Could not find ${tool_name}")
        endif ()

    endif ()
endmacro ()

# Library info
macro (info_library lib_name lib_prefix)
    if (${lib_prefix}_FOUND)

        message(STATUS "Found ${lib_name}")
        if (VERBOSE)
            if (${lib_prefix}_CFLAGS)
                set(_cflags)
                append_flags(_cflags "${${lib_prefix}_CFLAGS}")
                message(STATUS "Compiler flags: \"${_cflags}\"")
            endif (${lib_prefix}_CFLAGS)
            if (${lib_prefix}_LDFLAGS)
                set(_ldflags)
                append_flags(_ldflags "${${lib_prefix}_LDFLAGS}")
                message(STATUS "Linker flags: \"${_ldflags}\"")
            endif (${lib_prefix}_LDFLAGS)
        endif ()

    else ()

        if (${ARGV2})
            message(${ARGV2} "Could not find ${lib_name}")
        else (${ARGV2})
            message(STATUS "Could not find ${lib_name}")
        endif (${ARGV2})

    endif ()
endmacro ()

# Compile flags
macro (append_flags _flags _append)
    # Fix _flags
    if (NOT ${_flags})
        set(${_flags} "")
    else ()
        string(STRIP "${${_flags}}" ${_flags})
    endif ()

    # Parse _append
    set(_temp "")
    foreach (_item ${_append})
        string(STRIP "${_item}" _item)
        string(REGEX REPLACE "^/.*/lib([^/]*).so$" "-l\\1" _item "${_item}")
        string(STRIP "${_temp} ${_item}" _temp)
    endforeach ()

    # Save result
    if ("${${_flags}}" STREQUAL "")
        set(${_flags} "${_temp}")
    else ("${${_flags}}" STREQUAL "")
        set(${_flags} "${${_flags}} ${_temp}")
        string(STRIP "${${_flags}}" ${_flags})
    endif ("${${_flags}}" STREQUAL "")
endmacro ()
