cmake_minimum_required(VERSION 3.20)

set(filepattern_known_comps static shared)
set(filepattern_comp_static NO)
set(filepattern_comp_shared NO)
foreach (filepattern_comp IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    if (filepattern_comp IN_LIST filepattern_known_comps)
        set(filepattern_comp_${filepattern_comp} YES)
    else ()
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
            "filepattern does not recognize component `${filepattern_comp}`.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        return()
    endif ()
endforeach ()

if (filepattern_comp_static AND filepattern_comp_shared)
    set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "filepattern `static` and `shared` components are mutually exclusive.")
    set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    return()
endif ()

set(filepattern_static_targets "${CMAKE_CURRENT_LIST_DIR}/filepattern-static-targets.cmake")
set(filepattern_shared_targets "${CMAKE_CURRENT_LIST_DIR}/filepattern-shared-targets.cmake")

macro(filepattern_load_targets type)
    if (NOT EXISTS "${filepattern_${type}_targets}")
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
            "filepattern `${type}` libraries were requested but not found.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        return()
    endif ()
    include("${filepattern_${type}_targets}")
endmacro()

if (filepattern_comp_static)
    filepattern_load_targets(static)
elseif (filepattern_comp_shared)
    filepattern_load_targets(shared)
elseif (DEFINED filepattern_SHARED_LIBS AND filepattern_SHARED_LIBS)
    filepattern_load_targets(shared)
elseif (DEFINED filepattern_SHARED_LIBS AND NOT filepattern_SHARED_LIBS)
    filepattern_load_targets(static)
elseif (BUILD_SHARED_LIBS)
    if (EXISTS "${filepattern_shared_targets}")
        filepattern_load_targets(shared)
    else ()
        filepattern_load_targets(static)
    endif ()
else ()
    if (EXISTS "${filepattern_static_targets}")
        filepattern_load_targets(static)
    else ()
        filepattern_load_targets(shared)
    endif ()
endif ()
