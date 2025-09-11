###############################################################################
# EnableMP.cmake
#
# Configures OpenMP for shared-memory parallelism within a single machine.
# Drop this file in your project's `cmake/` directory and include it after
# the `project()` command in your top-level CMakeLists.txt:
#
#   option(ENABLE_OPENMP "Build with OpenMP shared-memory parallelism" ON)
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#   include(EnableMP)        # defines imported target openmp::openmp
#
# What you get:
#   • IMPORTED target `openmp::openmp` (INTERFACE) – safely empty if OpenMP is disabled
#   • COMPILE_DEF  ENABLE_OPENMP    – defined when OpenMP is enabled
#   • Convenience function add_openmp_target(<tgt>)
#   • Auto-detection of OpenMP runtime (libgomp, libomp, vcomp)
#
# Platforms supported:
#   • Linux:   GNU OpenMP (libgomp) or LLVM OpenMP (libomp)
#   • macOS:   LLVM OpenMP (libomp) via Homebrew or system
#   • Windows: MSVC OpenMP (vcomp) or Intel OpenMP
#
# Advanced options:
#   • OPENMP_RUNTIME: Force specific runtime (auto|libgomp|libomp|iomp|vcomp)
#   • FETCH_OPENMP:   Download LLVM OpenMP if system OpenMP not found
#   • OPENMP_NUM_THREADS_DEFAULT: Default thread count (0 = auto-detect)
###############################################################################

#-------------------------  include-once guard  -------------------------------
if(DEFINED _ENABLE_MP_INCLUDED)
    return()
endif()
set(_ENABLE_MP_INCLUDED TRUE)

# ------------------- user-facing toggles -------------------------------------
option(ENABLE_OPENMP "Build with OpenMP shared-memory parallelism" ON)
option(FETCH_OPENMP  "Download & build LLVM OpenMP if not found"  OFF)
set(OPENMP_RUNTIME "auto" CACHE STRING
        "OpenMP runtime to use (auto|libgomp|libomp|iomp|vcomp)")
set(OPENMP_NUM_THREADS_DEFAULT "0" CACHE STRING
        "Default OpenMP thread count (0=auto)")

# ------------------- early exit if OpenMP disabled ---------------------------
if(NOT ENABLE_OPENMP)
    message(STATUS "OpenMP support disabled (ENABLE_OPENMP=OFF)")
    add_library(openmp::openmp INTERFACE IMPORTED GLOBAL)  # empty stub
    return()
endif()

message(STATUS "Configuring OpenMP shared-memory parallelism...")

# ------------------- detect compiler and choose runtime ----------------------
function(_detect_openmp_runtime result_var)
    if(NOT OPENMP_RUNTIME STREQUAL "auto")
        set(${result_var} "${OPENMP_RUNTIME}" PARENT_SCOPE)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(${result_var} "libgomp" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(APPLE)
            # macOS Clang typically needs libomp from Homebrew
            set(${result_var} "libomp" PARENT_SCOPE)
        else()
            # Linux Clang can use either libgomp or libomp
            set(${result_var} "libomp" PARENT_SCOPE)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR
            CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(${result_var} "iomp" PARENT_SCOPE)
    elseif(MSVC)
        set(${result_var} "vcomp" PARENT_SCOPE)
    else()
        set(${result_var} "auto" PARENT_SCOPE)
    endif()
endfunction()

_detect_openmp_runtime(_openmp_runtime)
message(STATUS "  Detected OpenMP runtime preference: ${_openmp_runtime}")

# ------------------- try system OpenMP first ---------------------------------
find_package(OpenMP QUIET)

if(OpenMP_FOUND AND OpenMP_CXX_FOUND)
    message(STATUS "  Found system OpenMP:")
    message(STATUS "    Version:     ${OpenMP_CXX_VERSION}")
    message(STATUS "    Flags:       ${OpenMP_CXX_FLAGS}")
    message(STATUS "    Libraries:   ${OpenMP_CXX_LIBRARIES}")

    # Create our unified interface target
    add_library(openmp::openmp INTERFACE IMPORTED GLOBAL)

    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(openmp::openmp INTERFACE OpenMP::OpenMP_CXX)
    else()
        # Fallback for older CMake versions
        target_compile_options(openmp::openmp INTERFACE ${OpenMP_CXX_FLAGS})
        if(OpenMP_CXX_LIBRARIES)
            target_link_libraries(openmp::openmp INTERFACE ${OpenMP_CXX_LIBRARIES})
        endif()
    endif()

    target_compile_definitions(openmp::openmp INTERFACE ENABLE_OPENMP)

    # Set default thread count if specified
    if(NOT OPENMP_NUM_THREADS_DEFAULT STREQUAL "0")
        target_compile_definitions(openmp::openmp
                INTERFACE OMP_NUM_THREADS_DEFAULT=${OPENMP_NUM_THREADS_DEFAULT})
    endif()

elseif(FETCH_OPENMP)
    # ------------------- build LLVM OpenMP from source -------------------
    message(STATUS "  System OpenMP not found, fetching LLVM OpenMP...")

    include(FetchContent)
    include(ExternalProject)

    set(_openmp_install "${CMAKE_BINARY_DIR}/_deps/_openmp_install")
    set(_openmp_version "18.1.0")

    FetchContent_Declare(
            llvm_openmp
            GIT_REPOSITORY https://github.com/llvm/llvm-project.git
            GIT_TAG        llvmorg-${_openmp_version}
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  openmp
    )

    FetchContent_GetProperties(llvm_openmp)
    if(NOT llvm_openmp_POPULATED)
        FetchContent_Populate(llvm_openmp)

        # Configure LLVM OpenMP build
        set(_openmp_cmake_args
                -DCMAKE_INSTALL_PREFIX=${_openmp_install}
                -DCMAKE_BUILD_TYPE=Release
                -DLIBOMP_ENABLE_SHARED=ON
                -DLIBOMP_USE_HWLOC=OFF
                -DLIBOMP_OMPT_SUPPORT=OFF
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        )

        if(CMAKE_C_COMPILER)
            list(APPEND _openmp_cmake_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
        endif()
        if(CMAKE_CXX_COMPILER)
            list(APPEND _openmp_cmake_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
        endif()

        ExternalProject_Add(
                llvm_openmp_build
                SOURCE_DIR      ${llvm_openmp_SOURCE_DIR}/openmp
                CMAKE_ARGS      ${_openmp_cmake_args}
                BUILD_ALWAYS    FALSE
                STEP_TARGETS    install
        )

        # Create imported target for fetched OpenMP
        add_library(openmp::openmp INTERFACE IMPORTED GLOBAL)

        set(_openmp_lib_name "${CMAKE_SHARED_LIBRARY_PREFIX}omp${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(_openmp_lib_path "${_openmp_install}/lib/${_openmp_lib_name}")

        target_include_directories(openmp::openmp INTERFACE
                "${_openmp_install}/include")
        target_link_libraries(openmp::openmp INTERFACE
                "${_openmp_lib_path}")
        target_compile_options(openmp::openmp INTERFACE
                -fopenmp)
        target_compile_definitions(openmp::openmp INTERFACE
                ENABLE_OPENMP)

        # Ensure build order
        add_dependencies(openmp::openmp llvm_openmp_build)

        message(STATUS "  LLVM OpenMP will be built at configure time")
    endif()

else()
    # ------------------- OpenMP not found and fetch disabled -------------
    message(WARNING
            "OpenMP not found and FETCH_OPENMP=OFF. "
            "OpenMP support will be disabled.")
    add_library(openmp::openmp INTERFACE IMPORTED GLOBAL)  # empty stub
    return()
endif()

# ------------------- platform-specific adjustments ---------------------------
if(APPLE AND TARGET openmp::openmp)
    # macOS specific: help find Homebrew's libomp if needed
    if(NOT OpenMP_FOUND AND NOT FETCH_OPENMP)
        find_path(_homebrew_openmp_include omp.h
                PATHS /usr/local/opt/libomp/include
                /opt/homebrew/opt/libomp/include
                NO_DEFAULT_PATH)
        find_library(_homebrew_openmp_lib omp
                PATHS /usr/local/opt/libomp/lib
                /opt/homebrew/opt/libomp/lib
                NO_DEFAULT_PATH)

        if(_homebrew_openmp_include AND _homebrew_openmp_lib)
            message(STATUS "  Found Homebrew OpenMP:")
            message(STATUS "    Include: ${_homebrew_openmp_include}")
            message(STATUS "    Library: ${_homebrew_openmp_lib}")

            target_include_directories(openmp::openmp INTERFACE
                    ${_homebrew_openmp_include})
            target_link_libraries(openmp::openmp INTERFACE
                    ${_homebrew_openmp_lib})
            target_compile_options(openmp::openmp INTERFACE
                    -Xclang -fopenmp)
        endif()
    endif()
endif()

# ------------------- convenience function ------------------------------------
function(add_openmp_target target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "add_openmp_target: ${target} is not a valid target")
    endif()

    if(TARGET openmp::openmp)
        target_link_libraries(${target} PUBLIC openmp::openmp)
        message(STATUS "  Added OpenMP support to target: ${target}")
    endif()
endfunction()

# ------------------- runtime configuration helper ----------------------------
function(configure_openmp_runtime)
    if(TARGET openmp::openmp)
        # Add runtime library path for installed binaries
        if(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
            get_target_property(_openmp_libs openmp::openmp INTERFACE_LINK_LIBRARIES)
            foreach(_lib ${_openmp_libs})
                if(EXISTS ${_lib})
                    get_filename_component(_lib_dir ${_lib} DIRECTORY)
                    list(APPEND CMAKE_INSTALL_RPATH ${_lib_dir})
                endif()
            endforeach()
            set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ------------------- export configuration summary ----------------------------
if(TARGET openmp::openmp)
    message(STATUS "OpenMP configuration complete:")
    message(STATUS "  Runtime: ${_openmp_runtime}")
    message(STATUS "  Target:  openmp::openmp")
    if(NOT OPENMP_NUM_THREADS_DEFAULT STREQUAL "0")
        message(STATUS "  Default threads: ${OPENMP_NUM_THREADS_DEFAULT}")
    endif()

    # Set helpful variables for the parent scope
    set(OPENMP_AVAILABLE TRUE CACHE INTERNAL "OpenMP is available")
    set(OPENMP_RUNTIME_TYPE ${_openmp_runtime} CACHE INTERNAL "OpenMP runtime type")
else()
    set(OPENMP_AVAILABLE FALSE CACHE INTERNAL "OpenMP is not available")
endif()

# ------------------- testing support -----------------------------------------
if(BUILD_TESTS AND OPENMP_AVAILABLE)
    # Provide a simple OpenMP test
    function(add_openmp_test)
        set(_test_source "
#include <omp.h>
#include <iostream>
int main() {
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << \"OpenMP threads: \" << omp_get_num_threads() << std::endl;
    }
    return 0;
}")
        file(WRITE "${CMAKE_BINARY_DIR}/test_openmp.cpp" "${_test_source}")
        add_executable(test_openmp "${CMAKE_BINARY_DIR}/test_openmp.cpp")
        target_link_libraries(test_openmp PRIVATE openmp::openmp)
        add_test(NAME OpenMP_BasicTest COMMAND test_openmp)
    endfunction()
endif()