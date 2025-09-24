# EnableLAPACK.cmake
# Superbuild-style helper to provide LAPACK when FEM_NUMERIC_ENABLE_LAPACK=ON.
# Behavior:
# 1) Try find_package(LAPACK) normally (system-provided LAPACK/MKL/OpenBLAS)
# 2) If not found, fetch and build OpenBLAS as a provider (includes LAPACK)
# 3) Define LAPACK::LAPACK (and BLAS::BLAS) targets for consumers

cmake_minimum_required(VERSION 3.20)

if(NOT FEM_NUMERIC_ENABLE_LAPACK)
    message(STATUS "FEM Numeric: LAPACK backend disabled (FEM_NUMERIC_ENABLE_LAPACK=OFF)")
    return()
endif()

# First try system LAPACK
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
    message(STATUS "FEM Numeric: Found system LAPACK -> using LAPACK::LAPACK")
    set(FEM_NUMERIC_LAPACK_PROVIDER "system" CACHE STRING "LAPACK provider for FEM Numeric")
    return()
endif()

message(STATUS "FEM Numeric: LAPACK not found. Fetching OpenBLAS as LAPACK provider ...")

include(FetchContent)

# OpenBLAS provides BLAS and (optionally) LAPACK symbols. The CMake build can
# export a target we can alias. We keep settings conservative to ease builds.
set(OPENBLAS_BUILD_SHARED_LIBS ON CACHE BOOL "Build OpenBLAS shared" FORCE)
set(OPENBLAS_ENABLE_THREAD OFF CACHE BOOL "Disable OpenBLAS threading (let app decide)" FORCE)
set(OPENBLAS_USE_OPENMP OFF CACHE BOOL "Disable OpenMP in OpenBLAS" FORCE)
# Ensure LAPACK is included (requires Fortran). If no Fortran compiler exists,
# this step may fail; the user can install system LAPACK instead.
set(OPENBLAS_ENABLE_LAPACK ON CACHE BOOL "Build OpenBLAS with LAPACK" FORCE)

FetchContent_Declare(
    openblas
    GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
    GIT_TAG v0.3.30
    GIT_SHALLOW TRUE
)

set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(openblas)

# Try to identify OpenBLAS target name and create LAPACK::LAPACK alias
set(_openblas_target "")
if(TARGET OpenBLAS::OpenBLAS)
    set(_openblas_target OpenBLAS::OpenBLAS)
elseif(TARGET openblas)
    set(_openblas_target openblas)
elseif(TARGET OpenBLAS)
    set(_openblas_target OpenBLAS)
endif()

if(NOT _openblas_target)
    message(FATAL_ERROR "OpenBLAS target not found after FetchContent. Cannot provide LAPACK::LAPACK.")
endif()

# Provide LAPACK::LAPACK and BLAS::BLAS aliases for consumers
add_library(LAPACK::LAPACK ALIAS ${_openblas_target})
add_library(BLAS::BLAS ALIAS ${_openblas_target})

set(FEM_NUMERIC_LAPACK_PROVIDER "OpenBLAS(FetchContent)" CACHE STRING "LAPACK provider for FEM Numeric")
message(STATUS "FEM Numeric: Using ${FEM_NUMERIC_LAPACK_PROVIDER} as LAPACK provider")

