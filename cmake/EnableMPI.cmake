###############################################################################
# EnableMPI.cmake
#
# Drop this file in your project’s `cmake/` directory and include it **after**
# the `project()` command in your top-level CMakeLists.txt:
#
#   option(ENABLE_MPI "Build with MPI parallelism" ON)
#   option(FETCH_MPI  "Download and build MPI if system MPI not found" OFF)
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#   include(EnableMPI)        # defines imported target mpi::mpi
#
# What you get:
#   • IMPORTED target `mpi::mpi` (INTERFACE) – safely empty if MPI is disabled
#   • COMPILE_DEF  ENABLE_MPI               – defined when MPI is enabled
#   • Convenience function  add_mpi_target(<tgt>)
#
# Platforms supported
#   • macOS & Linux:   system Open MPI / MPICH or FetchContent(Open MPI)
#   • Windows:         Microsoft MPI / MPICH via find_package(), or
#                      FetchContent(MPICH) fallback (Open MPI is unsupported)
###############################################################################

#-------------------------  include-once guard  -------------------------------
if(DEFINED _ENABLE_MPI_INCLUDED)
    return()
endif()
set(_ENABLE_MPI_INCLUDED TRUE)

if(NOT ENABLE_MPI)
    message(STATUS "MPI support disabled (ENABLE_MPI=OFF)")
    add_library(mpi::bundled INTERFACE IMPORTED GLOBAL) # empty stub
    return()
endif()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ------------------- user-facing toggles -------------------------------------
option(ENABLE_MPI   "Build with MPI parallelism"               ON)
option(BUNDLED_MPI  "Download & build our own MPI"            ON)
set(MPI_VENDOR "auto" CACHE STRING "MPI to build (openmpi|mpich|auto)")

# ------------------- early exit if MPI disabled ------------------------------
if(NOT ENABLE_MPI)
    message(STATUS "MPI support disabled (ENABLE_MPI=OFF)")
    add_library(mpi::bundled INTERFACE IMPORTED GLOBAL) # empty stub
    return()
endif()

# ------------------- choose vendor per platform ------------------------------
if(MPI_VENDOR STREQUAL "auto")
    if(WIN32)
        set(_mpi_vendor "mpich")        # OpenMPI unsupported natively
    else()
        set(_mpi_vendor "openmpi")
    endif()
else()
    string(TOLOWER "${MPI_VENDOR}" _mpi_vendor)
endif()

# public target visible to the rest of the project
add_library(mpi::bundled INTERFACE IMPORTED GLOBAL)

include(FetchContent)
include(ExternalProject)
set(_mpi_install "${CMAKE_BINARY_DIR}/_deps/_mpi_install")
# ----------------------------- Helper macro ----------------------------------
macro(_create_missing_targets inc_dir lib_dir vendor)
    # Create missing imported targets if the vendor's CMake doesn't provide them
    if(NOT TARGET MPI::MPI_C)
        add_library(MPI::MPI_C UNKNOWN IMPORTED GLOBAL)
        set_target_properties(MPI::MPI_C PROPERTIES
                IMPORTED_LOCATION "${lib_dir}/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}"
                INTERFACE_INCLUDE_DIRECTORIES "${inc_dir}")
    endif()
    if(NOT TARGET MPI::MPI_CXX)
        # naming differences: libmpi_cxx.so (Open MPI) or libmpicxx.so (MPICH)
        if(EXISTS "${lib_dir}/libmpi_cxx${CMAKE_SHARED_LIBRARY_SUFFIX}")
            set(_cxxlib "${lib_dir}/libmpi_cxx${CMAKE_SHARED_LIBRARY_SUFFIX}")
        elseif(EXISTS "${lib_dir}/libmpicxx${CMAKE_SHARED_LIBRARY_SUFFIX}")
            set(_cxxlib "${lib_dir}/libmpicxx${CMAKE_SHARED_LIBRARY_SUFFIX}")
        elseif(WIN32)
            # Windows static .lib naming (mpich)
            file(GLOB _cand "${lib_dir}/*mpi*cxx*.lib")
            list(GET _cand 0 _cxxlib)
        else()
            message(FATAL_ERROR
                    "Cannot find the C++ MPI library inside ${lib_dir} (vendor=${vendor})")
        endif()

        add_library(MPI::MPI_CXX UNKNOWN IMPORTED GLOBAL)
        set_target_properties(MPI::MPI_CXX PROPERTIES
                IMPORTED_LOCATION "${_cxxlib}"
                INTERFACE_INCLUDE_DIRECTORIES "${inc_dir}")
    endif()
endmacro()


if(BUNDLED_MPI)
    message(STATUS "Configuring **bundled** ${_mpi_vendor} build (BUNDLED_MPI=ON)")

    if(_mpi_vendor STREQUAL "openmpi")
        include(EnsureAutotools)
        set(_mpi_ver v5.0.0)
        FetchContent_Declare(
                mpi_src
                GIT_REPOSITORY https://github.com/open-mpi/ompi.git
                GIT_TAG        ${_mpi_ver}
        )
        FetchContent_MakeAvailable(mpi_src)

        ExternalProject_Add(openmpi
                SOURCE_DIR      ${mpi_src_SOURCE_DIR}
                CONFIGURE_COMMAND ${mpi_src_SOURCE_DIR}/autogen.pl
                && ${mpi_src_SOURCE_DIR}/configure --prefix=${_mpi_install}
                --disable-mpi-fortran --disable-oshmem
                BUILD_IN_SOURCE TRUE
                BUILD_COMMAND   $(MAKE) -j
                INSTALL_COMMAND $(MAKE) install
                STEP_TARGETS    install
        )

        set(_mpi_inc "${_mpi_install}/include")
        set(_mpi_lib "${_mpi_install}/lib")

        list(PREPEND CMAKE_PREFIX_PATH "${_mpi_install}")
        find_package(MPI QUIET COMPONENTS C CXX)

        _create_missing_targets("${_mpi_inc}" "${_mpi_lib}" "openmpi")

        add_dependencies(MPI::MPI_C   openmpi)
        add_dependencies(MPI::MPI_CXX openmpi)

    elseif(_mpi_vendor STREQUAL "mpich")
        set(_mpi_ver 4.2.0)
        FetchContent_Declare(
                mpi_src
                URL      https://www.mpich.org/static/downloads/${_mpi_ver}/mpich-${_mpi_ver}.tar.gz
        )
        FetchContent_MakeAvailable(mpi_src)
        ExternalProject_Add(mpich
                SOURCE_DIR      ${mpi_src_SOURCE_DIR}
                CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=${_mpi_install}
                -DENABLE_FORTRAN=OFF
                -DENABLE_ROMIO=OFF
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DCMAKE_BUILD_TYPE=Release
                BUILD_ALWAYS    FALSE
                STEP_TARGETS    install
        )
        set(_mpi_inc "${_mpi_install}/include")
        set(_mpi_lib "${_mpi_install}/lib")

        list(PREPEND CMAKE_PREFIX_PATH "${_mpi_install}")
        find_package(MPI QUIET COMPONENTS C CXX)

        _create_missing_targets("${_mpi_inc}" "${_mpi_lib}" "mpich")

        add_dependencies(MPI::MPI_C   mpich)
        add_dependencies(MPI::MPI_CXX mpich)
    else()
        message(FATAL_ERROR "Unknown MPI_VENDOR=${_mpi_vendor}")
    endif()

    # ---------------- wire bundled target -------------------------------------
    target_link_libraries(mpi::bundled
            INTERFACE MPI::MPI_C MPI::MPI_CXX)
    target_compile_definitions(mpi::bundled
            INTERFACE ENABLE_MPI)
    return()
endif()

# ---------------- fallback: use system MPI -----------------------------------
message(STATUS "BUNDLED_MPI=OFF – trying system MPI")
find_package(MPI REQUIRED COMPONENTS C CXX)

target_link_libraries(mpi::bundled
        INTERFACE MPI::MPI_C MPI::MPI_CXX)
target_compile_definitions(mpi::bundled
        INTERFACE ENABLE_MPI)

if(NOT TARGET MPI::MPI_C OR NOT TARGET MPI::MPI_CXX)
    message(FATAL_ERROR
            "System MPI found, but CMake did not export MPI::MPI_C/_CXX targets.\n"
            "Please report your distro’s FindMPI module bug or enable BUNDLED_MPI.")
endif()
