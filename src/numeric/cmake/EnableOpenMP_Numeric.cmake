###############################################################################
# EnableOpenMP_Numeric.cmake
#
# OpenMP configuration for the FEM Numeric library that works both standalone
# and as a subproject. Place in numeric/cmake/ directory.
#
# Usage:
#   option(FEM_NUMERIC_ENABLE_OPENMP "Enable OpenMP in numeric library" ON)
#   include(cmake/EnableOpenMP_Numeric.cmake)
#
# Provides:
#   • Target: fem::numeric::openmp (INTERFACE)
#   • Compile definition: FEM_NUMERIC_HAS_OPENMP
#   • Reuses parent project's OpenMP if available
###############################################################################

#-------------------------  include-once guard  -------------------------------
if(DEFINED _FEM_NUMERIC_OPENMP_INCLUDED)
    return()
endif()
set(_FEM_NUMERIC_OPENMP_INCLUDED TRUE)

# ------------------- Check parent project configuration ----------------------
# If we're part of a larger project that already configured OpenMP, reuse it
if(NOT FEM_NUMERIC_STANDALONE)
    if(TARGET openmp::openmp)
        message(STATUS "Numeric: Reusing parent project's OpenMP configuration")
        add_library(fem::numeric::openmp INTERFACE IMPORTED GLOBAL)
        target_link_libraries(fem::numeric::openmp INTERFACE openmp::openmp)
        target_compile_definitions(fem::numeric::openmp INTERFACE FEM_NUMERIC_HAS_OPENMP)
        set(FEM_NUMERIC_OPENMP_AVAILABLE TRUE CACHE INTERNAL "OpenMP available for numeric library")
        return()
    elseif(OPENMP_AVAILABLE)
        # Parent has OpenMP but different target name
        message(STATUS "Numeric: Parent project has OpenMP available")
    endif()
endif()

# ------------------- Standalone configuration --------------------------------
if(NOT FEM_NUMERIC_ENABLE_OPENMP)
    message(STATUS "Numeric: OpenMP disabled (FEM_NUMERIC_ENABLE_OPENMP=OFF)")
    add_library(fem::numeric::openmp INTERFACE IMPORTED GLOBAL)
    set(FEM_NUMERIC_OPENMP_AVAILABLE FALSE CACHE INTERNAL "OpenMP not enabled for numeric library")
    return()
endif()

message(STATUS "Numeric: Configuring OpenMP support...")

# ------------------- Find or fetch OpenMP ------------------------------------
find_package(OpenMP QUIET)

if(OpenMP_FOUND AND OpenMP_CXX_FOUND)
    message(STATUS "Numeric: Found system OpenMP ${OpenMP_CXX_VERSION}")

    # Create our interface target
    add_library(fem::numeric::openmp INTERFACE IMPORTED GLOBAL)

    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fem::numeric::openmp INTERFACE OpenMP::OpenMP_CXX)
    else()
        # Fallback for older CMake
        target_compile_options(fem::numeric::openmp INTERFACE ${OpenMP_CXX_FLAGS})
        if(OpenMP_CXX_LIBRARIES)
            target_link_libraries(fem::numeric::openmp INTERFACE ${OpenMP_CXX_LIBRARIES})
        endif()
    endif()

    target_compile_definitions(fem::numeric::openmp INTERFACE
            FEM_NUMERIC_HAS_OPENMP
            FEM_NUMERIC_USE_OPENMP
    )

    set(FEM_NUMERIC_OPENMP_AVAILABLE TRUE CACHE INTERNAL "OpenMP available for numeric library")

elseif(FEM_NUMERIC_FETCH_OPENMP)
    # Fetch LLVM OpenMP
    message(STATUS "Numeric: Fetching LLVM OpenMP...")

    include(FetchContent)

    set(_openmp_install "${CMAKE_BINARY_DIR}/_deps/numeric_openmp_install")

    FetchContent_Declare(
            numeric_llvm_openmp
            GIT_REPOSITORY https://github.com/llvm/llvm-project.git
            GIT_TAG        llvmorg-18.1.0
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  openmp
    )

    # Configure options
    set(LIBOMP_ENABLE_SHARED ON CACHE BOOL "" FORCE)
    set(LIBOMP_USE_HWLOC OFF CACHE BOOL "" FORCE)
    set(OPENMP_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(numeric_llvm_openmp)

    # Create interface target
    add_library(fem::numeric::openmp INTERFACE IMPORTED GLOBAL)

    if(TARGET omp)
        target_link_libraries(fem::numeric::openmp INTERFACE omp)
    endif()

    target_compile_options(fem::numeric::openmp INTERFACE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fopenmp>
            $<$<CXX_COMPILER_ID:MSVC>:/openmp>
    )

    target_compile_definitions(fem::numeric::openmp INTERFACE
            FEM_NUMERIC_HAS_OPENMP
            FEM_NUMERIC_USE_OPENMP
    )

    set(FEM_NUMERIC_OPENMP_AVAILABLE TRUE CACHE INTERNAL "OpenMP available for numeric library")

else()
    message(STATUS "Numeric: OpenMP not found and fetching disabled")
    add_library(fem::numeric::openmp INTERFACE IMPORTED GLOBAL)
    set(FEM_NUMERIC_OPENMP_AVAILABLE FALSE CACHE INTERNAL "OpenMP not available for numeric library")
endif()

# ------------------- Platform-specific adjustments ---------------------------
if(APPLE AND FEM_NUMERIC_OPENMP_AVAILABLE AND NOT OpenMP_FOUND)
    # Try to find Homebrew's libomp
    find_path(_brew_omp_inc omp.h
            PATHS /usr/local/opt/libomp/include /opt/homebrew/opt/libomp/include
            NO_DEFAULT_PATH)
    find_library(_brew_omp_lib omp
            PATHS /usr/local/opt/libomp/lib /opt/homebrew/opt/libomp/lib
            NO_DEFAULT_PATH)

    if(_brew_omp_inc AND _brew_omp_lib)
        message(STATUS "Numeric: Found Homebrew OpenMP")
        target_include_directories(fem::numeric::openmp INTERFACE ${_brew_omp_inc})
        target_link_libraries(fem::numeric::openmp INTERFACE ${_brew_omp_lib})
        target_compile_options(fem::numeric::openmp INTERFACE -Xclang -fopenmp)
        set(FEM_NUMERIC_OPENMP_AVAILABLE TRUE CACHE INTERNAL "OpenMP available via Homebrew")
    endif()
endif()

# ------------------- Helper macros for numeric algorithms --------------------
if(FEM_NUMERIC_OPENMP_AVAILABLE)
    # Macro to parallelize numeric loops
    macro(fem_numeric_parallel_for var start end)
        if(FEM_NUMERIC_OPENMP_AVAILABLE)
            #pragma omp parallel for
        endif()
        for(${var} = ${start}; ${var} < ${end}; ++${var})
    endmacro()

    # Report configuration
    message(STATUS "Numeric: OpenMP configuration complete")
endif()

# ------------------- Export for parent project -------------------------------
if(NOT FEM_NUMERIC_STANDALONE AND FEM_NUMERIC_OPENMP_AVAILABLE)
    set(FEM_NUMERIC_HAS_OPENMP TRUE PARENT_SCOPE)
endif()