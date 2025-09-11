###############################################################################
# ConfigureParallel_Numeric.cmake
#
# Master configuration module for all parallelism in the FEM Numeric library.
# Coordinates OpenMP, MPI, and TBB to work together efficiently.
#
# Usage:
#   include(cmake/ConfigureParallel_Numeric.cmake)
#
# This will:
#   • Include all parallel modules
#   • Create unified target: fem::numeric::parallel
#   • Configure hybrid parallelism
#   • Generate configuration summary
###############################################################################

#-------------------------  include-once guard  -------------------------------
if(DEFINED _FEM_NUMERIC_PARALLEL_CONFIGURED)
    return()
endif()
set(_FEM_NUMERIC_PARALLEL_CONFIGURED TRUE)

# ============================================================================
# Default Options (can be overridden before including this file)
# ============================================================================

# Enable parallelism by default only in Release builds
if(NOT DEFINED FEM_NUMERIC_ENABLE_PARALLEL)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(FEM_NUMERIC_ENABLE_PARALLEL ON)
    else()
        set(FEM_NUMERIC_ENABLE_PARALLEL OFF)
    endif()
endif()

option(FEM_NUMERIC_ENABLE_PARALLEL "Enable all parallelism in numeric library" ${FEM_NUMERIC_ENABLE_PARALLEL})
option(FEM_NUMERIC_ENABLE_OPENMP "Enable OpenMP shared-memory parallelism" ${FEM_NUMERIC_ENABLE_PARALLEL})
option(FEM_NUMERIC_ENABLE_TBB "Enable Intel TBB task parallelism" ${FEM_NUMERIC_ENABLE_PARALLEL})
option(FEM_NUMERIC_ENABLE_MPI "Enable MPI distributed parallelism" OFF)  # Off by default as it's heavyweight
option(FEM_NUMERIC_ENABLE_HYBRID "Enable hybrid parallel combinations" ${FEM_NUMERIC_ENABLE_PARALLEL})
option(FEM_NUMERIC_FETCH_MISSING "Automatically fetch missing parallel libraries" ON)

# Propagate fetch option to individual modules
if(FEM_NUMERIC_FETCH_MISSING)
    set(FEM_NUMERIC_FETCH_OPENMP ON)
    set(FEM_NUMERIC_FETCH_TBB ON)
    set(FEM_NUMERIC_BUNDLED_MPI ${FEM_NUMERIC_ENABLE_MPI})  # Only bundle MPI if explicitly enabled
endif()

# ============================================================================
# Include Individual Parallel Modules
# ============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

# Include modules in dependency order
if(FEM_NUMERIC_ENABLE_TBB)
    include(EnableTBB_Numeric)
endif()

if(FEM_NUMERIC_ENABLE_OPENMP)
    include(EnableOpenMP_Numeric)
endif()

if(FEM_NUMERIC_ENABLE_MPI)
    include(EnableMPI_Numeric)
endif()

# ============================================================================
# Create Unified Parallel Target
# ============================================================================

add_library(fem::numeric::parallel INTERFACE IMPORTED GLOBAL)

# Track what's available
set(_parallel_components "")
set(_parallel_available FALSE)

# Link available components
if(FEM_NUMERIC_TBB_AVAILABLE)
    target_link_libraries(fem::numeric::parallel INTERFACE fem::numeric::tbb)
    list(APPEND _parallel_components "TBB")
    set(_parallel_available TRUE)
endif()

if(FEM_NUMERIC_OPENMP_AVAILABLE)
    target_link_libraries(fem::numeric::parallel INTERFACE fem::numeric::openmp)
    list(APPEND _parallel_components "OpenMP")
    set(_parallel_available TRUE)
endif()

if(FEM_NUMERIC_MPI_AVAILABLE)
    target_link_libraries(fem::numeric::parallel INTERFACE fem::numeric::mpi)
    list(APPEND _parallel_components "MPI")
    set(_parallel_available TRUE)
endif()

# Add unified compile definitions
if(_parallel_available)
    target_compile_definitions(fem::numeric::parallel INTERFACE FEM_NUMERIC_PARALLEL_ENABLED)
endif()

# ============================================================================
# Configure Hybrid Parallelism
# ============================================================================

if(FEM_NUMERIC_ENABLE_HYBRID AND _parallel_available)
    set(_hybrid_modes "")

    # OpenMP + TBB hybrid
    if(FEM_NUMERIC_OPENMP_AVAILABLE AND FEM_NUMERIC_TBB_AVAILABLE)
        target_compile_definitions(fem::numeric::parallel INTERFACE
                FEM_NUMERIC_HYBRID_OPENMP_TBB)
        list(APPEND _hybrid_modes "OpenMP+TBB")
    endif()

    # MPI + OpenMP hybrid
    if(FEM_NUMERIC_MPI_AVAILABLE AND FEM_NUMERIC_OPENMP_AVAILABLE)
        target_compile_definitions(fem::numeric::parallel INTERFACE
                FEM_NUMERIC_HYBRID_MPI_OPENMP)
        list(APPEND _hybrid_modes "MPI+OpenMP")
    endif()

    # MPI + TBB hybrid
    if(FEM_NUMERIC_MPI_AVAILABLE AND FEM_NUMERIC_TBB_AVAILABLE)
        target_compile_definitions(fem::numeric::parallel INTERFACE
                FEM_NUMERIC_HYBRID_MPI_TBB)
        list(APPEND _hybrid_modes "MPI+TBB")
    endif()

    # Full hybrid (all three)
    if(FEM_NUMERIC_MPI_AVAILABLE AND FEM_NUMERIC_OPENMP_AVAILABLE AND FEM_NUMERIC_TBB_AVAILABLE)
        target_compile_definitions(fem::numeric::parallel INTERFACE
                FEM_NUMERIC_HYBRID_FULL)
        list(APPEND _hybrid_modes "Full(MPI+OpenMP+TBB)")
    endif()
endif()

# ============================================================================
# Generate Configuration Header
# ============================================================================

set(_parallel_config_header "${CMAKE_CURRENT_BINARY_DIR}/include/fem_numeric_parallel_config.h")
configure_file(${CMAKE_CURRENT_LIST_DIR}/fem_numeric_parallel_config.h.in
        ${_parallel_config_header}
        @ONLY)

target_include_directories(fem::numeric::parallel INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Helper Functions
# ============================================================================

# Function to add parallel support to a target
function(fem_numeric_add_parallel target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "fem_numeric_add_parallel: ${target} is not a valid target")
    endif()

    if(TARGET fem::numeric::parallel)
        target_link_libraries(${target} PUBLIC fem::numeric::parallel)
        message(STATUS "Numeric: Added parallel support to ${target}")
    endif()
endfunction()

# Function to configure optimal thread counts
function(fem_numeric_configure_threads)
    if(NOT _parallel_available)
        return()
    endif()

    # Get system info
    include(ProcessorCount)
    ProcessorCount(NCORES)

    if(NCORES EQUAL 0)
        set(NCORES 4)  # Fallback
    endif()

    # Configure based on available parallelism
    if(FEM_NUMERIC_MPI_AVAILABLE)
        # With MPI, assume we might run multiple ranks per node
        set(RANKS_PER_NODE $ENV{RANKS_PER_NODE})
        if(NOT RANKS_PER_NODE)
            set(RANKS_PER_NODE 1)
        endif()

        set(THREADS_PER_RANK ${NCORES}/${RANKS_PER_NODE})

        if(FEM_NUMERIC_OPENMP_AVAILABLE)
            set(ENV{OMP_NUM_THREADS} ${THREADS_PER_RANK})
        endif()

        if(FEM_NUMERIC_TBB_AVAILABLE)
            set(ENV{TBB_NUM_THREADS} ${THREADS_PER_RANK})
        endif()
    else()
        # No MPI, use all cores
        if(FEM_NUMERIC_OPENMP_AVAILABLE AND FEM_NUMERIC_TBB_AVAILABLE)
            # Split cores between OpenMP and TBB
            set(ENV{OMP_NUM_THREADS} ${NCORES}/2)
            set(ENV{TBB_NUM_THREADS} ${NCORES}/2)
        endif()
    endif()
endfunction()

# ============================================================================
# Testing Support
# ============================================================================

if(FEM_NUMERIC_BUILD_TESTS AND _parallel_available)
    # Create test to verify parallel configuration
    enable_testing()

    set(_test_source "${CMAKE_CURRENT_BINARY_DIR}/test_parallel_config.cpp")
    file(WRITE ${_test_source} "
#include <fem_numeric_parallel_config.h>
#include <iostream>

int main() {
    std::cout << \"Parallel configuration test:\" << std::endl;
    std::cout << \"  OpenMP: \" << fem::numeric::parallel::openmp_available() << std::endl;
    std::cout << \"  TBB: \" << fem::numeric::parallel::tbb_available() << std::endl;
    std::cout << \"  MPI: \" << fem::numeric::parallel::mpi_available() << std::endl;

    if (fem::numeric::parallel::any_available()) {
        std::cout << \"Parallel support: ENABLED\" << std::endl;
        return 0;
    } else {
        std::cout << \"Parallel support: DISABLED\" << std::endl;
        return 1;
    }
}
")

    add_executable(fem_numeric_test_parallel ${_test_source})
    target_link_libraries(fem_numeric_test_parallel PRIVATE fem::numeric::parallel)
    add_test(NAME FEMNumeric_ParallelConfig COMMAND fem_numeric_test_parallel)
endif()

# ============================================================================
# Export Configuration
# ============================================================================

# Cache variables for use in parent project or elsewhere
set(FEM_NUMERIC_PARALLEL_AVAILABLE ${_parallel_available} CACHE INTERNAL
        "Parallel support available in numeric library")
set(FEM_NUMERIC_PARALLEL_COMPONENTS "${_parallel_components}" CACHE INTERNAL
        "Available parallel components")

if(NOT FEM_NUMERIC_STANDALONE)
    # Export to parent scope
    set(FEM_NUMERIC_PARALLEL_AVAILABLE ${_parallel_available} PARENT_SCOPE)
    set(FEM_NUMERIC_PARALLEL_COMPONENTS "${_parallel_components}" PARENT_SCOPE)
endif()

# ============================================================================
# Configuration Summary
# ============================================================================

if(FEM_NUMERIC_STANDALONE OR FEM_NUMERIC_VERBOSE)
    message(STATUS "")
    message(STATUS "===== FEM Numeric Parallel Configuration =====")
    message(STATUS "  Parallel Enabled:    ${_parallel_available}")

    if(_parallel_available)
        message(STATUS "  Components:          ${_parallel_components}")

        if(FEM_NUMERIC_ENABLE_HYBRID AND _hybrid_modes)
            message(STATUS "  Hybrid Modes:        ${_hybrid_modes}")
        endif()

        message(STATUS "  Individual Status:")
        message(STATUS "    OpenMP:            ${FEM_NUMERIC_OPENMP_AVAILABLE}")
        message(STATUS "    TBB:               ${FEM_NUMERIC_TBB_AVAILABLE}")
        message(STATUS "    MPI:               ${FEM_NUMERIC_MPI_AVAILABLE}")
    endif()

    message(STATUS "  Unified Target:      fem::numeric::parallel")
    message(STATUS "===============================================")
    message(STATUS "")
endif()