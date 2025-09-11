cmake_minimum_required(VERSION 3.20)

# ============================================================================
# Project Configuration
# ============================================================================

# Allow this to be used as a subproject
if(NOT DEFINED PROJECT_NAME)
    project(FEMNumeric VERSION 0.1.0 LANGUAGES CXX)
    set(FEM_NUMERIC_STANDALONE ON)
else()
    set(FEM_NUMERIC_STANDALONE OFF)
endif()

# ============================================================================
# Options (only set defaults if standalone)
# ============================================================================

if(FEM_NUMERIC_STANDALONE)
    option(FEM_NUMERIC_BUILD_TESTS "Build unit tests" ON)
    option(FEM_NUMERIC_BUILD_BENCHMARKS "Build benchmarks" OFF)
    option(FEM_NUMERIC_BUILD_EXAMPLES "Build examples" OFF)
    option(FEM_NUMERIC_BUILD_SHARED "Build as shared library" ON)
else()
    # When part of larger FEM project, inherit or set different defaults
    option(FEM_NUMERIC_BUILD_TESTS "Build numeric library tests" ${FEM_BUILD_TESTS})
    option(FEM_NUMERIC_BUILD_BENCHMARKS "Build numeric benchmarks" ${FEM_BUILD_BENCHMARKS})
    option(FEM_NUMERIC_BUILD_EXAMPLES "Build numeric examples" ${FEM_BUILD_EXAMPLES})
    option(FEM_NUMERIC_BUILD_SHARED "Build as shared library" ${FEM_BUILD_SHARED_LIBS})
endif()

# Additional options
option(FEM_NUMERIC_ENABLE_COVERAGE "Enable code coverage" OFF)
option(FEM_NUMERIC_ENABLE_MEMCHECK "Enable memory checking with Valgrind" OFF)
option(FEM_NUMERIC_USE_SANITIZERS "Enable address and UB sanitizers" OFF)
option(FEM_NUMERIC_HEADER_ONLY "Use as header-only library" OFF)
option(FEM_NUMERIC_ENABLE_SIMD "Enable SIMD optimizations" ON)

# ============================================================================
# Parallel Computing Configuration
# ============================================================================

# Determine default parallel settings based on build type and parent project
if(FEM_NUMERIC_STANDALONE)
    # Standalone defaults
    if(NOT DEFINED FEM_NUMERIC_ENABLE_PARALLEL)
        if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            set(FEM_NUMERIC_ENABLE_PARALLEL ON)
        else()
            set(FEM_NUMERIC_ENABLE_PARALLEL OFF)
        endif()
    endif()

    option(FEM_NUMERIC_ENABLE_PARALLEL "Enable all parallelism" ${FEM_NUMERIC_ENABLE_PARALLEL})
    option(FEM_NUMERIC_ENABLE_OPENMP "Enable OpenMP parallelism" ${FEM_NUMERIC_ENABLE_PARALLEL})
    option(FEM_NUMERIC_ENABLE_TBB "Enable Intel TBB parallelism" ${FEM_NUMERIC_ENABLE_PARALLEL})
    option(FEM_NUMERIC_ENABLE_MPI "Enable MPI parallelism" OFF)
    option(FEM_NUMERIC_ENABLE_HYBRID "Enable hybrid parallel combinations" ${FEM_NUMERIC_ENABLE_PARALLEL})
    #option(FEM_NUMERIC_FETCH_MISSING "Auto-fetch missing parallel libraries" ON)
else()
    # Inherit from parent project
    option(FEM_NUMERIC_ENABLE_PARALLEL "Enable parallelism in numeric lib"
            ${ENABLE_PARALLEL})
    option(FEM_NUMERIC_ENABLE_OPENMP "Enable OpenMP in numeric lib"
            ${ENABLE_OPENMP})
    option(FEM_NUMERIC_ENABLE_TBB "Enable TBB in numeric lib"
            ${TBB_FETCH})
    option(FEM_NUMERIC_ENABLE_MPI "Enable MPI in numeric lib"
            ${ENABLE_MPI})
    option(FEM_NUMERIC_ENABLE_HYBRID "Enable hybrid parallel in numeric lib"
            ${ENABLE_HYBRID_PARALLEL})
    #option(FEM_NUMERIC_FETCH_MISSING "Auto-fetch missing libs"
    #        ${FETCH_MISSING})
endif()

# Additional parallel options
option(FEM_NUMERIC_TBB_MALLOC "Use TBB scalable allocator" OFF)
option(FEM_NUMERIC_MPI_OPTIONAL "Make MPI optional at runtime" ON)
option(FEM_NUMERIC_BUNDLED_MPI "Build our own MPI if needed" OFF)
option(FEM_NUMERIC_SUPPRESS_PARALLEL_ATOMIC_WARNING "Suppress note about parallel assembly without atomics" OFF)
set(FEM_NUMERIC_TBB_VERSION "2021.12.0" CACHE STRING "TBB version to fetch if needed")
set(FEM_NUMERIC_MPI_VENDOR "auto" CACHE STRING "MPI vendor (auto|openmpi|mpich)")

# ============================================================================
# C++ Standard and Compiler Settings
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ============================================================================
# Compiler Detection and Settings
# ============================================================================

include(CheckCXXCompilerFlag)

# Function to add compiler flag if supported
function(add_compiler_flag_if_supported FLAG)
    string(MAKE_C_IDENTIFIER ${FLAG} FLAG_VAR)
    check_cxx_compiler_flag(${FLAG} SUPPORTS_${FLAG_VAR})
    if(SUPPORTS_${FLAG_VAR})
        add_compile_options(${FLAG})
    endif()
endfunction()

# Common warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compiler_flag_if_supported(-Wall)
    add_compiler_flag_if_supported(-Wextra)
    add_compiler_flag_if_supported(-Wpedantic)
    add_compiler_flag_if_supported(-Wcast-align)
    add_compiler_flag_if_supported(-Wcast-qual)
    add_compiler_flag_if_supported(-Wconversion)
    add_compiler_flag_if_supported(-Wformat=2)
    add_compiler_flag_if_supported(-Wnon-virtual-dtor)
    add_compiler_flag_if_supported(-Wold-style-cast)
    add_compiler_flag_if_supported(-Woverloaded-virtual)
    add_compiler_flag_if_supported(-Wsign-conversion)

    # Debug-specific flags
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compiler_flag_if_supported(-g3)
        add_compiler_flag_if_supported(-Og)
        add_compile_definitions(FEM_DEBUG)
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compiler_flag_if_supported(-O3)
        add_compiler_flag_if_supported(-march=native)
        if(FEM_NUMERIC_ENABLE_SIMD)
            add_compiler_flag_if_supported(-mavx2)
            add_compiler_flag_if_supported(-mfma)
        endif()
    endif()

    # Sanitizers
    if(FEM_NUMERIC_USE_SANITIZERS)
        add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()

    # Coverage
    if(FEM_NUMERIC_ENABLE_COVERAGE)
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(
            /W4                 # Warning level 4
            /permissive-        # Conformance mode
            /Zc:__cplusplus     # Enable __cplusplus macro
            /EHsc               # Enable C++ exceptions
    )

    # Debug-specific
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1)
        add_compile_definitions(FEM_DEBUG)
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Ob2 /GL)
        if(FEM_NUMERIC_ENABLE_SIMD)
            add_compile_options(/arch:AVX2)
        endif()
    endif()
endif()

# ============================================================================
# Configure Parallel Support
# ============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include the master parallel configuration
# This will handle OpenMP, TBB, and MPI configuration
include(cmake/ConfigureParallel_Numeric.cmake)

# ============================================================================
# Automatically Find All Headers
# ============================================================================

# Get all header files recursively
file(GLOB_RECURSE FEM_NUMERIC_HEADERS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS
        "include/*.hpp"
        "include/*.h"
)

# Create source groups for IDE organization
foreach(HEADER ${FEM_NUMERIC_HEADERS})
    get_filename_component(HEADER_PATH "${HEADER}" PATH)
    string(REPLACE "/" "\\" HEADER_GROUP "${HEADER_PATH}")
    source_group("${HEADER_GROUP}" FILES "${HEADER}")
endforeach()

# ============================================================================
# Create Library Target
# ============================================================================

if(FEM_NUMERIC_HEADER_ONLY)
    # Header-only library
    add_library(fem_numeric INTERFACE)
    add_library(fem::numeric ALIAS fem_numeric)

    target_include_directories(fem_numeric INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    target_compile_features(fem_numeric INTERFACE cxx_std_20)

else()
    # Find source files if they exist (for future compiled components)
    file(GLOB_RECURSE FEM_NUMERIC_SOURCES
            CONFIGURE_DEPENDS
            "src/*.cpp"
            "src/*.cc"
    )

    # Determine library type
    if(FEM_NUMERIC_BUILD_SHARED)
        set(LIBRARY_TYPE SHARED)
    else()
        set(LIBRARY_TYPE STATIC)
    endif()

    # Always create as interface library for now (since we're header-only currently)
    add_library(fem_numeric INTERFACE)
    add_library(fem::numeric ALIAS fem_numeric)

    # Set include directories for the interface library
    target_include_directories(fem_numeric INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    target_compile_features(fem_numeric INTERFACE cxx_std_20)

    # If we have source files in the future, we'll convert to a compiled library
    if(FEM_NUMERIC_SOURCES)
        message(STATUS "Source files found - converting to compiled library")
        # Remove the interface library
        # add_library(fem_numeric ${LIBRARY_TYPE} ${FEM_NUMERIC_SOURCES})

        # For now, keeping as interface since we're header-only
        message(STATUS "Currently maintaining header-only configuration")

        # Organize sources by parallelism (for future use)
        set(FEM_NUMERIC_SERIAL_SOURCES "")
        set(FEM_NUMERIC_PARALLEL_SOURCES "")

        foreach(SOURCE ${FEM_NUMERIC_SOURCES})
            if(SOURCE MATCHES ".*parallel.*" OR SOURCE MATCHES ".*openmp.*" OR
                    SOURCE MATCHES ".*tbb.*" OR SOURCE MATCHES ".*mpi.*")
                list(APPEND FEM_NUMERIC_PARALLEL_SOURCES ${SOURCE})
            else()
                list(APPEND FEM_NUMERIC_SERIAL_SOURCES ${SOURCE})
            endif()
        endforeach()
    endif()
endif()

# Add compile definitions
target_compile_definitions(fem_numeric INTERFACE
        $<$<CONFIG:Debug>:FEM_DEBUG>
        $<$<BOOL:${FEM_NUMERIC_ENABLE_SIMD}>:FEM_NUMERIC_SIMD_ENABLED>
        $<$<BOOL:${FEM_NUMERIC_PARALLEL_AVAILABLE}>:FEM_NUMERIC_PARALLEL>
        $<$<BOOL:${FEM_NUMERIC_SUPPRESS_PARALLEL_ATOMIC_WARNING}>:FEM_NUMERIC_SUPPRESS_PARALLEL_ATOMIC_WARNING>
)

# ============================================================================
# Link Parallel Support
# ============================================================================

if(FEM_NUMERIC_PARALLEL_AVAILABLE)
    # Link parallel support to main library
    target_link_libraries(fem_numeric INTERFACE fem::numeric::parallel)

    # Install parallel configuration header
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/include/fem_numeric_parallel_config.h
            ${CMAKE_CURRENT_BINARY_DIR}/include/fem_numeric_tbb_config.h
            ${CMAKE_CURRENT_BINARY_DIR}/include/fem_numeric_mpi_wrapper.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fem_numeric
            OPTIONAL  # Some headers might not exist depending on configuration
    )
endif()

# ============================================================================
# Dependencies (if any)
# ============================================================================

# Example: Find external dependencies if needed
# find_package(Eigen3 3.4 QUIET)
# if(Eigen3_FOUND)
#     target_link_libraries(fem_numeric INTERFACE Eigen3::Eigen)
#     target_compile_definitions(fem_numeric INTERFACE FEM_HAS_EIGEN)
# endif()

# ============================================================================
# Testing
# ============================================================================

if(FEM_NUMERIC_BUILD_TESTS)
    enable_testing()
    include(CTest)

    # Set test configuration for subdirectory
    set(FEM_NUMERIC_TEST_PARALLEL ${FEM_NUMERIC_PARALLEL_AVAILABLE})

    # Add tests subdirectory if it exists
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_subdirectory(tests)
    endif()

    # Add parallel-specific tests if available
    if(FEM_NUMERIC_PARALLEL_AVAILABLE)
        # Create parallel configuration test
        set(_test_source "${CMAKE_CURRENT_BINARY_DIR}/test_parallel_config.cpp")
        file(WRITE ${_test_source} "
#include <fem_numeric_parallel_config.h>
#include <iostream>

int main() {
    using namespace fem::numeric::parallel;
    std::cout << \"Parallel configuration test:\" << std::endl;
    std::cout << \"  OpenMP: \" << openmp_available() << std::endl;
    std::cout << \"  TBB: \" << tbb_available() << std::endl;
    std::cout << \"  MPI: \" << mpi_available() << std::endl;

    if (any_available()) {
        std::cout << \"Parallel support: ENABLED\" << std::endl;
        return 0;
    } else {
        std::cout << \"Parallel support: DISABLED\" << std::endl;
        return 1;
    }
}
")

        add_executable(fem_numeric_test_parallel ${_test_source})
        target_link_libraries(fem_numeric_test_parallel PRIVATE fem::numeric)

        # Add basic parallel test
        add_test(NAME FEMNumeric_ParallelConfig COMMAND fem_numeric_test_parallel)

        # OpenMP test
        if(FEM_NUMERIC_OPENMP_AVAILABLE)
            add_test(NAME FEMNumeric_OpenMPTest
                    COMMAND fem_numeric_test_parallel
                    CONFIGURATIONS Release RelWithDebInfo)
            set_tests_properties(FEMNumeric_OpenMPTest PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=4")
        endif()

        # TBB test
        if(FEM_NUMERIC_TBB_AVAILABLE)
            add_test(NAME FEMNumeric_TBBTest
                    COMMAND fem_numeric_test_parallel
                    CONFIGURATIONS Release RelWithDebInfo)
        endif()

        # MPI test
        if(FEM_NUMERIC_MPI_AVAILABLE)
            find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun)
            if(MPIEXEC_EXECUTABLE)
                add_test(NAME FEMNumeric_MPITest
                        COMMAND ${MPIEXEC_EXECUTABLE} -np 2
                        $<TARGET_FILE:fem_numeric_test_parallel>
                        CONFIGURATIONS Release RelWithDebInfo)
            endif()
        endif()
    endif()
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(FEM_NUMERIC_BUILD_BENCHMARKS)
    # Pass parallel configuration to benchmarks
    set(BENCHMARK_ENABLE_PARALLEL ${FEM_NUMERIC_PARALLEL_AVAILABLE})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
        add_subdirectory(benchmarks)
    endif()
endif()

# ============================================================================
# Examples
# ============================================================================

if(FEM_NUMERIC_BUILD_EXAMPLES)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        add_subdirectory(examples)
    else()
        # Create examples directory and add example
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples)

        # Copy example file if parallel is available
        if(FEM_NUMERIC_PARALLEL_AVAILABLE)
            # The example would be created from the artifact we made earlier
            message(STATUS "Examples directory created. Add numeric_parallel_example.cpp to it.")
        endif()
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install library
install(TARGETS fem_numeric
        EXPORT fem_numeric_targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.ipp"
)

# Export targets
install(EXPORT fem_numeric_targets
        FILE FEMNumericTargets.cmake
        NAMESPACE fem::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FEMNumeric
)

# Create and install package configuration files
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FEMNumericConfig.cmake.in)
    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FEMNumericConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/FEMNumericConfig.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FEMNumeric
    )
else()
    # Create an enhanced config file with parallel support info
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/FEMNumericConfig.cmake
            "# FEM Numeric Library Configuration
include(\"\${CMAKE_CURRENT_LIST_DIR}/FEMNumericTargets.cmake\")

set(FEMNumeric_FOUND TRUE)
set(FEMNumeric_VERSION ${PROJECT_VERSION})

# Parallel support information
set(FEMNumeric_PARALLEL_AVAILABLE ${FEM_NUMERIC_PARALLEL_AVAILABLE})
set(FEMNumeric_PARALLEL_COMPONENTS \"${FEM_NUMERIC_PARALLEL_COMPONENTS}\")
set(FEMNumeric_HAS_OPENMP ${FEM_NUMERIC_OPENMP_AVAILABLE})
set(FEMNumeric_HAS_TBB ${FEM_NUMERIC_TBB_AVAILABLE})
set(FEMNumeric_HAS_MPI ${FEM_NUMERIC_MPI_AVAILABLE})

# Provide parallel targets if available
if(FEMNumeric_PARALLEL_AVAILABLE)
    if(NOT TARGET fem::numeric::parallel)
        add_library(fem::numeric::parallel INTERFACE IMPORTED)
        # Link to individual components if they exist
        if(FEMNumeric_HAS_OPENMP AND TARGET fem::numeric::openmp)
            target_link_libraries(fem::numeric::parallel INTERFACE fem::numeric::openmp)
        endif()
        if(FEMNumeric_HAS_TBB AND TARGET fem::numeric::tbb)
            target_link_libraries(fem::numeric::parallel INTERFACE fem::numeric::tbb)
        endif()
        if(FEMNumeric_HAS_MPI AND TARGET fem::numeric::mpi)
            target_link_libraries(fem::numeric::parallel INTERFACE fem::numeric::mpi)
        endif()
    endif()
endif()

# Macros for client projects
macro(fem_numeric_configure_parallel)
    if(FEMNumeric_PARALLEL_AVAILABLE)
        target_link_libraries(\${ARGV0} PRIVATE fem::numeric::parallel)
    endif()
endmacro()
")
endif()

# Create version file
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/FEMNumericConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FEMNumericConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/FEMNumericConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FEMNumeric
)

# Create pkg-config file if on Unix and template exists
if(UNIX AND NOT APPLE)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fem_numeric.pc.in)
        configure_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fem_numeric.pc.in
                ${CMAKE_CURRENT_BINARY_DIR}/fem_numeric.pc
                @ONLY
        )

        install(FILES
                ${CMAKE_CURRENT_BINARY_DIR}/fem_numeric.pc
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        )
    else()
        # Create a basic pkg-config file
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fem_numeric.pc
                "prefix=${CMAKE_INSTALL_PREFIX}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}
includedir=\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}

Name: FEM Numeric
Description: FEM Numeric Library with Parallel Support
Version: ${PROJECT_VERSION}
Cflags: -I\${includedir}
Libs: -L\${libdir}
")

        install(FILES
                ${CMAKE_CURRENT_BINARY_DIR}/fem_numeric.pc
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        )
    endif()
endif()

# ============================================================================
# Export for Parent Project
# ============================================================================

# Export targets for parent project if this is a subproject
if(NOT FEM_NUMERIC_STANDALONE)
    # Make our targets available to parent
    set(FEM_NUMERIC_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            PARENT_SCOPE)
    set(FEM_NUMERIC_LIBRARIES fem::numeric PARENT_SCOPE)
    set(FEM_NUMERIC_FOUND TRUE PARENT_SCOPE)

    # Export parallel configuration to parent
    set(FEM_NUMERIC_HAS_PARALLEL ${FEM_NUMERIC_PARALLEL_AVAILABLE} PARENT_SCOPE)
    set(FEM_NUMERIC_HAS_OPENMP ${FEM_NUMERIC_OPENMP_AVAILABLE} PARENT_SCOPE)
    set(FEM_NUMERIC_HAS_TBB ${FEM_NUMERIC_TBB_AVAILABLE} PARENT_SCOPE)
    set(FEM_NUMERIC_HAS_MPI ${FEM_NUMERIC_MPI_AVAILABLE} PARENT_SCOPE)
endif()

# ============================================================================
# Status Messages
# ============================================================================

if(FEM_NUMERIC_STANDALONE OR VERBOSE)
    message(STATUS "")
    message(STATUS "===================================================")
    message(STATUS "FEM Numeric Library Configuration Summary")
    message(STATUS "===================================================")
    message(STATUS "General:")
    message(STATUS "  Version:          ${PROJECT_VERSION}")
    message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Standalone:       ${FEM_NUMERIC_STANDALONE}")
    if(NOT FEM_NUMERIC_HEADER_ONLY)
        message(STATUS "  Library Type:     ${LIBRARY_TYPE}")
    else()
        message(STATUS "  Library Type:     HEADER-ONLY")
    endif()
    message(STATUS "")
    message(STATUS "Compiler:")
    message(STATUS "  ID:               ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  Version:          ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  Standard:         C++${CMAKE_CXX_STANDARD}")
    message(STATUS "  Flags:            ${CMAKE_CXX_FLAGS}")
    message(STATUS "")
    message(STATUS "Build Options:")
    message(STATUS "  Tests:            ${FEM_NUMERIC_BUILD_TESTS}")
    message(STATUS "  Benchmarks:       ${FEM_NUMERIC_BUILD_BENCHMARKS}")
    message(STATUS "  Examples:         ${FEM_NUMERIC_BUILD_EXAMPLES}")
    message(STATUS "  Coverage:         ${FEM_NUMERIC_ENABLE_COVERAGE}")
    message(STATUS "  Sanitizers:       ${FEM_NUMERIC_USE_SANITIZERS}")
    message(STATUS "  SIMD:             ${FEM_NUMERIC_ENABLE_SIMD}")
    message(STATUS "  Header-only:      ${FEM_NUMERIC_HEADER_ONLY}")
    message(STATUS "  Suppress atomic warning: ${FEM_NUMERIC_SUPPRESS_PARALLEL_ATOMIC_WARNING}")
    message(STATUS "")
    message(STATUS "Parallelism:")
    message(STATUS "  Enabled:          ${FEM_NUMERIC_PARALLEL_AVAILABLE}")
    if(FEM_NUMERIC_PARALLEL_AVAILABLE)
        message(STATUS "  Components:       ${FEM_NUMERIC_PARALLEL_COMPONENTS}")
        message(STATUS "  Details:")
        message(STATUS "    OpenMP:         ${FEM_NUMERIC_OPENMP_AVAILABLE}")
        message(STATUS "    TBB:            ${FEM_NUMERIC_TBB_AVAILABLE}")
        if(FEM_NUMERIC_TBB_AVAILABLE)
            message(STATUS "      Version:      ${FEM_NUMERIC_TBB_VERSION}")
            message(STATUS "      Malloc:       ${FEM_NUMERIC_TBB_MALLOC}")
        endif()
        message(STATUS "    MPI:            ${FEM_NUMERIC_MPI_AVAILABLE}")
        if(FEM_NUMERIC_MPI_AVAILABLE)
            message(STATUS "      Optional:     ${FEM_NUMERIC_MPI_OPTIONAL}")
            message(STATUS "      Vendor:       ${FEM_NUMERIC_MPI_VENDOR}")
        endif()
        if(FEM_NUMERIC_ENABLE_HYBRID AND _hybrid_modes)
            message(STATUS "  Hybrid Modes:     ${_hybrid_modes}")
        endif()
    endif()
    message(STATUS "")
    message(STATUS "Installation:")
    message(STATUS "  Prefix:           ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  Bindir:           ${CMAKE_INSTALL_BINDIR}")
    message(STATUS "  Libdir:           ${CMAKE_INSTALL_LIBDIR}")
    message(STATUS "  Includedir:       ${CMAKE_INSTALL_INCLUDEDIR}")
    message(STATUS "===================================================")
    message(STATUS "")
endif()