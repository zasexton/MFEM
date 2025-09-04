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
option(FEM_NUMERIC_ENABLE_OPENMP "Enable OpenMP parallelization" OFF)

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

# OpenMP support
if(FEM_NUMERIC_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_compile_definitions(FEM_NUMERIC_USE_OPENMP)
    endif()
endif()

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
    endif()
endif()

# Add compile definitions
target_compile_definitions(fem_numeric INTERFACE
        $<$<CONFIG:Debug>:FEM_DEBUG>
        $<$<BOOL:${FEM_NUMERIC_ENABLE_SIMD}>:FEM_NUMERIC_SIMD_ENABLED>
)

# ============================================================================
# Dependencies (if any)
# ============================================================================

# Find dependencies if needed
# find_package(Eigen3 3.4 QUIET)
# if(Eigen3_FOUND)
#     target_link_libraries(fem_numeric PUBLIC Eigen3::Eigen)
#     target_compile_definitions(fem_numeric PUBLIC FEM_HAS_EIGEN)
# endif()

# ============================================================================
# Testing
# ============================================================================

if(FEM_NUMERIC_BUILD_TESTS)
    enable_testing()
    include(CTest)

    # Add tests subdirectory
    add_subdirectory(tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(FEM_NUMERIC_BUILD_BENCHMARKS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Examples
# ============================================================================

if(FEM_NUMERIC_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    add_subdirectory(examples)
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
    # Create a basic config file if template doesn't exist
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/FEMNumericConfig.cmake
            "# FEM Numeric Library Configuration
include(\"\${CMAKE_CURRENT_LIST_DIR}/FEMNumericTargets.cmake\")
set(FEMNumeric_FOUND TRUE)
set(FEMNumeric_VERSION ${PROJECT_VERSION})
"
    )
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
        message(STATUS "pkg-config template not found - skipping pkg-config generation")
    endif()
endif()

# ============================================================================
# Export for Parent Project
# ============================================================================

# Export targets for parent project if this is a subproject
if(NOT FEM_NUMERIC_STANDALONE)
    # Make our targets available to parent
    set(FEM_NUMERIC_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
    set(FEM_NUMERIC_LIBRARIES fem::numeric PARENT_SCOPE)
    set(FEM_NUMERIC_FOUND TRUE PARENT_SCOPE)
endif()

# ============================================================================
# Status Messages
# ============================================================================

if(FEM_NUMERIC_STANDALONE OR VERBOSE)
    message(STATUS "")
    message(STATUS "FEM Numeric Library Configuration:")
    message(STATUS "  Version:          ${PROJECT_VERSION}")
    message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Standalone:       ${FEM_NUMERIC_STANDALONE}")
    message(STATUS "  Library Type:     ${LIBRARY_TYPE}")
    message(STATUS "")
    message(STATUS "  Compiler:")
    message(STATUS "    ID:             ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "    Version:        ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "    Standard:       C++${CMAKE_CXX_STANDARD}")
    message(STATUS "")
    message(STATUS "  Options:")
    message(STATUS "    Tests:          ${FEM_NUMERIC_BUILD_TESTS}")
    message(STATUS "    Benchmarks:     ${FEM_NUMERIC_BUILD_BENCHMARKS}")
    message(STATUS "    Examples:       ${FEM_NUMERIC_BUILD_EXAMPLES}")
    message(STATUS "    Coverage:       ${FEM_NUMERIC_ENABLE_COVERAGE}")
    message(STATUS "    Sanitizers:     ${FEM_NUMERIC_USE_SANITIZERS}")
    message(STATUS "    SIMD:           ${FEM_NUMERIC_ENABLE_SIMD}")
    message(STATUS "    OpenMP:         ${FEM_NUMERIC_ENABLE_OPENMP}")
    message(STATUS "    Header-only:    ${FEM_NUMERIC_HEADER_ONLY}")
    message(STATUS "")
    message(STATUS "  Install Prefix:   ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "")
endif()