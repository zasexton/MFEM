cmake_minimum_required(VERSION 3.20)

# ==============================================================================
# Detect if this is a standalone build or part of parent project
# ==============================================================================
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(CORE_STANDALONE_BUILD TRUE)
    project(FEMCore
            VERSION 0.1.0
            LANGUAGES CXX)

    # Standalone build needs its own options
    option(BUILD_UNIT_TESTS "Build unit tests" ON)
    option(BUILD_PERFORMANCE_TESTS "Build performance tests" OFF)
    option(WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
    option(ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

    # Set C++ standard for standalone
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED YES)
    set(CMAKE_CXX_EXTENSIONS NO)

    # Create standalone options target
    add_library(femsolver_options INTERFACE)
    target_compile_options(femsolver_options INTERFACE -fPIC)

    # Define warning macro for standalone
    macro(femsolver_enable_warnings target)
        if(MSVC)
            target_compile_options(${target} PRIVATE /permissive- /W4 /utf-8)
            if(WARNINGS_AS_ERRORS)
                target_compile_options(${target} PRIVATE /WX)
            endif()
        else()
            target_compile_options(${target} PRIVATE
                    -Wall -Wextra -Wpedantic
                    -Wshadow -Wconversion -Wfloat-equal
                    -Wnon-virtual-dtor -Wold-style-cast
                    -Woverloaded-virtual -Wimplicit-fallthrough)
            if(WARNINGS_AS_ERRORS)
                target_compile_options(${target} PRIVATE -Werror)
            endif()
        endif()
    endmacro()

    # Sanitizers for standalone debug builds
    if(ENABLE_SANITIZERS AND NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(SAN_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
        add_compile_options(${SAN_FLAGS})
        add_link_options(${SAN_FLAGS})
    endif()

    message(STATUS "Building Core Library in STANDALONE mode")
else()
    set(CORE_STANDALONE_BUILD FALSE)
    message(STATUS "Building Core Library as part of MFEMSolver")
endif()

# ==============================================================================
# Core Library Configuration
# ==============================================================================
# This library provides foundational, domain-agnostic infrastructure
# It should have NO dependencies on other FEM-specific libraries
# ==============================================================================

# Define the library name
set(LIBRARY_NAME core)

# ==============================================================================
# Core-specific Options (available in both standalone and integrated builds)
# ==============================================================================
option(CORE_ENABLE_LOGGING "Enable logging functionality" ON)
option(CORE_ENABLE_PROFILING "Enable profiling support" ON)
option(CORE_ENABLE_METRICS "Enable metrics collection" ON)
option(CORE_ENABLE_TRACING "Enable execution tracing" ${CORE_ENABLE_PROFILING})
option(CORE_ENABLE_ASSERTS "Enable assertions in release builds" OFF)

# Thread-related options (only relevant if not inheriting from parent)
if(CORE_STANDALONE_BUILD)
    option(CORE_ENABLE_THREADING "Enable multi-threading support" ON)
else()
    # Inherit from parent project's parallelism settings
    if(ENABLE_OPENMP OR ENABLE_MPI OR TBB_FETCH)
        set(CORE_ENABLE_THREADING ON)
    else()
        set(CORE_ENABLE_THREADING OFF)
    endif()
endif()

# ==============================================================================
# Collect Source Files
# ==============================================================================

# List of subdirectories to include (add new ones as they're implemented)
set(CORE_MODULES
        base
        error
        utilities
        memory
        filesystem
        io
        logging
        config
        concurrency
        events
        serialization
        metrics
        reflection
        plugins
        workflow
        tracing
)

# Initialize source and header lists
set(CORE_SOURCES)
set(CORE_HEADERS)
set(CORE_INCLUDE_DIRS)

# Collect sources from each module that exists
foreach(module ${CORE_MODULES})
    set(MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${module})

    if(EXISTS ${MODULE_DIR})
        # Collect source files
        file(GLOB MODULE_SOURCES
                ${MODULE_DIR}/*.cpp
                ${MODULE_DIR}/*.cc
                ${MODULE_DIR}/*.cxx
        )

        # Collect header files
        file(GLOB MODULE_HEADERS
                ${MODULE_DIR}/*.h
                ${MODULE_DIR}/*.hpp
                ${MODULE_DIR}/*.hxx
        )

        # Add to lists if files exist
        if(MODULE_SOURCES)
            list(APPEND CORE_SOURCES ${MODULE_SOURCES})
        endif()

        if(MODULE_HEADERS)
            list(APPEND CORE_HEADERS ${MODULE_HEADERS})
        endif()

        # Add module directory to include paths
        list(APPEND CORE_INCLUDE_DIRS ${MODULE_DIR})

        # Report what was found
        message(STATUS "  Core module '${module}': found")
    else()
        # Module directory doesn't exist yet (not implemented)
        message(STATUS "  Core module '${module}': not yet implemented")
    endif()
endforeach()

# Also check for any sources directly in core/ (if any)
file(GLOB ROOT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)
file(GLOB ROOT_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

if(ROOT_SOURCES)
    list(APPEND CORE_SOURCES ${ROOT_SOURCES})
endif()
if(ROOT_HEADERS)
    list(APPEND CORE_HEADERS ${ROOT_HEADERS})
endif()

# ==============================================================================
# Create Library
# ==============================================================================

if(CORE_SOURCES)
    # Create static library if we have source files
    add_library(${LIBRARY_NAME} STATIC ${CORE_SOURCES} ${CORE_HEADERS})

    # Set C++ standard (inherit from parent or use C++20)
    if(NOT CORE_STANDALONE_BUILD)
        # Inherit from parent
        target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
    else()
        target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
    endif()

    # Enable warnings
    if(COMMAND femsolver_enable_warnings)
        femsolver_enable_warnings(${LIBRARY_NAME})
    endif()
else()
    # Create interface library if header-only
    add_library(${LIBRARY_NAME} INTERFACE)
    if(NOT CORE_STANDALONE_BUILD)
        target_compile_features(${LIBRARY_NAME} INTERFACE cxx_std_${CMAKE_CXX_STANDARD})
    else()
        target_compile_features(${LIBRARY_NAME} INTERFACE cxx_std_20)
    endif()
endif()

# ==============================================================================
# Configure Include Directories
# ==============================================================================

# The main include directory is core/ itself
if(CORE_SOURCES)
    target_include_directories(${LIBRARY_NAME}
            PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>  # To allow #include "core/..."
            $<INSTALL_INTERFACE:include>
    )
else()
    target_include_directories(${LIBRARY_NAME}
            INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
            $<INSTALL_INTERFACE:include>
    )
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

# Core should have minimal dependencies - only system and third-party libraries
# NO dependencies on other project libraries (numeric, geometry, mesh, etc.)

# System libraries
if(CORE_ENABLE_THREADING)
    find_package(Threads REQUIRED)
    if(CORE_SOURCES)
        target_link_libraries(${LIBRARY_NAME} PUBLIC Threads::Threads)
    else()
        target_link_libraries(${LIBRARY_NAME} INTERFACE Threads::Threads)
    endif()
endif()

# Link to parent project's options if available
if(NOT CORE_STANDALONE_BUILD AND TARGET femsolver_options)
    if(CORE_SOURCES)
        target_link_libraries(${LIBRARY_NAME} PUBLIC femsolver_options)
    else()
        target_link_libraries(${LIBRARY_NAME} INTERFACE femsolver_options)
    endif()
endif()

# ==============================================================================
# Third-party libraries
# ==============================================================================

# fmt library (from parent or fetch if standalone)
if(NOT CORE_STANDALONE_BUILD)
    # Use fmt from parent if available
    if(TARGET fmt::fmt OR TARGET fmt)
        if(CORE_SOURCES)
            target_link_libraries(${LIBRARY_NAME} PUBLIC fmt::fmt)
            target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_HAS_FMT)
        else()
            target_link_libraries(${LIBRARY_NAME} INTERFACE fmt::fmt)
            target_compile_definitions(${LIBRARY_NAME} INTERFACE CORE_HAS_FMT)
        endif()
        message(STATUS "  Core using fmt from parent project")
    endif()
elseif(CORE_STANDALONE_BUILD)
    # Fetch fmt for standalone build
    include(FetchContent)
    FetchContent_Declare(fmtlib
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG        10.2.1
            GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(fmtlib)

    if(CORE_SOURCES)
        target_link_libraries(${LIBRARY_NAME} PUBLIC fmt::fmt)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_HAS_FMT)
    else()
        target_link_libraries(${LIBRARY_NAME} INTERFACE fmt::fmt)
        target_compile_definitions(${LIBRARY_NAME} INTERFACE CORE_HAS_FMT)
    endif()
    message(STATUS "  Core fetched fmt for standalone build")
endif()

# Optional: spdlog for advanced logging (example)
# find_package(spdlog QUIET)
# if(spdlog_FOUND AND CORE_ENABLE_LOGGING)
#     target_link_libraries(${LIBRARY_NAME} PUBLIC spdlog::spdlog)
#     target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_HAS_SPDLOG)
# endif()

# ==============================================================================
# Platform-Specific Configuration
# ==============================================================================

if(WIN32)
    if(CORE_SOURCES)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_PLATFORM_WINDOWS)
    else()
        target_compile_definitions(${LIBRARY_NAME} INTERFACE CORE_PLATFORM_WINDOWS)
    endif()
elseif(UNIX AND NOT APPLE)
    if(CORE_SOURCES)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_PLATFORM_LINUX)
    else()
        target_compile_definitions(${LIBRARY_NAME} INTERFACE CORE_PLATFORM_LINUX)
    endif()
elseif(APPLE)
    if(CORE_SOURCES)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_PLATFORM_MACOS)
    else()
        target_compile_definitions(${LIBRARY_NAME} INTERFACE CORE_PLATFORM_MACOS)
    endif()
endif()

# ==============================================================================
# Build Configuration
# ==============================================================================

# Debug/Release specific settings
if(CORE_SOURCES)
    target_compile_definitions(${LIBRARY_NAME}
            PUBLIC
            $<$<CONFIG:Debug>:CORE_DEBUG>
            $<$<CONFIG:Release>:CORE_RELEASE>
            $<$<CONFIG:RelWithDebInfo>:CORE_RELEASE>
            $<$<CONFIG:MinSizeRel>:CORE_RELEASE>
    )

    # Feature flags
    if(CORE_ENABLE_LOGGING)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_ENABLE_LOGGING)
    endif()

    if(CORE_ENABLE_PROFILING)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_ENABLE_PROFILING)
    endif()

    if(CORE_ENABLE_METRICS)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_ENABLE_METRICS)
    endif()

    if(CORE_ENABLE_TRACING)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_ENABLE_TRACING)
    endif()

    if(CORE_ENABLE_ASSERTS)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_ENABLE_ASSERTS)
    endif()

    if(CORE_ENABLE_THREADING)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC CORE_ENABLE_THREADING)
    endif()
endif()

# ==============================================================================
# Testing Support
# ==============================================================================

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add test subdirectory if it exists and testing is enabled
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    if(NOT CORE_STANDALONE_BUILD)
        # Use parent's test settings
        if(BUILD_UNIT_TESTS OR BUILD_PERFORMANCE_TESTS)
            add_subdirectory(tests)
            message(STATUS "  Core tests enabled (using parent settings)")
        endif()
    else()
        # Standalone test setup
        if(BUILD_UNIT_TESTS OR BUILD_PERFORMANCE_TESTS)
            # Fetch GoogleTest for standalone
            include(FetchContent)
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0
            )
            FetchContent_MakeAvailable(googletest)
            enable_testing()
            add_subdirectory(tests)
            message(STATUS "  Core tests enabled (standalone)")
        endif()
    endif()
endif()

# ==============================================================================
# Installation Rules (for standalone builds)
# ==============================================================================

if(CORE_STANDALONE_BUILD)
    include(GNUInstallDirs)

    install(TARGETS ${LIBRARY_NAME}
            EXPORT CoreTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Install headers preserving directory structure
    foreach(module ${CORE_MODULES})
        set(MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${module})
        if(EXISTS ${MODULE_DIR})
            install(DIRECTORY ${MODULE_DIR}
                    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core
                    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
            )
        endif()
    endforeach()

    # Export targets
    install(EXPORT CoreTargets
            FILE CoreTargets.cmake
            NAMESPACE Core::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Core
    )

    # Create and install config file
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CoreConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/CoreConfig.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Core
    )

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/CoreConfig.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Core
    )
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "Core Library Configuration:")
message(STATUS "  Build mode: ${CMAKE_BUILD_TYPE}")
if(CORE_STANDALONE_BUILD)
    message(STATUS "  Type: STANDALONE")
    message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
else()
    message(STATUS "  Type: INTEGRATED (part of MFEMSolver)")
    message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD} (inherited)")
endif()
if(CORE_SOURCES)
    message(STATUS "  Library type: STATIC")
    message(STATUS "  Modules found: ${CORE_MODULES}")
else()
    message(STATUS "  Library type: INTERFACE (header-only)")
endif()
message(STATUS "  Threading: ${CORE_ENABLE_THREADING}")
message(STATUS "  Logging: ${CORE_ENABLE_LOGGING}")
message(STATUS "  Profiling: ${CORE_ENABLE_PROFILING}")
message(STATUS "  Metrics: ${CORE_ENABLE_METRICS}")
message(STATUS "  Tracing: ${CORE_ENABLE_TRACING}")
message(STATUS "  Asserts: ${CORE_ENABLE_ASSERTS}")
if(TARGET fmt::fmt)
    message(STATUS "  fmt library: AVAILABLE")
endif()
if(BUILD_UNIT_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    message(STATUS "  Unit tests: ENABLED (tests/unit/)")
endif()
if(BUILD_PERFORMANCE_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    message(STATUS "  Benchmarks: ENABLED (benchmarks/)")
endif()
message(STATUS "")