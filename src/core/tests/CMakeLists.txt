cmake_minimum_required(VERSION 3.20)

# ==============================================================================
# Core Library Test Suite
# ==============================================================================

message(STATUS "Configuring Core Library Tests")

# ==============================================================================
# Google Test Configuration
# ==============================================================================

# Find or fetch Google Test
if(NOT TARGET gtest)
    # Check if GTest is already available from parent project
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        # Fetch Google Test if not found
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG        v1.14.0
                GIT_SHALLOW    TRUE
        )

        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# ==============================================================================
# Test Configuration
# ==============================================================================

# Test modules that mirror the core folder structure
set(TEST_MODULES
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

# ==============================================================================
# Unit Tests
# ==============================================================================

set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unit)
set(UNIT_TEST_SOURCES)

# Main test runner (optional - gtest provides its own)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)
if(EXISTS ${TEST_MAIN})
    list(APPEND UNIT_TEST_SOURCES ${TEST_MAIN})
    set(USE_CUSTOM_MAIN TRUE)
else()
    set(USE_CUSTOM_MAIN FALSE)
endif()

# Collect unit test files from each module
foreach(module ${TEST_MODULES})
    set(MODULE_TEST_DIR ${UNIT_TEST_DIR}/${module})

    if(EXISTS ${MODULE_TEST_DIR})
        file(GLOB MODULE_TEST_SOURCES
                ${MODULE_TEST_DIR}/*_test.cpp
                ${MODULE_TEST_DIR}/*_tests.cpp
                ${MODULE_TEST_DIR}/*Test.cpp
                ${MODULE_TEST_DIR}/*Tests.cpp
                ${MODULE_TEST_DIR}/test_*.cpp
        )

        if(MODULE_TEST_SOURCES)
            list(APPEND UNIT_TEST_SOURCES ${MODULE_TEST_SOURCES})
            message(STATUS "  Found unit tests for module '${module}'")
        endif()
    endif()
endforeach()

# Create unit test executable if sources exist
if(UNIT_TEST_SOURCES)
    set(UNIT_TEST_TARGET core_unit_tests)
    add_executable(${UNIT_TEST_TARGET} ${UNIT_TEST_SOURCES})

    # Link with core library and Google Test
    if(USE_CUSTOM_MAIN)
        target_link_libraries(${UNIT_TEST_TARGET}
                PRIVATE
                core
                GTest::gtest
                GTest::gmock
        )
    else()
        target_link_libraries(${UNIT_TEST_TARGET}
                PRIVATE
                core
                GTest::gtest_main  # Provides main()
                GTest::gmock
        )
    endif()

    # Include directories
    target_include_directories(${UNIT_TEST_TARGET}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}  # For test_utils.h
            ${CMAKE_SOURCE_DIR}/src       # To access core headers with "core/..."
    )

    # Set C++ standard (inherit from core or use C++20)
    if(TARGET core)
        # Inherit from core library
    else()
        target_compile_features(${UNIT_TEST_TARGET} PRIVATE cxx_std_20)
    endif()

    # Enable warnings for tests
    if(COMMAND femsolver_enable_warnings)
        femsolver_enable_warnings(${UNIT_TEST_TARGET})
    endif()

    # Register tests with CTest
    include(GoogleTest)
    gtest_discover_tests(${UNIT_TEST_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            PROPERTIES
            LABELS "core;unit"
            DISCOVERY_TIMEOUT 30
    )

    # Add convenience targets
    add_custom_target(test_core
            COMMAND ${CMAKE_CTEST_COMMAND} -R "^core" --output-on-failure -C $<CONFIG>
            DEPENDS ${UNIT_TEST_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running Core library unit tests"
    )

    add_custom_target(test_core_verbose
            COMMAND ${UNIT_TEST_TARGET} --gtest_color=yes --gtest_print_time=1
            DEPENDS ${UNIT_TEST_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running Core library unit tests (verbose)"
    )

    message(STATUS "Core unit tests: ${UNIT_TEST_TARGET}")
else()
    message(STATUS "No unit test sources found for Core library")
endif()

# ==============================================================================
# Integration Tests (future)
# ==============================================================================

set(INTEGRATION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/integration)
if(EXISTS ${INTEGRATION_TEST_DIR})
    # Add integration test configuration here when needed
    message(STATUS "Integration tests directory found (not yet configured)")
endif()

# ==============================================================================
# Test Utilities and Helpers
# ==============================================================================

# Create test data directory
set(TEST_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_data)
file(MAKE_DIRECTORY ${TEST_DATA_DIR})

# Make test data directory and other definitions available to tests
if(TARGET ${UNIT_TEST_TARGET})
    target_compile_definitions(${UNIT_TEST_TARGET}
            PRIVATE
            TEST_DATA_DIR="${TEST_DATA_DIR}"
            TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            $<$<CONFIG:Debug>:CORE_DEBUG>
    )
endif()

# ==============================================================================
# Code Coverage
# ==============================================================================

option(ENABLE_COVERAGE "Enable code coverage for tests" OFF)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(TARGET ${UNIT_TEST_TARGET})
        target_compile_options(${UNIT_TEST_TARGET} PRIVATE --coverage -O0 -g)
        target_link_options(${UNIT_TEST_TARGET} PRIVATE --coverage)

        add_custom_target(coverage_core
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage/*
                COMMAND ${UNIT_TEST_TARGET}
                COMMAND lcov --capture --directory ${CMAKE_BINARY_DIR} --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
                COMMAND lcov --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info
                '/usr/*' '*/googletest/*' '*/tests/*' '*/test/*' '*/benchmarks/*'
                --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
                COMMAND genhtml ${CMAKE_BINARY_DIR}/coverage/coverage.info
                --output-directory ${CMAKE_BINARY_DIR}/coverage/html
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                DEPENDS ${UNIT_TEST_TARGET}
                COMMENT "Generating code coverage report"
        )

        message(STATUS "Coverage target: coverage_core")
    endif()
endif()

# ==============================================================================
# Memory Checking
# ==============================================================================

find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE AND TARGET ${UNIT_TEST_TARGET})
    add_custom_target(memcheck_core
            COMMAND ${VALGRIND_EXECUTABLE}
            --leak-check=full
            --show-leak-kinds=all
            --track-origins=yes
            --verbose
            --log-file=valgrind.log
            ./${UNIT_TEST_TARGET}
            DEPENDS ${UNIT_TEST_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running memory check with Valgrind"
    )
    message(STATUS "Memory check target: memcheck_core")
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "Core Test Configuration:")
if(TARGET ${UNIT_TEST_TARGET})
    list(LENGTH UNIT_TEST_SOURCES num_test_files)
    message(STATUS "  Unit test executable: ${UNIT_TEST_TARGET}")
    message(STATUS "  Number of test files: ${num_test_files}")
endif()
message(STATUS "  Code coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Test commands:")
message(STATUS "    make test_core         - Run unit tests via CTest")
message(STATUS "    make test_core_verbose - Run unit tests with detailed output")
if(ENABLE_COVERAGE)
    message(STATUS "    make coverage_core     - Generate coverage report")
endif()
if(VALGRIND_EXECUTABLE)
    message(STATUS "    make memcheck_core     - Run memory check")
endif()
message(STATUS "")