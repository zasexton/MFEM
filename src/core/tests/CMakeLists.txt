cmake_minimum_required(VERSION 3.20)

# ==============================================================================
# Core Library Test Suite
# ==============================================================================
#
# This CMakeLists.txt automatically discovers and builds separate test executables
# for each test file in the core library. It follows these conventions:
#
# 1. Test Discovery:
#    - Automatically scans unit/ subdirectories for test modules
#    - No need to manually maintain module lists
#    - Recursively finds test files matching naming patterns:
#      * test_*.cpp (preferred)
#      * *_test.cpp, *_tests.cpp
#      * *Test.cpp, *Tests.cpp
#
# 2. Directory Structure:
#    - tests/unit/<module>/test_<component>.cpp
#    - Example: tests/unit/base/test_component.cpp
#    - Build output: build/tests/unit/<module>/test_<component>
#
# 3. Build System:
#    - Separate executable for each test file
#    - Automatic GoogleTest discovery and CTest integration
#    - Support for coverage, memory checking, and verbose output
#    - Individual test execution capability
#
# 4. Adding New Tests:
#    - Simply create test files following naming convention
#    - Create new module directories under unit/ as needed
#    - No build system changes required
#
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

# Automatically discover test modules by scanning unit test directory
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unit)

# Find all subdirectories in the unit test directory
file(GLOB UNIT_TEST_SUBDIRS LIST_DIRECTORIES true ${UNIT_TEST_DIR}/*)

set(TEST_MODULES)
foreach(item ${UNIT_TEST_SUBDIRS})
    if(IS_DIRECTORY ${item})
        get_filename_component(module_name ${item} NAME)
        list(APPEND TEST_MODULES ${module_name})
    endif()
endforeach()

# Sort for consistent ordering
list(SORT TEST_MODULES)

message(STATUS "Discovered test modules: ${TEST_MODULES}")

# ==============================================================================
# Unit Tests - Individual Executables
# ==============================================================================

# Function to recursively collect test files from a directory
function(collect_test_files base_dir result_var)
    # Get all .cpp files that match common test naming patterns
    file(GLOB_RECURSE test_files
        ${base_dir}/*_test.cpp
        ${base_dir}/*_tests.cpp
        ${base_dir}/*Test.cpp
        ${base_dir}/*Tests.cpp
        ${base_dir}/test_*.cpp
    )

    # Filter out any files in subdirectories that might have their own CMakeLists.txt
    set(filtered_files)
    foreach(file ${test_files})
        get_filename_component(file_dir ${file} DIRECTORY)
        # Check if there's a CMakeLists.txt in the file's directory
        if(NOT EXISTS ${file_dir}/CMakeLists.txt OR ${file_dir} STREQUAL ${base_dir})
            list(APPEND filtered_files ${file})
        endif()
    endforeach()

    set(${result_var} ${filtered_files} PARENT_SCOPE)
endfunction()

# Function to create a single test executable
function(create_test_executable test_file)
    # Get relative path from unit test directory
    file(RELATIVE_PATH rel_path ${UNIT_TEST_DIR} ${test_file})
    get_filename_component(test_dir ${rel_path} DIRECTORY)
    get_filename_component(test_name_full ${rel_path} NAME_WE)

    # Create target name
    string(REPLACE "/" "_" target_name "${test_dir}_${test_name_full}")

    # Create executable
    add_executable(${target_name} ${test_file})

    # Set output directory to match the source structure
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/unit/${test_dir}"
        OUTPUT_NAME "${test_name_full}"
    )

    # Create output directory
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/unit/${test_dir}")

    # Link with core library and Google Test
    target_link_libraries(${target_name}
        PRIVATE
        core
        GTest::gtest_main  # Provides main()
        GTest::gmock
    )

    # Include directories
    target_include_directories(${target_name}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}  # For test_utils.h
        ${CMAKE_SOURCE_DIR}/src       # To access core headers with "core/..."
    )

    # Set C++ standard (inherit from core or use C++20)
    if(TARGET core)
        # Inherit from core library
    else()
        target_compile_features(${target_name} PRIVATE cxx_std_20)
    endif()

    # Enable warnings for tests
    if(COMMAND femsolver_enable_warnings)
        femsolver_enable_warnings(${target_name})
    endif()

    # Register tests with CTest
    include(GoogleTest)
    gtest_discover_tests(${target_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/unit/${test_dir}
        PROPERTIES LABELS "core;unit;${test_dir}"
        DISCOVERY_TIMEOUT 30
    )

    set(target_name "${target_name}" PARENT_SCOPE)
endfunction()

# Collect all test files and create individual executables
set(ALL_TEST_TARGETS)
collect_test_files(${UNIT_TEST_DIR} MODULE_TEST_FILES)

if(MODULE_TEST_FILES)
    # Create individual executables for each test file
    foreach(test_file ${MODULE_TEST_FILES})
        create_test_executable(${test_file})
        list(APPEND ALL_TEST_TARGETS ${target_name})
    endforeach()

    # Group and report by module for better visibility
    foreach(module ${TEST_MODULES})
        set(MODULE_TEST_DIR ${UNIT_TEST_DIR}/${module})
        set(MODULE_FILES)

        foreach(file ${MODULE_TEST_FILES})
            string(FIND ${file} ${MODULE_TEST_DIR} pos)
            if(pos EQUAL 0)
                list(APPEND MODULE_FILES ${file})
            endif()
        endforeach()

        if(MODULE_FILES)
            list(LENGTH MODULE_FILES num_files)
            message(STATUS "  Found ${num_files} unit test(s) for module '${module}'")
        endif()
    endforeach()

    list(LENGTH MODULE_TEST_FILES total_test_files)
    list(LENGTH ALL_TEST_TARGETS total_targets)
    message(STATUS "  Total unit test files discovered: ${total_test_files}")
    message(STATUS "  Total test executables created: ${total_targets}")
endif()

# ==============================================================================
# Convenience Targets
# ==============================================================================

if(ALL_TEST_TARGETS)
    # Create a target that builds all tests
    add_custom_target(build_all_tests
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Building all core unit test executables"
    )

    # Add convenience targets for running tests
    add_custom_target(test_core
        COMMAND ${CMAKE_CTEST_COMMAND} -L "core" --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running all Core library unit tests"
    )

    # Add per-module test targets
    foreach(module ${TEST_MODULES})
        add_custom_target(test_core_${module}
            COMMAND ${CMAKE_CTEST_COMMAND} -L "${module}" --output-on-failure
            DEPENDS ${ALL_TEST_TARGETS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running Core library ${module} module tests"
        )
    endforeach()

    message(STATUS "Core unit test executables will be built in: ${CMAKE_CURRENT_BINARY_DIR}/unit/")
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
if(ALL_TEST_TARGETS)
    foreach(target ${ALL_TEST_TARGETS})
        target_compile_definitions(${target}
            PRIVATE
            TEST_DATA_DIR="${TEST_DATA_DIR}"
            TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            $<$<CONFIG:Debug>:CORE_DEBUG>
        )
    endforeach()
endif()

# ==============================================================================
# Code Coverage
# ==============================================================================

option(ENABLE_COVERAGE "Enable code coverage for tests" OFF)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND ALL_TEST_TARGETS)
    foreach(target ${ALL_TEST_TARGETS})
        target_compile_options(${target} PRIVATE --coverage -O0 -g)
        target_link_options(${target} PRIVATE --coverage)
    endforeach()

    add_custom_target(coverage_core
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage/*
        COMMAND ${CMAKE_CTEST_COMMAND} -L "core"
        COMMAND lcov --capture --directory ${CMAKE_BINARY_DIR} --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
        COMMAND lcov --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info
        '/usr/*' '*/googletest/*' '*/tests/*' '*/test/*' '*/benchmarks/*'
        --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
        COMMAND genhtml ${CMAKE_BINARY_DIR}/coverage/coverage.info
        --output-directory ${CMAKE_BINARY_DIR}/coverage/html
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Generating code coverage report"
    )

    message(STATUS "Coverage target: coverage_core")
endif()

# ==============================================================================
# Memory Checking
# ==============================================================================

find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE AND ALL_TEST_TARGETS)
    add_custom_target(memcheck_core
        COMMAND ${CMAKE_CTEST_COMMAND} -L "core" -T memcheck
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running memory check with Valgrind on all tests"
    )
    message(STATUS "Memory check target: memcheck_core")
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "Core Test Configuration:")
if(ALL_TEST_TARGETS)
    list(LENGTH MODULE_TEST_FILES num_test_files)
    list(LENGTH ALL_TEST_TARGETS num_executables)
    message(STATUS "  Test files discovered: ${num_test_files}")
    message(STATUS "  Test executables: ${num_executables}")
    message(STATUS "  Output directory: ${CMAKE_CURRENT_BINARY_DIR}/unit/")
endif()
message(STATUS "  Code coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Test commands:")
message(STATUS "    make build_all_tests   - Build all test executables")
message(STATUS "    make test_core         - Run all unit tests via CTest")
foreach(module ${TEST_MODULES})
    message(STATUS "    make test_core_${module}    - Run ${module} module tests")
endforeach()
if(ENABLE_COVERAGE)
    message(STATUS "    make coverage_core     - Generate coverage report")
endif()
if(VALGRIND_EXECUTABLE)
    message(STATUS "    make memcheck_core     - Run memory check")
endif()
message(STATUS "")