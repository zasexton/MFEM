# ============================================================================
# Test Configuration for FEM Numeric Library
# ============================================================================

# Testing framework selection
set(FEM_NUMERIC_TEST_FRAMEWORK "GoogleTest" CACHE STRING "Testing framework to use")
set_property(CACHE FEM_NUMERIC_TEST_FRAMEWORK PROPERTY STRINGS "GoogleTest" "Catch2" "DocTest")

# ============================================================================
# Fetch Testing Framework
# ============================================================================

include(FetchContent)

if(FEM_NUMERIC_TEST_FRAMEWORK STREQUAL "GoogleTest")
    # Fetch Google Test
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    # Set test framework libraries
    set(TEST_FRAMEWORK_LIBS GTest::gtest GTest::gtest_main)
    set(TEST_FILE_PATTERN "*test*.cpp")

elseif(FEM_NUMERIC_TEST_FRAMEWORK STREQUAL "Catch2")
    # Fetch Catch2
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
            GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

    set(TEST_FRAMEWORK_LIBS Catch2::Catch2WithMain)
    set(TEST_FILE_PATTERN "*test*.cpp")

elseif(FEM_NUMERIC_TEST_FRAMEWORK STREQUAL "DocTest")
    # Fetch DocTest
    FetchContent_Declare(
            doctest
            GIT_REPOSITORY https://github.com/doctest/doctest.git
            GIT_TAG v2.4.11
            GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(doctest)

    set(TEST_FRAMEWORK_LIBS doctest::doctest)
    set(TEST_FILE_PATTERN "*test*.cpp")
endif()

# ============================================================================
# Create Test Utilities Library
# ============================================================================

add_library(fem_numeric_test_utils INTERFACE)

# Ensure all numeric test targets use C++20 features
target_compile_features(fem_numeric_test_utils INTERFACE cxx_std_20)

target_link_libraries(fem_numeric_test_utils INTERFACE
        fem::numeric
        ${TEST_FRAMEWORK_LIBS}
)

# Add test-specific compile definitions
target_compile_definitions(fem_numeric_test_utils INTERFACE
        FEM_NUMERIC_TESTING
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
)

# Add test include directories if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(fem_numeric_test_utils INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# Debug: Print include directories to verify they're set correctly
get_target_property(NUMERIC_INCLUDE_DIRS fem::numeric INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "FEM Numeric include directories: ${NUMERIC_INCLUDE_DIRS}")

# ============================================================================
# Function to Add Unit Headers for Tests with Proper Naming and Organization
# ============================================================================

function(add_headers_for_coverage TEST_EXEC_NAME CATEGORY)
    # Find all headers in the category
    file(GLOB CATEGORY_HEADERS
            "${CMAKE_SOURCE_DIR}/include/${CATEGORY}/*.h"
            "${CMAKE_SOURCE_DIR}/include/${CATEGORY}/*.hpp"
    )

    if(CATEGORY_HEADERS)
        # Add headers as sources to make them visible to coverage tools
        target_sources(${TEST_EXEC_NAME} PRIVATE ${CATEGORY_HEADERS})

        # Mark headers as header-only to avoid compilation issues
        set_source_files_properties(${CATEGORY_HEADERS} PROPERTIES
                HEADER_FILE_ONLY TRUE
        )

        # Ensure headers are instrumented for coverage
        if(FEM_NUMERIC_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            foreach(HEADER ${CATEGORY_HEADERS})
                set_property(SOURCE ${HEADER} APPEND PROPERTY
                        COMPILE_FLAGS "--coverage -fprofile-arcs -ftest-coverage"
                )
            endforeach()
        endif()
    endif()
endfunction()

# ============================================================================
# Function to Add Unit Tests with Proper Naming and Organization
# ============================================================================

function(add_unit_test_executable TEST_CATEGORY TEST_NAME TEST_FILE)
    # Create the standardized test executable name
    set(TEST_EXEC_NAME "unit_${TEST_CATEGORY}_${TEST_NAME}")

    # Skip if already added
    if(TARGET ${TEST_EXEC_NAME})
        return()
    endif()

    # Create test executable
    add_executable(${TEST_EXEC_NAME} ${TEST_FILE})

    # add headers for the associated test executables
    add_headers_for_coverage(${TEST_EXEC_NAME} ${TEST_CATEGORY})

    # Set output directory to mirror test structure
    set_target_properties(${TEST_EXEC_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/unit/${TEST_CATEGORY}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/tests/unit/${TEST_CATEGORY}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests/unit/${TEST_CATEGORY}"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/tests/unit/${TEST_CATEGORY}"
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/tests/unit/${TEST_CATEGORY}"
    )

    # Link with test utilities
    target_link_libraries(${TEST_EXEC_NAME} PRIVATE fem_numeric_test_utils)

    # Add coverage flags if enabled
    if(FEM_NUMERIC_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TEST_EXEC_NAME} PRIVATE
                --coverage
                -fprofile-arcs
                -ftest-coverage
                -g
                -O0
        )
        target_link_options(${TEST_EXEC_NAME} PRIVATE
                --coverage
                -fprofile-arcs
                -ftest-coverage
        )
    endif()

    # Add compile options
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TEST_EXEC_NAME} PRIVATE
                -Wno-unused-variable
                -Wno-unused-parameter
        )
    endif()

    # Register with CTest
    add_test(NAME ${TEST_EXEC_NAME}
            COMMAND ${TEST_EXEC_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Set test properties
    set_tests_properties(${TEST_EXEC_NAME} PROPERTIES
            TIMEOUT 30
            LABELS "numeric;unit/${TEST_CATEGORY}"
    )

    # Organize in IDE
    set_target_properties(${TEST_EXEC_NAME} PROPERTIES
            FOLDER "Tests/unit/${TEST_CATEGORY}"
    )

    # Add to global test list
    list(APPEND ALL_TEST_TARGETS ${TEST_EXEC_NAME})
    set(ALL_TEST_TARGETS ${ALL_TEST_TARGETS} PARENT_SCOPE)

    # Add to category-specific list
    list(APPEND ${TEST_CATEGORY}_TEST_TARGETS ${TEST_EXEC_NAME})
    set(${TEST_CATEGORY}_TEST_TARGETS ${${TEST_CATEGORY}_TEST_TARGETS} PARENT_SCOPE)
endfunction()

# ============================================================================
# Discover and Add Tests
# ============================================================================

# Clear test target lists
set(ALL_TEST_TARGETS "")
set(TEST_CATEGORIES "")

# Function to process a test directory
function(process_test_directory BASE_DIR CATEGORY)
    # Find all test files in this category
    file(GLOB TEST_FILES
            CONFIGURE_DEPENDS
            "${BASE_DIR}/${CATEGORY}/${TEST_FILE_PATTERN}"
    )

    if(TEST_FILES)
        message(STATUS "  Found ${CATEGORY} tests:")
        list(APPEND TEST_CATEGORIES ${CATEGORY})
        set(TEST_CATEGORIES ${TEST_CATEGORIES} PARENT_SCOPE)

        foreach(TEST_FILE ${TEST_FILES})
            get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
            message(STATUS "    - ${TEST_NAME}")
            add_unit_test_executable(${CATEGORY} ${TEST_NAME} ${TEST_FILE})
        endforeach()
    endif()
endfunction()

# Look for unit test structure
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit")
    message(STATUS "Discovering unit tests in ${CMAKE_CURRENT_SOURCE_DIR}/unit:")

    # Get all subdirectories in unit/
    file(GLOB UNIT_SUBDIRS
            LIST_DIRECTORIES true
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/unit/*"
    )

    foreach(SUBDIR ${UNIT_SUBDIRS})
        if(IS_DIRECTORY ${SUBDIR})
            get_filename_component(CATEGORY ${SUBDIR} NAME)
            if(NOT CATEGORY STREQUAL "CMakeFiles")
                process_test_directory("${CMAKE_CURRENT_SOURCE_DIR}/unit" ${CATEGORY})
            endif()
        endif()
    endforeach()
else()
    # Fallback: Look for any test structure
    message(STATUS "No unit/ directory found, searching for test files in current directory")

    # Search for directories that might contain tests
    file(GLOB TEST_DIRS
            LIST_DIRECTORIES true
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/*"
    )

    foreach(DIR ${TEST_DIRS})
        if(IS_DIRECTORY ${DIR})
            get_filename_component(CATEGORY ${DIR} NAME)
            if(NOT CATEGORY STREQUAL "CMakeFiles" AND
                    NOT CATEGORY STREQUAL "include" AND
                    NOT CATEGORY STREQUAL "data")
                # Check if this directory contains test files
                file(GLOB TEST_FILES "${DIR}/${TEST_FILE_PATTERN}")
                if(TEST_FILES)
                    process_test_directory(${CMAKE_CURRENT_SOURCE_DIR} ${CATEGORY})
                endif()
            endif()
        endif()
    endforeach()
endif()

# ============================================================================
# Create Aggregate Test Targets
# ============================================================================

# Create target to build all tests
if(ALL_TEST_TARGETS)
    add_custom_target(build_all_tests
            DEPENDS ${ALL_TEST_TARGETS}
            COMMENT "Building all FEM Numeric tests"
    )

    # Also create the original target name for compatibility
    add_custom_target(fem_numeric_build_tests
            DEPENDS ${ALL_TEST_TARGETS}
            COMMENT "Building all FEM Numeric tests"
    )
endif()

# Create target to run all tests
add_custom_target(run_all_tests
        COMMAND ${CMAKE_CTEST_COMMAND}
        --output-on-failure
        --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        -R "^unit_"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all unit tests"
        DEPENDS ${ALL_TEST_TARGETS}
)

# Also create the original target name for compatibility
add_custom_target(fem_numeric_run_tests
        COMMAND ${CMAKE_CTEST_COMMAND}
        --output-on-failure
        --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        -L numeric
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all FEM Numeric tests"
        DEPENDS ${ALL_TEST_TARGETS}
)

# Create category-specific test targets
foreach(CATEGORY ${TEST_CATEGORIES})
    if(${CATEGORY}_TEST_TARGETS)
        add_custom_target(run_${CATEGORY}_tests
                COMMAND ${CMAKE_CTEST_COMMAND}
                --output-on-failure
                --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
                -R "^unit_${CATEGORY}_"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running ${CATEGORY} unit tests"
                DEPENDS ${${CATEGORY}_TEST_TARGETS}
        )
    endif()
endforeach()

# Quick test target (runs a subset of fast tests)
add_custom_target(fem_numeric_quick_tests
        COMMAND ${CMAKE_CTEST_COMMAND}
        --output-on-failure
        --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        -L "numeric"
        -LE "slow|integration"
        --timeout 5
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running quick FEM Numeric tests"
)

# ============================================================================
# Test Coverage Configuration
# ============================================================================

if(FEM_NUMERIC_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Find required tools
        find_program(GCOV_EXECUTABLE gcov)
        find_program(LCOV_EXECUTABLE lcov)
        find_program(GENHTML_EXECUTABLE genhtml)

        if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
            # Create coverage target
            add_custom_target(coverage
                    # Clean previous coverage
                    COMMAND ${CMAKE_COMMAND} -E remove_directory coverage_html
                    COMMAND find . -name "*.gcda" -delete

                    # Run tests
                    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure

                    # Capture coverage
                    COMMAND ${LCOV_EXECUTABLE} --capture --directory . --output-file coverage.info --gcov-tool ${GCOV_EXECUTABLE}

                    # Filter out system and test files
                    COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' '*/test/*' '*/tests/*' '*/_deps/*' --output-file coverage.filtered.info

                    # Generate HTML report
                    COMMAND ${GENHTML_EXECUTABLE} coverage.filtered.info --output-directory coverage_html --title "FEM Numeric Coverage" --legend --show-details

                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMENT "Generating code coverage report"
                    DEPENDS ${ALL_TEST_TARGETS}
            )

            # Category-specific coverage targets
            foreach(CATEGORY ${TEST_CATEGORIES})
                add_custom_target(coverage_${CATEGORY}
                        COMMAND ${CMAKE_COMMAND} -E remove_directory coverage_${CATEGORY}_html
                        COMMAND find . -path "*/unit/${CATEGORY}/*" -name "*.gcda" -delete
                        COMMAND ${CMAKE_CTEST_COMMAND} -R "^unit_${CATEGORY}_" --output-on-failure
                        COMMAND ${LCOV_EXECUTABLE} --capture --directory . --output-file coverage_${CATEGORY}.info --gcov-tool ${GCOV_EXECUTABLE}
                        COMMAND ${LCOV_EXECUTABLE} --extract coverage_${CATEGORY}.info "*/${CATEGORY}/*.h" "*/${CATEGORY}/*.hpp" --output-file coverage_${CATEGORY}.filtered.info
                        COMMAND ${GENHTML_EXECUTABLE} coverage_${CATEGORY}.filtered.info --output-directory coverage_${CATEGORY}_html --title "${CATEGORY} Coverage" --legend --show-details
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                        COMMENT "Generating ${CATEGORY} coverage report"
                        DEPENDS ${${CATEGORY}_TEST_TARGETS}
                )
            endforeach()

            message(STATUS "Coverage targets available: 'make coverage' and 'make coverage_<category>'")
        else()
            message(WARNING "Coverage enabled but lcov/genhtml not found. Coverage targets will not be available.")
        endif()
    endif()
endif()

# ============================================================================
# Memory Check (Valgrind) - From Original
# ============================================================================

if(FEM_NUMERIC_ENABLE_MEMCHECK)
    find_program(VALGRIND_EXECUTABLE valgrind)

    if(NOT VALGRIND_EXECUTABLE)
        message(WARNING "Memory checking enabled but Valgrind not found")
        message(STATUS "")
        message(STATUS "To install Valgrind:")

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            if(EXISTS "/etc/debian_version")
                message(STATUS "  Ubuntu/Debian: sudo apt-get install valgrind")
            elseif(EXISTS "/etc/redhat-release")
                message(STATUS "  RHEL/CentOS/Fedora: sudo dnf install valgrind")
            elseif(EXISTS "/etc/arch-release")
                message(STATUS "  Arch Linux: sudo pacman -S valgrind")
            else()
                message(STATUS "  Linux: Install 'valgrind' using your package manager")
            endif()
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            message(STATUS "  macOS: Valgrind is not well supported on recent macOS versions")
            message(STATUS "         Consider using AddressSanitizer instead (-DFEM_NUMERIC_USE_SANITIZERS=ON)")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            message(STATUS "  Windows: Valgrind is not available on Windows")
            message(STATUS "           Consider using AddressSanitizer or Dr. Memory instead")
        endif()

        message(STATUS "")
        set(FEM_NUMERIC_ENABLE_MEMCHECK OFF CACHE BOOL "Enable memory checking" FORCE)
    else()
        message(STATUS "Valgrind found: ${VALGRIND_EXECUTABLE}")

        # Configure CTest to use Valgrind
        set(MEMORYCHECK_COMMAND ${VALGRIND_EXECUTABLE})
        set(MEMORYCHECK_COMMAND_OPTIONS
                "--leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1"
                CACHE STRING "Valgrind options"
        )
        include(CTest)

        # Create memory check target for all tests
        add_custom_target(memcheck
                COMMAND ${CMAKE_CTEST_COMMAND} -T memcheck
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running tests with Valgrind memory checking"
                DEPENDS ${ALL_TEST_TARGETS}
        )

        # Category-specific memory check targets
        foreach(CATEGORY ${TEST_CATEGORIES})
            add_custom_target(memcheck_${CATEGORY}
                    COMMAND ${CMAKE_CTEST_COMMAND} -T memcheck -R "^unit_${CATEGORY}_"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMENT "Running ${CATEGORY} tests with Valgrind"
                    DEPENDS ${${CATEGORY}_TEST_TARGETS}
            )
        endforeach()
    endif()
endif()

# ============================================================================
# Benchmarks Support - From Original
# ============================================================================

if(FEM_NUMERIC_BUILD_BENCHMARKS)
    # Check for benchmark files in unit test directories
    set(BENCHMARK_FILES "")

    # Search for benchmarks in unit test structure
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit")
        file(GLOB_RECURSE BENCHMARK_FILES_UNIT
                CONFIGURE_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/unit/*benchmark*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/unit/*bench*.cpp"
        )
        list(APPEND BENCHMARK_FILES ${BENCHMARK_FILES_UNIT})
    endif()

    # Also search in benchmarks directory if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
        file(GLOB_RECURSE BENCHMARK_FILES_DEDICATED
                CONFIGURE_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
        )
        list(APPEND BENCHMARK_FILES ${BENCHMARK_FILES_DEDICATED})
    endif()

    if(BENCHMARK_FILES)
        # Fetch Google Benchmark if needed
        FetchContent_Declare(
                googlebenchmark
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG v1.8.3
                GIT_SHALLOW TRUE
        )

        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)

        set(BENCHMARK_TARGETS "")

        # Add each benchmark
        foreach(BENCH_FILE ${BENCHMARK_FILES})
            get_filename_component(BENCH_NAME ${BENCH_FILE} NAME_WE)
            get_filename_component(BENCH_DIR ${BENCH_FILE} DIRECTORY)
            get_filename_component(BENCH_CATEGORY ${BENCH_DIR} NAME)

            # Create benchmark executable with naming convention
            set(BENCH_EXEC_NAME "benchmark_${BENCH_CATEGORY}_${BENCH_NAME}")

            add_executable(${BENCH_EXEC_NAME} ${BENCH_FILE})

            # Set output directory for benchmarks
            set_target_properties(${BENCH_EXEC_NAME} PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/${BENCH_CATEGORY}"
                    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/benchmarks/${BENCH_CATEGORY}"
                    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/benchmarks/${BENCH_CATEGORY}"
            )

            target_link_libraries(${BENCH_EXEC_NAME} PRIVATE
                    fem::numeric
                    benchmark::benchmark
                    benchmark::benchmark_main
            )

            # Register as test for CTest
            add_test(NAME ${BENCH_EXEC_NAME} COMMAND ${BENCH_EXEC_NAME})
            set_tests_properties(${BENCH_EXEC_NAME} PROPERTIES
                    LABELS "numeric;benchmark;${BENCH_CATEGORY}"
                    TIMEOUT 300
            )

            list(APPEND BENCHMARK_TARGETS ${BENCH_EXEC_NAME})
        endforeach()

        # Create target to run all benchmarks
        add_custom_target(run_benchmarks
                COMMAND ${CMAKE_CTEST_COMMAND} -L benchmark --verbose
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running FEM Numeric benchmarks"
                DEPENDS ${BENCHMARK_TARGETS}
        )

        # Also keep original target name for compatibility
        add_custom_target(fem_numeric_benchmarks
                COMMAND ${CMAKE_CTEST_COMMAND} -L benchmark --verbose
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running FEM Numeric benchmarks"
                DEPENDS ${BENCHMARK_TARGETS}
        )

        message(STATUS "  Benchmarks:            ${BENCHMARK_TARGETS}")
    endif()
endif()

# ============================================================================
# Test Installation - From Original
# ============================================================================

option(FEM_NUMERIC_INSTALL_TESTS "Install test executables" OFF)

if(FEM_NUMERIC_INSTALL_TESTS)
    # Install test executables preserving directory structure
    foreach(TEST_TARGET ${ALL_TEST_TARGETS})
        # Extract category from target name (unit_<category>_<test>)
        string(REGEX REPLACE "^unit_([^_]+)_.*" "\\1" TEST_CATEGORY ${TEST_TARGET})

        install(TARGETS ${TEST_TARGET}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests/numeric/unit/${TEST_CATEGORY}
        )
    endforeach()

    # Install benchmark executables if they exist
    if(BENCHMARK_TARGETS)
        foreach(BENCH_TARGET ${BENCHMARK_TARGETS})
            string(REGEX REPLACE "^benchmark_([^_]+)_.*" "\\1" BENCH_CATEGORY ${BENCH_TARGET})

            install(TARGETS ${BENCH_TARGET}
                    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests/numeric/benchmarks/${BENCH_CATEGORY}
            )
        endforeach()
    endif()

    # Install test data if it exists
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
        install(DIRECTORY data/
                DESTINATION ${CMAKE_INSTALL_DATADIR}/fem_numeric/test_data
        )
    endif()

    # Create a test runner script
    file(WRITE ${CMAKE_BINARY_DIR}/run_installed_tests.sh
            "#!/bin/bash
# Run installed FEM Numeric tests
TEST_DIR=\"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/tests/numeric\"
if [ -d \"\$TEST_DIR\" ]; then
    echo \"Running tests from \$TEST_DIR\"
    find \"\$TEST_DIR\" -type f -executable -exec {} \\;
else
    echo \"Test directory not found: \$TEST_DIR\"
    exit 1
fi
")

    install(PROGRAMS ${CMAKE_BINARY_DIR}/run_installed_tests.sh
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            RENAME fem_numeric_run_tests
    )
endif()

# ============================================================================
# Status Output - Enhanced from Original
# ============================================================================

if(NOT DEFINED FEM_NUMERIC_QUIET_TESTS)
    message(STATUS "")
    message(STATUS "================================================================================")
    message(STATUS "FEM Numeric Tests Configuration Summary")
    message(STATUS "================================================================================")
    message(STATUS "  Test Framework:        ${FEM_NUMERIC_TEST_FRAMEWORK}")
    message(STATUS "  Test Directory:        ${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "  Output Directory:      ${CMAKE_BINARY_DIR}/tests/unit/<category>/")
    message(STATUS "  Naming Convention:     unit_<category>_<test>")
    message(STATUS "")

    if(TEST_CATEGORIES)
        message(STATUS "  Test Categories Found:")
        foreach(CATEGORY ${TEST_CATEGORIES})
            list(LENGTH ${CATEGORY}_TEST_TARGETS CATEGORY_COUNT)
            message(STATUS "    - ${CATEGORY}: ${CATEGORY_COUNT} tests")
        endforeach()
    endif()

    list(LENGTH ALL_TEST_TARGETS NUM_TESTS)
    message(STATUS "  Total Test Count:      ${NUM_TESTS}")
    message(STATUS "")

    message(STATUS "  Features:")

    if(FEM_NUMERIC_ENABLE_COVERAGE)
        message(STATUS "    ✓ Coverage:          ENABLED")
        if(LCOV_EXECUTABLE)
            message(STATUS "      Tool:              lcov ${LCOV_VERSION}")
        endif()
        message(STATUS "      Targets:")
        message(STATUS "        make coverage                   - Full coverage report")
        foreach(CATEGORY ${TEST_CATEGORIES})
            message(STATUS "        make coverage_${CATEGORY}            - ${CATEGORY} coverage only")
        endforeach()
    else()
        message(STATUS "    ✗ Coverage:          DISABLED")
    endif()

    if(FEM_NUMERIC_ENABLE_MEMCHECK AND VALGRIND_EXECUTABLE)
        message(STATUS "    ✓ Memory Checking:   ENABLED (Valgrind)")
        message(STATUS "      Targets:")
        message(STATUS "        make memcheck                   - Check all tests")
        foreach(CATEGORY ${TEST_CATEGORIES})
            message(STATUS "        make memcheck_${CATEGORY}            - Check ${CATEGORY} tests")
        endforeach()
    else()
        message(STATUS "    ✗ Memory Checking:   DISABLED")
    endif()

    if(FEM_NUMERIC_BUILD_BENCHMARKS AND BENCHMARK_TARGETS)
        list(LENGTH BENCHMARK_TARGETS NUM_BENCHMARKS)
        message(STATUS "    ✓ Benchmarks:        ENABLED (${NUM_BENCHMARKS} found)")
        message(STATUS "      Target: make run_benchmarks")
    else()
        message(STATUS "    ✗ Benchmarks:        DISABLED or none found")
    endif()

    if(FEM_NUMERIC_INSTALL_TESTS)
        message(STATUS "    ✓ Test Installation: ENABLED")
        message(STATUS "      Install to: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/tests/numeric/")
    else()
        message(STATUS "    ✗ Test Installation: DISABLED")
    endif()

    message(STATUS "")
    message(STATUS "  Available Targets:")
    message(STATUS "    make build_all_tests         - Build all test executables")
    message(STATUS "    make run_all_tests           - Run all tests")

    foreach(CATEGORY ${TEST_CATEGORIES})
        message(STATUS "    make run_${CATEGORY}_tests          - Run ${CATEGORY} tests only")
    endforeach()

    if(FEM_NUMERIC_ENABLE_COVERAGE AND LCOV_EXECUTABLE)
        message(STATUS "    make coverage                - Generate full coverage report")
    endif()

    if(FEM_NUMERIC_ENABLE_MEMCHECK AND VALGRIND_EXECUTABLE)
        message(STATUS "    make memcheck                - Run tests with memory checking")
    endif()

    if(BENCHMARK_TARGETS)
        message(STATUS "    make run_benchmarks          - Run performance benchmarks")
    endif()

    message(STATUS "================================================================================")
    message(STATUS "")
endif()