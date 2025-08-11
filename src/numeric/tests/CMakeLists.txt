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
# Automatically Discover and Add Tests
# ============================================================================

# Function to recursively add test files
function(add_test_directory TEST_DIR)
    # Get all test files in this directory and subdirectories
    file(GLOB_RECURSE TEST_FILES
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            CONFIGURE_DEPENDS
            "${TEST_DIR}/${TEST_FILE_PATTERN}"
    )

    foreach(TEST_FILE ${TEST_FILES})
        # Extract test name from file path
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        get_filename_component(TEST_PATH ${TEST_FILE} DIRECTORY)

        # Create hierarchical test name based on path
        string(REPLACE "/" "_" TEST_PATH_NAME ${TEST_PATH})
        if(TEST_PATH_NAME)
            set(FULL_TEST_NAME "test_${TEST_PATH_NAME}_${TEST_NAME}")
        else()
            set(FULL_TEST_NAME "test_${TEST_NAME}")
        endif()

        # Skip if already added
        if(NOT TARGET ${FULL_TEST_NAME})
            # Create test executable
            add_executable(${FULL_TEST_NAME} ${TEST_FILE})

            # Link with test utilities
            target_link_libraries(${FULL_TEST_NAME} PRIVATE fem_numeric_test_utils)

            # Add compile options
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(${FULL_TEST_NAME} PRIVATE
                        -Wno-unused-variable
                        -Wno-unused-parameter
                )
            endif()

            # Register with CTest
            add_test(NAME ${FULL_TEST_NAME} COMMAND ${FULL_TEST_NAME})

            # Set test properties
            set_tests_properties(${FULL_TEST_NAME} PROPERTIES
                    TIMEOUT 30
                    LABELS "numeric;${TEST_PATH}"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )

            # Organize in IDE
            string(REPLACE "/" "\\" TEST_GROUP "Tests/${TEST_PATH}")
            set_target_properties(${FULL_TEST_NAME} PROPERTIES
                    FOLDER "${TEST_GROUP}"
            )

            # Add to test list for collective targets
            list(APPEND ALL_TEST_TARGETS ${FULL_TEST_NAME})
            set(ALL_TEST_TARGETS ${ALL_TEST_TARGETS} PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

# ============================================================================
# Discover All Test Files
# ============================================================================

# Clear test target list
set(ALL_TEST_TARGETS "")

# Automatically discover test directories
file(GLOB TEST_DIRECTORIES
        LIST_DIRECTORIES true
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS
        "*"
)

# Add tests from each directory
foreach(DIR ${TEST_DIRECTORIES})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DIR} AND NOT DIR STREQUAL "CMakeFiles")
        add_test_directory(${DIR})
    endif()
endforeach()

# Also check for test files in the root test directory
add_test_directory(".")

# ============================================================================
# Create Aggregate Test Targets
# ============================================================================

# Create target to build all tests
if(ALL_TEST_TARGETS)
    add_custom_target(fem_numeric_build_tests
            DEPENDS ${ALL_TEST_TARGETS}
            COMMENT "Building all FEM Numeric tests"
    )
endif()

# Create target to run all tests
add_custom_target(fem_numeric_run_tests
        COMMAND ${CMAKE_CTEST_COMMAND}
        --output-on-failure
        --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        -L numeric
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all FEM Numeric tests"
        DEPENDS fem_numeric_build_tests
)

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
# Test Coverage
# ============================================================================

if(FEM_NUMERIC_ENABLE_COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if(LCOV AND GENHTML)
        # Create coverage target
        add_custom_target(fem_numeric_coverage
                # Clean previous coverage info
                COMMAND ${LCOV} --directory ${CMAKE_BINARY_DIR} --zerocounters

                # Run tests
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L numeric

                # Capture coverage info
                COMMAND ${LCOV}
                --directory ${CMAKE_BINARY_DIR}
                --capture
                --output-file coverage.info

                # Remove external and test files from coverage
                COMMAND ${LCOV}
                --remove coverage.info
                '/usr/*' '*/test/*' '*/tests/*' '*/googletest/*' '*/catch2/*'
                --output-file coverage.info.cleaned

                # Generate HTML report
                COMMAND ${GENHTML}
                coverage.info.cleaned
                --output-directory ${CMAKE_BINARY_DIR}/coverage
                --title "FEM Numeric Library Coverage"
                --show-details
                --legend

                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                DEPENDS fem_numeric_build_tests
                COMMENT "Generating code coverage report"
        )

        # Print coverage summary
        add_custom_command(TARGET fem_numeric_coverage POST_BUILD
                COMMAND ${LCOV} --summary coverage.info.cleaned
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    else()
        message(STATUS "lcov/genhtml not found - coverage target will not be available")
    endif()
endif()

# ============================================================================
# Memory Check (Valgrind)
# ============================================================================

find_program(VALGRIND valgrind)
if(VALGRIND)
    add_custom_target(fem_numeric_memcheck
            COMMAND ${CMAKE_CTEST_COMMAND}
            -T memcheck
            -L numeric
            --output-on-failure
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running tests with Valgrind memory check"
    )

    # Configure Valgrind
    set(MEMORYCHECK_COMMAND_OPTIONS
            "--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose"
            CACHE STRING "Valgrind options"
    )
endif()

# ============================================================================
# Benchmarks Support
# ============================================================================

if(FEM_NUMERIC_BUILD_BENCHMARKS)
    # Check for benchmark files
    file(GLOB_RECURSE BENCHMARK_FILES
            CONFIGURE_DEPENDS
            "*benchmark*.cpp"
            "*bench*.cpp"
    )

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

        # Add each benchmark
        foreach(BENCH_FILE ${BENCHMARK_FILES})
            get_filename_component(BENCH_NAME ${BENCH_FILE} NAME_WE)
            add_executable(${BENCH_NAME} ${BENCH_FILE})
            target_link_libraries(${BENCH_NAME} PRIVATE
                    fem::numeric
                    benchmark::benchmark
                    benchmark::benchmark_main
            )

            # Register as test for CTest
            add_test(NAME ${BENCH_NAME} COMMAND ${BENCH_NAME})
            set_tests_properties(${BENCH_NAME} PROPERTIES
                    LABELS "numeric;benchmark"
                    TIMEOUT 300
            )
        endforeach()

        # Create target to run all benchmarks
        add_custom_target(fem_numeric_benchmarks
                COMMAND ${CMAKE_CTEST_COMMAND} -L benchmark --verbose
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running FEM Numeric benchmarks"
        )
    endif()
endif()

# ============================================================================
# Test Installation (optional)
# ============================================================================

option(FEM_NUMERIC_INSTALL_TESTS "Install test executables" OFF)

if(FEM_NUMERIC_INSTALL_TESTS)
    install(TARGETS ${ALL_TEST_TARGETS}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests/numeric
    )

    # Install test data if it exists
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
        install(DIRECTORY data/
                DESTINATION ${CMAKE_INSTALL_DATADIR}/fem_numeric/test_data
        )
    endif()
endif()

# ============================================================================
# Status Output
# ============================================================================

if(NOT DEFINED FEM_NUMERIC_QUIET_TESTS)
    message(STATUS "FEM Numeric Tests Configuration:")
    message(STATUS "  Test Framework:    ${FEM_NUMERIC_TEST_FRAMEWORK}")
    message(STATUS "  Test Files Found:  ${CMAKE_CURRENT_SOURCE_DIR}")

    list(LENGTH ALL_TEST_TARGETS NUM_TESTS)
    message(STATUS "  Number of Tests:   ${NUM_TESTS}")

    if(FEM_NUMERIC_ENABLE_COVERAGE)
        message(STATUS "  Coverage:          ENABLED")
    endif()

    if(VALGRIND)
        message(STATUS "  Memory Checking:   Available (valgrind)")
    endif()
endif()