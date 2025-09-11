cmake_minimum_required(VERSION 3.20)

# ==============================================================================
# Core Library Benchmarks
# ==============================================================================

message(STATUS "Configuring Core Library Benchmarks")

# ==============================================================================
# Google Benchmark Configuration
# ==============================================================================

# Find or fetch Google Benchmark
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG        v1.8.3
            GIT_SHALLOW    TRUE
    )

    # Disable benchmark's tests
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googlebenchmark)
endif()

# ==============================================================================
# Benchmark Configuration
# ==============================================================================

# Modules that mirror the core folder structure
set(BENCHMARK_MODULES
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
# Collect Benchmark Sources
# ==============================================================================

set(BENCHMARK_SOURCES)

# Optional custom benchmark main
set(BENCHMARK_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_main.cpp)
if(EXISTS ${BENCHMARK_MAIN})
    list(APPEND BENCHMARK_SOURCES ${BENCHMARK_MAIN})
    set(USE_CUSTOM_BENCHMARK_MAIN TRUE)
else()
    set(USE_CUSTOM_BENCHMARK_MAIN FALSE)
endif()

# Collect benchmark files from each module
foreach(module ${BENCHMARK_MODULES})
    set(MODULE_BENCHMARK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${module})

    if(EXISTS ${MODULE_BENCHMARK_DIR})
        file(GLOB MODULE_BENCH_SOURCES
                ${MODULE_BENCHMARK_DIR}/*_benchmark.cpp
                ${MODULE_BENCHMARK_DIR}/*_bench.cpp
                ${MODULE_BENCHMARK_DIR}/*Benchmark.cpp
                ${MODULE_BENCHMARK_DIR}/*Bench.cpp
        )

        if(MODULE_BENCH_SOURCES)
            list(APPEND BENCHMARK_SOURCES ${MODULE_BENCH_SOURCES})
            message(STATUS "  Found benchmarks for module '${module}'")
        endif()
    endif()
endforeach()

# ==============================================================================
# Create Benchmark Executable
# ==============================================================================

if(BENCHMARK_SOURCES)
    set(BENCHMARK_TARGET core_benchmarks)
    add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCES})

    # Link with core library and Google Benchmark
    if(USE_CUSTOM_BENCHMARK_MAIN)
        target_link_libraries(${BENCHMARK_TARGET}
                PRIVATE
                core
                benchmark::benchmark
        )
    else()
        target_link_libraries(${BENCHMARK_TARGET}
                PRIVATE
                core
                benchmark::benchmark_main  # Provides main()
        )
    endif()

    # Include directories
    target_include_directories(${BENCHMARK_TARGET}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src  # To access core headers
    )

    # Set C++ standard (inherit from core or use C++20)
    if(TARGET core)
        # Inherit from core library
    else()
        target_compile_features(${BENCHMARK_TARGET} PRIVATE cxx_std_20)
    endif()

    # Enable warnings for benchmarks
    if(COMMAND femsolver_enable_warnings)
        femsolver_enable_warnings(${BENCHMARK_TARGET})
    endif()

    # ==============================================================================
    # Benchmark Targets
    # ==============================================================================

    # Run all benchmarks
    add_custom_target(benchmark_core
            COMMAND ${BENCHMARK_TARGET}
            --benchmark_format=console
            --benchmark_counters_tabular=true
            DEPENDS ${BENCHMARK_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running Core library benchmarks"
    )

    # Run benchmarks with JSON output
    add_custom_target(benchmark_core_json
            COMMAND ${BENCHMARK_TARGET}
            --benchmark_format=json
            --benchmark_out=${CMAKE_CURRENT_BINARY_DIR}/benchmark_results.json
            DEPENDS ${BENCHMARK_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running Core benchmarks with JSON output"
    )

    # Quick benchmarks (fewer iterations)
    add_custom_target(benchmark_core_quick
            COMMAND ${BENCHMARK_TARGET}
            --benchmark_min_time=0.01
            DEPENDS ${BENCHMARK_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running quick Core benchmarks"
    )

    # Detailed benchmarks with statistics
    add_custom_target(benchmark_core_detailed
            COMMAND ${BENCHMARK_TARGET}
            --benchmark_format=console
            --benchmark_counters_tabular=true
            --benchmark_repetitions=10
            --benchmark_report_aggregates_only=false
            --benchmark_display_aggregates_only=false
            DEPENDS ${BENCHMARK_TARGET}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running detailed Core benchmarks with statistics"
    )

    message(STATUS "Core benchmarks: ${BENCHMARK_TARGET}")

    # ==============================================================================
    # Benchmark Filters (convenience targets for specific modules)
    # ==============================================================================

    foreach(module ${BENCHMARK_MODULES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${module})
            string(TOLOWER ${module} module_lower)
            add_custom_target(benchmark_core_${module_lower}
                    COMMAND ${BENCHMARK_TARGET}
                    --benchmark_filter=".*${module}.*"
                    --benchmark_format=console
                    DEPENDS ${BENCHMARK_TARGET}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Running Core ${module} benchmarks"
            )
        endif()
    endforeach()

else()
    message(STATUS "No benchmark sources found for Core library")
endif()

# ==============================================================================
# Benchmark Data Directory
# ==============================================================================

set(BENCHMARK_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark_data)
file(MAKE_DIRECTORY ${BENCHMARK_DATA_DIR})

if(TARGET ${BENCHMARK_TARGET})
    target_compile_definitions(${BENCHMARK_TARGET}
            PRIVATE
            BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR}"
            BENCHMARK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

# ==============================================================================
# Performance Profiling Support
# ==============================================================================

# Support for perf profiling on Linux
if(UNIX AND NOT APPLE)
    find_program(PERF_EXECUTABLE perf)
    if(PERF_EXECUTABLE AND TARGET ${BENCHMARK_TARGET})
        add_custom_target(profile_core
                COMMAND ${PERF_EXECUTABLE} record -g -o perf.data -- ./${BENCHMARK_TARGET}
                COMMAND ${PERF_EXECUTABLE} report -g -i perf.data
                DEPENDS ${BENCHMARK_TARGET}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Profiling Core benchmarks with perf"
        )
        message(STATUS "Profiling target: profile_core")
    endif()
endif()

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "Core Benchmark Configuration:")
if(TARGET ${BENCHMARK_TARGET})
    list(LENGTH BENCHMARK_SOURCES num_bench_files)
    message(STATUS "  Benchmark executable: ${BENCHMARK_TARGET}")
    message(STATUS "  Number of benchmark files: ${num_bench_files}")
    message(STATUS "  Benchmark commands:")
    message(STATUS "    make benchmark_core          - Run all benchmarks")
    message(STATUS "    make benchmark_core_quick    - Run quick benchmarks")
    message(STATUS "    make benchmark_core_detailed - Run with statistics")
    message(STATUS "    make benchmark_core_json     - Output to JSON")
    message(STATUS "    make benchmark_core_<module> - Run specific module")
    if(PERF_EXECUTABLE)
        message(STATUS "    make profile_core           - Profile with perf")
    endif()
endif()
message(STATUS "")