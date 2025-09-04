# cmake/CodeCoverage.cmake

# Function to setup target for coverage
function(setup_target_for_coverage)
    set(options NONE)
    set(oneValueArgs NAME EXECUTABLE OUTPUT_DIR)
    set(multiValueArgs EXCLUDE_PATTERNS DEPENDENCIES)
    cmake_parse_arguments(Coverage "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT Coverage_NAME)
        message(FATAL_ERROR "Name not specified for coverage target")
    endif()

    if(NOT Coverage_OUTPUT_DIR)
        set(Coverage_OUTPUT_DIR ${CMAKE_BINARY_DIR}/coverage/${Coverage_NAME})
    endif()

    # Setup exclude patterns
    set(EXCLUDE_ARGS "")
    foreach(pattern ${Coverage_EXCLUDE_PATTERNS})
        list(APPEND EXCLUDE_ARGS "--exclude" "${pattern}")
    endforeach()

    # Find tools
    find_program(GCOVR_EXECUTABLE gcovr)
    find_program(LCOV_EXECUTABLE lcov)
    find_program(GENHTML_EXECUTABLE genhtml)

    if(GCOVR_EXECUTABLE)
        add_custom_target(${Coverage_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${Coverage_OUTPUT_DIR}

                # Clear previous results
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${Coverage_OUTPUT_DIR}/html
                COMMAND ${CMAKE_COMMAND} -E make_directory ${Coverage_OUTPUT_DIR}/html

                # Run the executable
                COMMAND ${Coverage_EXECUTABLE} ${Coverage_UNPARSED_ARGUMENTS}

                # Generate reports
                COMMAND ${GCOVR_EXECUTABLE}
                --root ${CMAKE_SOURCE_DIR}
                --object-directory ${CMAKE_BINARY_DIR}
                --html-details ${Coverage_OUTPUT_DIR}/html/index.html
                --xml ${Coverage_OUTPUT_DIR}/coverage.xml
                --json ${Coverage_OUTPUT_DIR}/coverage.json
                --csv ${Coverage_OUTPUT_DIR}/coverage.csv
                --txt ${Coverage_OUTPUT_DIR}/coverage.txt
                ${EXCLUDE_ARGS}
                --print-summary

                DEPENDS ${Coverage_DEPENDENCIES}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running coverage analysis for ${Coverage_NAME}"
        )

    elseif(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
        add_custom_target(${Coverage_NAME}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${Coverage_OUTPUT_DIR}

                # Zero counters
                COMMAND ${LCOV_EXECUTABLE} --zerocounters --directory .

                # Run the executable
                COMMAND ${Coverage_EXECUTABLE} ${Coverage_UNPARSED_ARGUMENTS}

                # Capture coverage
                COMMAND ${LCOV_EXECUTABLE} --capture --directory . --output-file ${Coverage_OUTPUT_DIR}/coverage.info

                # Remove unwanted files
                COMMAND ${LCOV_EXECUTABLE} --remove ${Coverage_OUTPUT_DIR}/coverage.info
                ${EXCLUDE_ARGS}
                --output-file ${Coverage_OUTPUT_DIR}/coverage.filtered.info

                # Generate HTML
                COMMAND ${GENHTML_EXECUTABLE} ${Coverage_OUTPUT_DIR}/coverage.filtered.info
                --output-directory ${Coverage_OUTPUT_DIR}/html
                --title "${Coverage_NAME} Coverage"
                --show-details
                --legend

                DEPENDS ${Coverage_DEPENDENCIES}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running coverage analysis for ${Coverage_NAME}"
        )
    else()
        message(WARNING "No coverage tools found. Install lcov+genhtml or gcovr.")
    endif()
endfunction()

# Function to add coverage to a test target
function(add_coverage_to_test TEST_TARGET)
    if(FEM_NUMERIC_ENABLE_COVERAGE)
        target_compile_options(${TEST_TARGET} PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:--coverage -fprofile-arcs -ftest-coverage>
        )
        target_link_options(${TEST_TARGET} PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:--coverage>
        )
    endif()
endfunction()