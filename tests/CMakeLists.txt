# ─────────────────────────────────────────────────────────────────────────────
#  tests/CMakeLists.txt   –  builds GoogleTest binaries for all test flavours
# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONS (inherit sane defaults from the top level in cache)
#      BUILD_UNIT_TESTS         ON
#      BUILD_PERFORMANCE_TESTS  OFF
#      BUILD_INTEGRATION_TESTS  ON
#      BUILD_REGRESSION_TESTS   ON
# ─────────────────────────────────────────────────────────────────────────────

option(BUILD_UNIT_TESTS        "Build fast, isolated unit tests"          ON)
option(BUILD_PERFORMANCE_TESTS "Build micro-benchmarks"                    OFF)
option(BUILD_INTEGRATION_TESTS "Build integration/system tests"           ON)
option(BUILD_REGRESSION_TESTS  "Build regression (bug-fix) tests"         ON)

# ---------------------------------------------------------------------------
#  Helper macro: create a GoogleTest executable & ctest target
#                create one executable + ctest entry
#  Name pattern:   <cat>.<relative.path.with.dots>.<file_name>
#    e.g.  unit.numeric.vector.test_vector
# ---------------------------------------------------------------------------
function(add_gtest_executable src_path label)
    # relative path from tests root, replace '/' with '.'
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${src_path}")
    string(REPLACE "/" "." rel "${rel}")
    string(REPLACE "\\" "." rel "${rel}")
    string(REPLACE ".cpp" ""  rel "${rel}")

    set(testname "${label}.${rel}")

    add_executable("${testname}" "${src_path}")
    target_link_libraries("${testname}"
            PRIVATE GTest::gtest GTest::gtest_main femsolver_options)

    femsolver_enable_warnings("${testname}")

    add_test(NAME ${testname} COMMAND "${testname}")
    set_tests_properties(${testname} PROPERTIES LABELS "${label}")
endfunction()

# ---------------------------------------------------------------------------
#  Gather tests per category
# ---------------------------------------------------------------------------
set(cat_dirs
        unit         ${BUILD_UNIT_TESTS}
        performance  ${BUILD_PERFORMANCE_TESTS}
        integration  ${BUILD_INTEGRATION_TESTS}
        regression   ${BUILD_REGRESSION_TESTS}
)

list(LENGTH cat_dirs _len)
math(EXPR _max_index "${_len} - 1")

foreach(idx RANGE 0 ${_max_index} 2)
    list(GET cat_dirs ${idx}   cat)
    math(EXPR idx1 "${idx} + 1")
    list(GET cat_dirs ${idx1}  cat_on)

    if(NOT cat_on)
        message(STATUS "  [tests] Skipping ${cat} tests (option OFF)")
        continue()
    endif()

    # *Recursive* glob – picks up any depth like unit/numeric/…
    file(GLOB_RECURSE SRC
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${cat}/*.cpp)

    list(LENGTH SRC _cnt)
    if(_cnt EQUAL 0)
        message(WARNING "  [tests] No sources found for category '${cat}' — check path")
    else()
        message(STATUS  "  [tests] ${cat}: ${_cnt} source(s)")
    endif()

    foreach(srcfile IN LISTS SRC)
        if(cat STREQUAL "performance")
            add_gtest_executable("${srcfile}" "perf")
        elseif(cat STREQUAL "unit")
            add_gtest_executable("${srcfile}" "unit")
        elseif(cat STREQUAL "integration")
            add_gtest_executable("${srcfile}" "int")
        else()  # regression
            add_gtest_executable("${srcfile}" "reg")
        endif()
    endforeach()
endforeach()
