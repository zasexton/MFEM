cmake_minimum_required (VERSION 3.24)   # needed for C++23 modules + FetchContent goodies
project (MFEMSolver
        VERSION 0.1.0
        LANGUAGES C CXX)                # we’ll add Fortran/C later if needed

# -----------------------------------------------------------------------------
# ----------- Build-time feature toggles (all OFF by default) -----------------
# -----------------------------------------------------------------------------
option (BUILD_EXAMPLES          "Build tiny driver examples"               OFF)
option (BUILD_PYTHON            "Expose pybind11 bindings"                 OFF)
option (ENABLE_SANITIZERS       "Add address/undefined sanitizers (Debug)" ON)
option (WARNINGS_AS_ERRORS      "Treat warnings as errors"                 ON)
option (BUILD_UNIT_TESTS        "Build unit tests"                         ON)
option (BUILD_PERFORMANCE_TESTS "Build micro-benchmarks"                    ON)

# Optional LAPACK backend for numeric library (header-only code may call vendor routines)
option (FEM_NUMERIC_ENABLE_LAPACK "Enable LAPACK/MKL backend for numeric algorithms" OFF)
option (FEM_NUMERIC_ENABLE_LAPACKE "Enable LAPACKE row-major interfaces (requires LAPACK)" OFF)

# Numeric decomposition tunables (compile-time constants)
set(FEM_EIGEN_BLOCK_SIZE 32 CACHE STRING "Block size for eigen tridiagonalization (fallback)")
set(FEM_WY_TILE_SIZE 256 CACHE STRING "Tile size for WY two-sided updates")
set(FEM_EIGEN_TRIDIAG_BLOCK_THRESHOLD 128 CACHE STRING "n-threshold to enable blocked tridiagonalization by default")

# -----------------------------------------------------------------------------
# ------------------------ Parallelism Options --------------------------------
# -----------------------------------------------------------------------------
option (ENABLE_MPI              "Build distributed-memory layer"           ON)
option (BUNDLED_MPI             "Build our own MPI"                        OFF)
option (ENABLE_OPENMP           "Enable OpenMP shared-memory parallelism"  ON)
option (FETCH_OPENMP            "Download LLVM OpenMP if not found"        OFF)
option (ENABLE_HYBRID_PARALLEL  "Enable MPI+OpenMP hybrid parallelism"     ON)
option (TBB_FETCH               "Automatically download oneTBB if missing" ON)
# -----------------------------------------------------------------------------
#  Compiler & language standard
# -----------------------------------------------------------------------------
# We target C++23; most compilers provide 99 % of the features already.

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        set(CMAKE_CXX_STANDARD 20)
    else()
        set(CMAKE_CXX_STANDARD 20)    # fallback, works with our bit_cast shim
        message(STATUS "CXX Compiler <10")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_STANDARD 20)
elseif (MSVC)
    # 19.38 == Visual Studio 2022 v17.8 (first with full /std:c++23 flag in CMake)
    if (MSVC_VERSION GREATER_EQUAL 1938)
        set(CMAKE_CXX_STANDARD 23)
    else()
        message(STATUS
                "MSVC tool-set ${MSVC_VERSION} lacks /std:c++23 – falling back to C++20")
        set(CMAKE_CXX_STANDARD 20)
    endif()
else()
    message(STATUS "Unknown compiler – using C++20")
    set(CMAKE_CXX_STANDARD 20)
endif()
set (CMAKE_CXX_STANDARD_REQUIRED YES)
set (CMAKE_CXX_EXTENSIONS NO)

# -----------------------------------------------------------------------------
#  Detect availability of lock-free std::atomic<double>
# -----------------------------------------------------------------------------
set(_PLATFORM_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/numeric/include/config/platform.h")
check_cxx_source_compiles("\n#include <atomic>\n#include \"${_PLATFORM_HEADER}\"\n#if !FEM_NUMERIC_HAS_ATOMIC_DOUBLE\n#error \"Atomic double not supported\"\n#endif\nint main() {\n    static_assert(std::atomic<double>::is_always_lock_free);\n    std::atomic<double> v{0.0};\n    (void)v;\n    return 0;\n}\n" FEM_HAVE_ATOMIC_DOUBLE)

if (FEM_HAVE_ATOMIC_DOUBLE)
    add_compile_definitions(FEM_NUMERIC_ENABLE_ATOMIC_ASSEMBLY=1)
else()
    # Atomics unavailable: disable parallel assembly and suppress related note
    add_compile_definitions(
        FEM_NUMERIC_SUPPRESS_PARALLEL_ATOMIC_WARNING=1
        FEM_NUMERIC_ENABLE_PARALLEL=0
    )
endif()

# -----------------------------------------------------------------------------
#  Global warning flags (compiler-specific)
# -----------------------------------------------------------------------------
macro(femsolver_enable_warnings target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /permissive- /W4 /utf-8)
        if (WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else() # GCC / Clang / AppleClang
        target_compile_options(${target} PRIVATE
                -Wall -Wextra -Wpedantic
                -Wshadow -Wconversion -Wfloat-equal
                -Wnon-virtual-dtor -Wold-style-cast
                -Woverloaded-virtual -Wimplicit-fallthrough
                -Wno-duplicate-enum)   # duplicate-enum noisy with concepts
        if (WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endmacro()

# -----------------------------------------------------------------------------
#  Sanitizers (only meaningful on GCC/Clang-like & Debug builds)
# -----------------------------------------------------------------------------
if (ENABLE_SANITIZERS AND NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (SAN_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
    add_compile_options(${SAN_FLAGS})
    add_link_options(${SAN_FLAGS})
endif()

# -----------------------------------------------------------------------------
#  Helper INTERFACE library that carries common compile flags + include dirs
# -----------------------------------------------------------------------------
add_library(femsolver_options INTERFACE)
# Position-independent code lets us link into shared libs later (Python).
target_compile_options(femsolver_options INTERFACE -fPIC)
target_include_directories(femsolver_options INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)   # future install()

# Propagate numeric tunables as preprocessor defines
target_compile_definitions(femsolver_options INTERFACE
        FEM_EIGEN_BLOCK_SIZE=${FEM_EIGEN_BLOCK_SIZE}
        FEM_WY_TILE_SIZE=${FEM_WY_TILE_SIZE}
        FEM_EIGEN_TRIDIAG_BLOCK_THRESHOLD=${FEM_EIGEN_TRIDIAG_BLOCK_THRESHOLD})

# -----------------------------------------------------------------------------
#  Load parallelism modules
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenMP (shared memory)
if (ENABLE_OPENMP)
    include(EnableMP)
endif()

# MPI (distributed memory)
if (ENABLE_MPI)
    include(EnableMPI)
endif()

if (TBB_FETCH)
    include(FetchTBB)
endif()

# -----------------------------------------------------------------------------
#  Define hybrid parallelism target
# -----------------------------------------------------------------------------
add_library(femsolver_parallel INTERFACE)

if (ENABLE_OPENMP AND TARGET openmp::openmp)
    target_link_libraries(femsolver_parallel INTERFACE openmp::openmp)
    message(STATUS "Shared-memory parallelism: OpenMP enabled")
endif()

if (ENABLE_MPI AND TARGET mpi::bundled)
    target_link_libraries(femsolver_parallel INTERFACE mpi::bundled)
    message(STATUS "Distributed-memory parallelism: MPI enabled")
endif()

if (ENABLE_HYBRID_PARALLEL AND ENABLE_MPI AND ENABLE_OPENMP)
    target_compile_definitions(femsolver_parallel INTERFACE ENABLE_HYBRID_PARALLEL)
    message(STATUS "Hybrid parallelism: MPI+OpenMP enabled")
endif()

# Link the parallel support to your options library
target_link_libraries(femsolver_options INTERFACE femsolver_parallel)

# -----------------------------------------------------------------------------
#  Optional LAPACK (for eigensolvers, QR, etc.)
# -----------------------------------------------------------------------------
if (FEM_NUMERIC_ENABLE_LAPACK)
    # Reuse the numeric library helper to provide LAPACK::LAPACK via system or OpenBLAS
    include(${CMAKE_CURRENT_SOURCE_DIR}/src/numeric/cmake/EnableLAPACK.cmake)
    if (TARGET LAPACK::LAPACK)
        target_compile_definitions(femsolver_options INTERFACE FEM_NUMERIC_ENABLE_LAPACK=1)
        target_link_libraries(femsolver_options INTERFACE LAPACK::LAPACK)
        message(STATUS "Numeric: LAPACK backend enabled for header-only algorithms")
    else()
        message(WARNING "FEM_NUMERIC_ENABLE_LAPACK=ON but LAPACK not found; vendor paths will be disabled.")
    endif()
    if (FEM_NUMERIC_ENABLE_LAPACKE)
        find_package(LAPACKE QUIET)
        if (LAPACKE_FOUND)
            target_compile_definitions(femsolver_options INTERFACE FEM_NUMERIC_ENABLE_LAPACKE=1)
            if (TARGET LAPACKE::LAPACKE)
                target_link_libraries(femsolver_options INTERFACE LAPACKE::LAPACKE)
            else()
                find_library(LAPACKE_LIBRARY lapacke)
                if (LAPACKE_LIBRARY)
                    target_link_libraries(femsolver_options INTERFACE ${LAPACKE_LIBRARY})
                endif()
            endif()
            message(STATUS "Numeric: LAPACKE row-major wrappers enabled")
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
#  Third-party header-only deps via FetchContent
# -----------------------------------------------------------------------------
include(FetchContent)

# GoogleTest (tests)
if (BUILD_UNIT_TESTS OR BUILD_PERFORMANCE_TESTS OR
    BUILD_INTEGRATION_TESTS OR BUILD_REGRESSION_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif()

# fmt (nice printing everywhere)
FetchContent_Declare(fmtlib
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        10.2.1)
FetchContent_MakeAvailable(fmtlib)
#add_library(fmt::fmt ALIAS fmt)   # unify alias for MSVC + others

# -----------------------------------------------------------------------------
#  Sub-directories (layer order bottom → top)
# -----------------------------------------------------------------------------
# Core library (foundational infrastructure)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    add_subdirectory(src/core)
endif()

# Core tests (if testing is enabled and tests exist)
if(BUILD_UNIT_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/core/tests)
    # Tests are added by core's CMakeLists.txt
endif()

# Core benchmarks (if performance testing is enabled and benchmarks exist)
if(BUILD_PERFORMANCE_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/core/benchmarks)
    # Benchmarks are added by core's CMakeLists.txt
endif()

# -----------------------------------------------------------------------------
# Python Support (Bindings n such)
# -----------------------------------------------------------------------------

if (BUILD_PYTHON)
    add_subdirectory(src/drivers/python)  # pybind11 glue
endif()

# -----------------------------------------------------------------------------
#  Installation + packaging bits (stub; flesh out later)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
# install(TARGETS …)  etc.

# -----------------------------------------------------------------------------
#  Project summary banner
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "-------------------------------------------")
message(STATUS "  MFEMSolver configuration summary:")
message(STATUS "    C++ compiler      = ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "    Build type        = ${CMAKE_BUILD_TYPE}")
message(STATUS "    Sanitizers        = ${ENABLE_SANITIZERS}")
message(STATUS "    Tests             = ${BUILD_TESTS}")
message(STATUS "    Examples          = ${BUILD_EXAMPLES}")
message(STATUS "    Python bindings   = ${BUILD_PYTHON}")
message(STATUS "-------------------------------------------")
