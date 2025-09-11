###############################################################################
# FetchTBB.cmake
#
# Fetches and configures Intel Threading Building Blocks (oneTBB) for
# task-based parallelism and concurrent data structures.
#
# Usage:
#   option(TBB_FETCH "Automatically download oneTBB if missing" ON)
#   include(cmake/FetchTBB.cmake)
#
# Provides:
#   • Target: TBB::tbb
#   • Optional components: tbbmalloc, tbbmalloc_proxy
###############################################################################

if(DEFINED _FETCH_TBB_INCLUDED)
  return()
endif()
set(_FETCH_TBB_INCLUDED TRUE)

option(TBB_FETCH "Automatically download oneTBB if missing" ON)
option(TBB_USE_MALLOC "Use TBB scalable memory allocator" OFF)

# Try to find system TBB first
find_package(TBB QUIET)

if(TBB_FOUND)
  message(STATUS "Found system TBB version ${TBB_VERSION}")

  # Create alias for consistency
  if(NOT TARGET TBB::tbb AND TARGET TBB::tbb)
    add_library(TBB::tbb ALIAS TBB::tbb)
  endif()

elseif(TBB_FETCH)
  message(STATUS "Fetching Intel oneTBB...")

  include(FetchContent)

  # oneTBB version - using a stable release
  set(TBB_VERSION "2021.11.0")

  FetchContent_Declare(
          oneTBB
          GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
          GIT_TAG        v${TBB_VERSION}
          GIT_SHALLOW    TRUE
  )

  # TBB-specific options
  set(TBB_TEST OFF CACHE BOOL "" FORCE)
  set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(TBB_STRICT OFF CACHE BOOL "" FORCE)
  set(TBB_WINDOWS_DRIVER OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

  # Disable TBBBind by default (requires hwloc)
  set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(oneTBB)

  message(STATUS "Intel oneTBB ${TBB_VERSION} configured")

  # Export TBB for the rest of the project
  if(TARGET tbb)
    # Create the expected target name
    add_library(TBB::tbb ALIAS tbb)

    if(TBB_USE_MALLOC AND TARGET tbbmalloc)
      add_library(TBB::tbbmalloc ALIAS tbbmalloc)
      add_library(TBB::tbbmalloc_proxy ALIAS tbbmalloc_proxy)
    endif()
  endif()

else()
  message(STATUS "TBB not found and TBB_FETCH=OFF")
endif()

# -----------------------------------------------------------------------------
# Convenience function to add TBB to a target
# -----------------------------------------------------------------------------
function(add_tbb_target target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "add_tbb_target: ${target} is not a valid target")
  endif()

  if(TARGET TBB::tbb)
    target_link_libraries(${target} PUBLIC TBB::tbb)
    target_compile_definitions(${target} PUBLIC ENABLE_TBB)

    if(TBB_USE_MALLOC AND TARGET TBB::tbbmalloc)
      target_link_libraries(${target} PUBLIC TBB::tbbmalloc)
      target_compile_definitions(${target} PUBLIC TBB_USE_SCALABLE_ALLOCATOR)
    endif()

    message(STATUS "Added TBB support to target: ${target}")
  else()
    message(WARNING "TBB not available for target: ${target}")
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Helper macro for TBB parallel patterns
# -----------------------------------------------------------------------------
macro(enable_tbb_patterns)
  if(TARGET TBB::tbb)
    add_compile_definitions(
            USE_TBB_PARALLEL_FOR
            USE_TBB_PARALLEL_REDUCE
            USE_TBB_CONCURRENT_CONTAINERS
    )
  endif()
endmacro()

# -----------------------------------------------------------------------------
# Export configuration status
# -----------------------------------------------------------------------------
if(TARGET TBB::tbb)
  set(TBB_AVAILABLE TRUE CACHE INTERNAL "TBB is available")
  message(STATUS "TBB configuration complete: TBB::tbb available")
else()
  set(TBB_AVAILABLE FALSE CACHE INTERNAL "TBB is not available")
endif()