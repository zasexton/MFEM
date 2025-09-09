# Attempt to locate TBB 2020.
# If not found, optionally fetch oneTBB via FetchContent and disable its tests.

if(NOT DEFINED TBB_FETCH)
  set(TBB_FETCH ON)
endif()

find_package(TBB 2020 QUIET)

if(NOT TBB_FOUND)
  if(TBB_FETCH)
    message(STATUS "TBB not found; fetching oneTBB")
    include(FetchContent)
    # Disable oneTBB tests to speed up configuration
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(TBB
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2021.12.0
    )
    FetchContent_MakeAvailable(TBB)
  else()
    message(FATAL_ERROR "TBB not found and automatic download disabled")
  endif()
endif()

