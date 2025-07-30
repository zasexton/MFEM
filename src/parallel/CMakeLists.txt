
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(EnableMPI)        # defines  mpi::mpi   and   add_mpi_target()

# ---------------------------------------------------------------------------
# Build the 'parallel' target (INTERFACE until you add .cpp files)
# ---------------------------------------------------------------------------

file(GLOB_RECURSE PAR_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

file(GLOB_RECURSE PAR_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (PAR_SRC)
    add_library(parallel STATIC ${PAR_SRC})
else()
    add_library(parallel INTERFACE)
endif()

target_link_libraries(parallel
        INTERFACE femsolver_options numeric core algebra mpi::bundled)

target_include_directories(parallel
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (PAR_SRC)
    femsolver_enable_warnings(parallel)
endif()
