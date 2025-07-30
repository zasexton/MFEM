# ──────────────────────────────────────────────────────────────
#  I/O layer – readers / writers / parsers
#  Links: core (mesh + fields), numeric (math), parallel (MPI HDF5 writers)
# ──────────────────────────────────────────────────────────────

file(GLOB_RECURSE IO_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

file(GLOB_RECURSE IO_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# If sources already exist build a STATIC lib, otherwise keep header-only.
if (IO_SRC)
    add_library(io STATIC ${IO_SRC})
else()
    add_library(io INTERFACE)
endif()

# ── Public dependencies and include path
target_link_libraries(io
        PUBLIC
        femsolver_options      # compile flags, include path to <src/>
        numeric                # math types
        core                   # mesh / field data structures
        parallel               # ghost exchange helpers & MPI (optional)
        fmt::fmt)              # lightweight printing

target_include_directories(io
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ── Per-target warnings (only for real sources)
if (IO_SRC)
    femsolver_enable_warnings(io)
endif()

# ──────────────────────────────────────────────────────────────
#  Optional extras (commented out until you add code that needs them)
# ──────────────────────────────────────────────────────────────
#
# option(ENABLE_HDF5 "Build XDMF/HDF5 writers" ON)
# if (ENABLE_HDF5)
#     find_package(HDF5 COMPONENTS C HL REQUIRED)
#     target_link_libraries(io PUBLIC HDF5::HDF5)
#     target_compile_definitions(io PUBLIC FEMSOLVER_USE_HDF5)
# endif()
#
# option(ENABLE_ZLIB "Enable gzip/zip compression for I/O" OFF)
# if (ENABLE_ZLIB)
#     find_package(ZLIB REQUIRED)
#     target_link_libraries(io PUBLIC ZLIB::ZLIB)
# endif()
