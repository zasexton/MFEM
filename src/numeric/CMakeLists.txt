# Fundamental math utilities – header-only for now
file(GLOB_RECURSE NUM_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(numeric INTERFACE)

target_link_libraries(numeric
        INTERFACE femsolver_options fmt::fmt)
target_include_directories(numeric
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
