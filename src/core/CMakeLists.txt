file(GLOB_RECURSE CORE_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

file(GLOB_RECURSE CORE_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (CORE_SRC)
    add_library(core STATIC ${CORE_SRC})
else()
    add_library(core INTERFACE)
endif()

target_link_libraries(core
        INTERFACE femsolver_options numeric continuum)
target_include_directories(core
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (CORE_SRC)
    femsolver_enable_warnings(core)
endif()
