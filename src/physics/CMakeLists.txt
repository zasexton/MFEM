file(GLOB_RECURSE PHYSICS_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

file(GLOB_RECURSE PHYSICS_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (PHYSICS_SRC)
    add_library(physics STATIC ${PHYSICS_SRC})
else()
    add_library(physics INTERFACE)
endif()

target_link_libraries(physics
        INTERFACE femsolver_options numeric continuum core algebra parallel)
target_include_directories(physics
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (PHYSICS_SRC)
    femsolver_enable_warnings(physics)
endif()
