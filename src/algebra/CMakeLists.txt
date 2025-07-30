file(GLOB_RECURSE ALG_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

if (ALG_SRC)
    add_library(algebra STATIC ${ALG_SRC})
else()
    add_library(algebra INTERFACE)
endif()

target_link_libraries(algebra
        INTERFACE femsolver_options numeric core)
target_include_directories(algebra
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (ALG_SRC)
    femsolver_enable_warnings(algebra)
endif()
