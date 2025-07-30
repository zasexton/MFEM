file(GLOB_RECURSE CONTINUUM_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

if (CONTINUUM_SRC)
    add_library(continuum STATIC ${CONTINUUM_SRC})
else()
    add_library(continuum INTERFACE)
endif()

target_link_libraries(continuum
        INTERFACE numeric femsolver_options)
target_include_directories(continuum
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (CONTINUUM_SRC)  # INTERFACE targets can’t take compile flags
    femsolver_enable_warnings(continuum)
endif()
