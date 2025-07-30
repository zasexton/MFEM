file(GLOB_RECURSE UTILS_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

file(GLOB_RECURSE UTILS_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (UTILS_SRC)
    add_library(utils STATIC ${UTILS_SRC})
else()
    add_library(utils INTERFACE)
endif()

target_link_libraries(utils
        PUBLIC femsolver_options fmt::fmt)
target_include_directories(utils
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (UTILS_SRC)
    femsolver_enable_warnings(utils)
endif()
