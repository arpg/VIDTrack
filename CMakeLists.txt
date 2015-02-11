cmake_minimum_required(VERSION 2.8.10)
project(DEVIL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# Library files.
add_subdirectory(libdevil)


# Example applications.
option(BUILD_APPLICATIONS "Build Applications" ON)
if(BUILD_APPLICATIONS)
    add_subdirectory(applications)
endif()
