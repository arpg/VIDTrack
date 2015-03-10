cmake_minimum_required(VERSION 2.8.10)
project(VIDTrack)

# Include local cmake modules directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

# Default flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wextra")


# Libraries.
add_subdirectory(libvidtrack)
add_subdirectory(libelas)


# Example applications.
option(BUILD_APPLICATIONS "Build Applications" ON)
if(BUILD_APPLICATIONS)
    add_subdirectory(applications)
endif()
