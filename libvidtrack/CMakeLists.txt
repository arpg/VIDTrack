cmake_policy(SET CMP0042 OLD)

include(def_library)
include(export_package)

set(VIDTRACK_VERSION_MAJOR 1)
set(VIDTRACK_VERSION_MINOR 0)
set(VIDTRACK_VERSION ${VIDTRACK_VERSION_MAJOR}.${VIDTRACK_VERSION_MINOR})

option(BUILD_SHARED_LIBS "Build Shared Library" ON)

find_package(MINIGLOG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(Calibu 0.1 REQUIRED)
find_package(BA REQUIRED)

find_package(TBB QUIET)
if(TBB_FOUND)
  set(VIDTRACK_USE_TBB 1 CACHE INTERNAL "VIDTrack TBB Flag" FORCE)
endif()

#################################################
# Append all includes.
list(APPEND VIDTRACK_INC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_BINARY_DIR}/include/
    ${MINIGLOG_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIR}
    ${OpenCV2_INCLUDE_DIR}
    ${Calibu_INCLUDE_DIRS}
    ${BA_INCLUDE_DIRS}
   )

if(TBB_FOUND)
list(APPEND VIDTRACK_INC_DIRS ${TBB_INCLUDE_DIRS})
endif()


# Add include directories to library.
include_directories(${VIDTRACK_INC_DIRS})


#################################################
# Append all library paths.
list(APPEND VIDTRACK_LINK_DIRS
    ${Pangolin_LIBRARY_DIRS}
   )


#################################################
# Append all libraries.
list(APPEND VIDTRACK_LIBS
    ${MINIGLOG_LIBRARIES}
    ${OpenCV2_LIBRARIES}
    ${Calibu_LIBRARIES}
    ${BA_LIBRARIES}
   )

if(TBB_FOUND)
  list(APPEND VIDTRACK_LIBS ${TBB_LIBRARIES})
endif()

#################################################
# Library headers and sources.
set(VIDTRACK_HDRS
    include/vidtrack/dtrack.h
    include/vidtrack/tracker.h
   )

set(VIDTRACK_SRCS
    src/dtrack.cpp
    src/tracker.cpp
   )

######################################################
## Create configure file for inclusion in library.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/vidtrack/config.h"
)


#################################################
# Define library.
def_library(libvidtrack
  SOURCES ${VIDTRACK_HDRS} ${VIDTRACK_SRCS}
  LINK_LIBS ${VIDTRACK_LIBS}
  PACKAGE VIDTrack)


#################################################
# Export library to cmake.
export_package(VIDTrack
  TARGETS libvidtrack
  VERSION ${VIDTRACK_VERSION}
  LIBRARY libvidtrack
  LIBRARIES libvidtrack
  INCLUDE_DIRS ${VIDTRACK_INC_DIRS}
  LINK_DIRS ${VIDTRACK_LINK_DIRS}
  )

