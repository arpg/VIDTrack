cmake_policy(SET CMP0042 OLD)

include(def_library)
include(export_package)

set(DEVIL_VERSION_MAJOR 1)
set(DEVIL_VERSION_MINOR 0)
set(DEVIL_VERSION ${DEVIL_VERSION_MAJOR}.${DEVIL_VERSION_MINOR})

option(BUILD_SHARED_LIBS "Build Shared Library" ON)

find_package(MINIGLOG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(Calibu 0.1 REQUIRED)
find_package(BA REQUIRED)
find_package(TBB REQUIRED)


#################################################
# Append all includes.
list(APPEND DEVIL_INC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${MINIGLOG_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIR}
    ${OpenCV2_INCLUDE_DIR}
    ${Calibu_INCLUDE_DIRS}
    ${BA_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
   )

# Add include directories to library.
include_directories(${DEVIL_INC_DIRS})


#################################################
# Append all library paths.
list(APPEND DEVIL_LINK_DIRS
    ${Pangolin_LIBRARY_DIRS}
   )


#################################################
# Append all libraries.
list(APPEND DEVIL_LIBS
    ${MINIGLOG_LIBRARIES}
    ${OpenCV2_LIBRARIES}
    ${Calibu_LIBRARIES}
    ${BA_LIBRARIES}
    ${TBB_LIBRARIES}
   )


#################################################
# Library headers and sources.
set(DEVIL_HDRS
    include/devil/dtrack.h
    include/devil/tracker.h
   )

set(DEVIL_SRCS
    src/dtrack.cpp
    src/tracker.cpp
   )


#################################################
# Define library.
def_library(libdevil
  SOURCES ${DEVIL_HDRS} ${DEVIL_SRCS}
  LINK_LIBS ${DEVIL_LIBS}
  PACKAGE DevilTrack)


#################################################
# Export library to cmake.
export_package(DevilTrack
  TARGETS libdevil
  VERSION ${DEVIL_VERSION}
  LIBRARY libdevil
  LIBRARIES libdevil
  INCLUDE_DIRS ${DEVIL_INC_DIRS}
  LINK_DIRS ${DEVIL_LINK_DIRS}
  )
