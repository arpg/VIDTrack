
include(def_library)
include(export_package)

set(VIDTRACK_VERSION_MAJOR 1)
set(VIDTRACK_VERSION_MINOR 0)
set(VIDTRACK_VERSION ${VIDTRACK_VERSION_MAJOR}.${VIDTRACK_VERSION_MINOR})

option(BUILD_SHARED_LIBS "Build Shared Library" ON)

find_package(MINIGLOG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Calibu 0.1 REQUIRED)
find_package(BA REQUIRED)

find_package(TBB QUIET)
if(TBB_FOUND)
  set(VIDTRACK_USE_TBB 1 CACHE INTERNAL "VIDTrack TBB Flag" FORCE)
endif()

find_package(CUDA QUIET)
if(CUDA_FOUND)
  option(ENABLE_CUDA "Enable CUDA for VIDTrack" OFF)
  if(ENABLE_CUDA)
    set(VIDTRACK_USE_CUDA 1 CACHE INTERNAL "VIDTrack CUDA Flag" FORCE)
    if(TBB_FOUND)
      message(WARNING "Since CUDA is enabled, TBB has been turned off.")
      set(VIDTRACK_USE_TBB 0 CACHE INTERNAL "VIDTrack TBB Flag" FORCE)
    endif()
  else()
    set(VIDTRACK_USE_CUDA 0 CACHE INTERNAL "VIDTrack CUDA Flag" FORCE)
    if(TBB_FOUND)
      message(WARNING "Since CUDA is disabled, TBB has been turned on.")
      set(VIDTRACK_USE_TBB 1 CACHE INTERNAL "VIDTrack TBB Flag" FORCE)
    endif()
  endif()
endif()


#################################################
# Append all includes.
list(APPEND VIDTRACK_INC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_BINARY_DIR}/include/
    ${MINIGLOG_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${Calibu_INCLUDE_DIRS}
    ${BA_INCLUDE_DIRS}
   )

if(VIDTRACK_USE_TBB)
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
    ${OpenCV_LIBRARIES}
    ${Calibu_LIBRARIES}
    ${BA_LIBRARIES}
   )

if(VIDTRACK_USE_TBB)
  list(APPEND VIDTRACK_LIBS ${TBB_LIBRARIES})
endif()

if(VIDTRACK_USE_CUDA)
  list(APPEND VIDTRACK_LIBS dtrack_cuda)
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

if(VIDTRACK_USE_CUDA)
  set(CUDA_FILES include/vidtrack/dtrack.cuh src/dtrack.cu)

  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#  set(CUDA_NVCC_FLAGS "-use_fast_math -gencode arch=compute_30,code=sm_30" )
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30")
  cuda_add_library(dtrack_cuda ${CUDA_FILES})

  install(TARGETS dtrack_cuda
    EXPORT VIDTrack
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
endif()

def_library(vidtrack
  SOURCES ${VIDTRACK_HDRS} ${VIDTRACK_SRCS}
  LINK_LIBS ${VIDTRACK_LIBS}
  PACKAGE VIDTrack)


#################################################
# Export library to cmake.
export_package(VIDTrack
  TARGETS vidtrack
  VERSION ${VIDTRACK_VERSION}
  LIBRARY vidtrack
  LIBRARIES vidtrack
  INCLUDE_DIRS ${VIDTRACK_INC_DIRS}
  LINK_DIRS ${VIDTRACK_LINK_DIRS}
  )

