cmake_minimum_required(VERSION 2.8.10)
project(DEVIL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(HAL 0.1 REQUIRED)
find_package(PbMsgs 0.1 REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(SceneGraph 0.1 REQUIRED)
find_package(Calibu 0.1 REQUIRED)
find_package(BA REQUIRED)
find_package(CERES REQUIRED)
find_package(TBB QUIET)
find_package(Protobuf QUIET)
find_package(MINIGLOG QUIET)

if(TBB_FOUND)
    add_definitions(-DHAVE_TBB)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    add_definitions(-DHAVE_CLANG)
endif()

# Use OpenMP or not
option(USE_OPENMP "USE_OPENMP" OFF)

if(USE_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fopenmp -Werror")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Werror")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIR})
include_directories(${OpenCV2_INCLUDE_DIR})
include_directories(${HAL_INCLUDE_DIRS})
include_directories(${PbMsgs_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${SceneGraph_INCLUDE_DIRS})
include_directories(${Calibu_INCLUDE_DIRS})
include_directories(${BA_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIR})
include_directories(${TBB_INCLUDE_DIRS})

link_directories(${Pangolin_LIBRARY_DIRS})

add_subdirectory(AuxGUI)

list(APPEND HDRS dtrack.h)
list(APPEND SRCS main.cpp)

add_executable(devil ${HDRS} ${SRCS})

add_dependencies(devil AuxGUI)

target_link_libraries(devil AuxGUI)
target_link_libraries(devil ${OpenCV2_LIBRARIES})
target_link_libraries(devil ${HAL_LIBRARIES})
target_link_libraries(devil ${PbMsgs_LIBRARIES})
target_link_libraries(devil ${Pangolin_LIBRARIES})
target_link_libraries(devil ${SceneGraph_LIBRARIES})
target_link_libraries(devil ${Calibu_LIBRARIES})
target_link_libraries(devil ${BA_LIBRARIES})
target_link_libraries(devil ${CERES_LIBRARIES})
target_link_libraries(devil ${PROTOBUF_LIBRARIES})
if(TBB_FOUND)
  target_link_libraries(devil ${TBB_LIBRARIES})
endif()
