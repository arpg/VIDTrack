cmake_minimum_required(VERSION 2.8.10)
project(libGUI)

find_package(Eigen3 REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(SceneGraph 0.1 REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${SceneGraph_INCLUDE_DIRS}
  )

set(GUI_HDRS
        AnalyticsView.h
        ColorPalette.h
        GLVarHistory.h
        TimerView.h
   )

set(GUI_SRCS
        AnalyticsView.cpp
        ColorPalette.cpp
   )

add_library(libGUI ${GUI_HDRS} ${GUI_SRCS})

link_directories(
  ${Pangolin_LIBRARY_DIRS}
  )

target_link_libraries(libGUI
  ${Pangolin_LIBRARIES}
  ${SceneGraph_LIBRARIES}
  )
