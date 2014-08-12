find_package(Pangolin 0.1 REQUIRED)
find_package(SceneGraph 0.1 REQUIRED)

include_directories(
  ${SceneGraph_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
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

add_library(AuxGUI ${GUI_HDRS} ${GUI_SRCS})

link_directories(
  ${Pangolin_LIBRARY_DIRS}
  )

target_link_libraries(AuxGUI
  ${SceneGraph_LIBRARIES}
  ${Pangolin_LIBRARIES}
  )
