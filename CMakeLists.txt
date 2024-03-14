cmake_minimum_required(VERSION 3.0.2)
project(dr_lidar_calib)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -Wall")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV)
find_package(Threads)
find_package(Ceres REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
)


find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# message("!!!!!!!!!CMAKE_CURRENT_SOURCE_DIR:", ${CMAKE_CURRENT_SOURCE_DIR})

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/dr_lidar_calib/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/dr_lidar_calib/config
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/rapidxml-1.13)

add_executable(test_main
  app/test_main.cpp
  src/multi_lidars_calib.cpp
  src/dr_lidar_calib.cpp
  src/lidar_odometry.cpp
  src/lidar_compensation.cpp
  src/common.cpp
  src/handeye.cpp
  src/pose_optimization.cpp
  src/pose_local_parameterization.cpp
  src/extract_lidar_feature.cpp
  src/extract_image_feature.cpp
  src/file_io.cpp
  src/sensor_extrinsic.pb.cc
  src/match_features.cpp
  src/show_tools.cpp
  src/floor_plane_constriant.cpp)
target_link_libraries(test_main
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${PROTOBUF_LIBRARY}
  ${catkin_LIBRARIES}
)     

add_executable(test_extract_lidar_feature 
  app/test_extract_lidar_feature.cpp
  src/extract_lidar_feature.cpp)
target_link_libraries(test_extract_lidar_feature
  ${PCL_LIBRARIES} 
  ${catkin_LIBRARIES}
)

add_executable(test_extract_image_features 
  app/test_extract_image_features.cpp
  src/extract_image_feature.cpp)
target_link_libraries(test_extract_image_features
  ${OpenCV_LIBS} 
  ${catkin_LIBRARIES}
)

add_executable(extract_floor_plane 
  app/extract_floor_plane.cpp)
target_link_libraries(extract_floor_plane
  ${PCL_LIBRARIES} ${catkin_LIBRARIES}
)

add_executable(test_dr_lidar_calib 
  app/test_dr_lidar_calib.cpp 
  src/dr_lidar_calib.cpp
  src/pose_local_parameterization.cpp
  src/extract_lidar_feature.cpp
  src/extract_image_feature.cpp
  src/file_io.cpp
  src/sensor_extrinsic.pb.cc
  src/match_features.cpp
  src/show_tools.cpp
  src/floor_plane_constriant.cpp)
target_link_libraries(test_dr_lidar_calib
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${PROTOBUF_LIBRARY}
  ${catkin_LIBRARIES}
)

# add_executable(test_show_tools app/test_show_tools.cpp src/show_tools.cpp)
# target_link_libraries(test_show_tools
#   file_io
#   ${OpenCV_LIBS}
#   ${catkin_LIBRARIES}
# )



install(TARGETS
  # multi_dr_lidar_calib
  
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  