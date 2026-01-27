cmake_minimum_required(VERSION 3.0.2)
project(calib_tools)

## Compile as C++14
# add_compile_options(-std=c++14)
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED filesystem)
find_package(PCL REQUIRED)
# set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)
message("OpenCV_VERSION: ${OpenCV_VERSION}")
find_package(Threads)
find_package(Ceres REQUIRED) # ceres version: 2.0.0
####print ceres info###
get_target_property(CERES_INCLUDE_DIR Ceres::ceres INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS ">>> Ceres include dirs: ${CERES_INCLUDE_DIR}")

get_target_property(CERES_LIBS Ceres::ceres INTERFACE_LINK_LIBRARIES)
message(STATUS ">>> Ceres libraries: ${CERES_LIBS}")
####print ceres info###

find_package(Protobuf REQUIRED)
message("Protobuf_VERSION: ${Protobuf_VERSION}")

# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTOUIC ON)
# set(CMAKE_AUTORCC ON)
# set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/rapidxml-1.13
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIRS}
)

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
)     

add_executable(test_extract_lidar_feature 
  app/test_extract_lidar_feature.cpp
  src/extract_lidar_feature.cpp)
target_link_libraries(test_extract_lidar_feature
  ${PCL_LIBRARIES} 
)

add_executable(test_extract_image_features 
  app/test_extract_image_features.cpp
  src/extract_image_feature.cpp)
target_link_libraries(test_extract_image_features
  ${OpenCV_LIBS} 
)

add_executable(test_extract_floor_plane 
  app/test_extract_floor_plane.cpp
  src/file_io.cpp
  src/sensor_extrinsic.pb.cc
  src/floor_plane_constriant.cpp)
target_link_libraries(test_extract_floor_plane
  ${PCL_LIBRARIES} 
  ${PROTOBUF_LIBRARY}
  ${OpenCV_LIBS}
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
)

add_executable(test_show_tool app/test_show_tool.cpp 
  src/show_tools.cpp
  src/file_io.cpp
  src/sensor_extrinsic.pb.cc)
target_link_libraries(test_show_tool
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${PROTOBUF_LIBRARY}
)

# add_executable(test_show_tool_3d app/test_show_tool_3d.cpp 
#   src/show_tools.cpp
#   src/file_io.cpp
#   src/sensor_extrinsic.pb.cc)
# target_link_libraries(test_show_tool_3d
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBS}
#   ${PROTOBUF_LIBRARY}
# )

add_executable(test_file_io app/test_file_io.cpp
              src/file_io 
              src/sensor_extrinsic.pb.cc)
target_link_libraries(test_file_io
  ${OpenCV_LIBS}
  ${PROTOBUF_LIBRARY}
)     
