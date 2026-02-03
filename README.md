# Sensor_calib_tools
This module is a robust, high accuracy extrinsic calibration tool between high resolution LiDARs (e.g. Livox), cameras and base_link in targetless environment. Our algorithm can run in both indoor and outdoor scenes, and only requires edge information in the scene. If the scene is suitable, we can achieve pixel-level accuracy similar to or even beyond the target based method.

## Info
- test_main.cpp,使用config_calib.yaml,目前不支持激光雷达之间的标定
- test_dr_lidar_calib.cpp,使用config_dr_lidar_calib.yaml

## TODO
* [ ] 增加多激光雷达标定功能
* [ ] 增加标定lidars-cameras标定功能
* [ ] target: sensor_calib_tools

## Ref
- https://github.com/AFEICHINA/extended_lidar_camera_calib
- https://github.com/hku-mars/livox_camera_calib [livox_camera_calib]
- https://github.com/hku-mars/mlcc [mlcc]

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04 or 20.04. [ROS Installation](http://wiki.ros.org/ROS/Installation) and its additional ROS pacakge:

```
    sudo apt-get install ros-XXX-cv-bridge ros-xxx-pcl-conversions
```

### 1.2 **Eigen**
Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page)

### 1.3 **Ceres Solver**
Ceres Version: 1.14.0 or 2.0.0

### 1.4 **PCL**
PCL Version: 1.10.0

### 1.5 **OpenCV**
OpenCV Version: 4.2.0

### 1.6 **Boost**
Boost Version: 1.71.0

### 1.6 **Protobuf**
Protobuf Version: 3.6.1

## 2. Build
Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/hku-mars/livox_camera_calib.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Run our example
1. calib base_link and lidar
```
    rosrun dr_lidar_calib test_main ../config/config_calib.yaml
```
2. 
