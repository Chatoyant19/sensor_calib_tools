test_main.cpp,使用config_calib.yaml,目前不支持激光雷达之间的标定
test_dr_lidar_calib.cpp,使用config_dr_lidar_calib.yaml

* [x] 去ros
* [x] 更新ceres版本
* [ ] 增加多激光雷达标定功能
* [ ] 增加标定lidars-cameras标定功能
* [ ] target: sensor_calib_tools

=================================================================
ref:
1. https://github.com/AFEICHINA/extended_lidar_camera_calib
2. https://github.com/hku-mars/livox_camera_calib [livox_camera_calib]
3. https://github.com/hku-mars/mlcc [mlcc]

====================================================================
Ceres version: 2.2.0
OpenCV version: 4.6.0
Protobuf version: 3.21.12