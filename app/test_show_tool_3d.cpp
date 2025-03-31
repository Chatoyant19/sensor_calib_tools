// Copyright 2025. All Rights Reserved.
// Author: Dan Wang
/**********************************************************
 * purpose:
 *  show calibration result-pointclouds with color of images.
 *
 * pipeline:
 *    input: odometry pose file,
 *           pcd and image folder, 
 *           camera intrinsics file, 
 *           T_base-link_camera and T_base-link_lidar file 
 *    output: save pcd file, show calibration result
 *    
 * usage:
 *    ./test_show_tool_3d /home/danwa/projects/calib_tools/config/config_test_show_tool_3d.yaml
 * 
 *********************************************************/