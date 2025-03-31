// Copyright 2025. All Rights Reserved.
// Author: Dan Wang
/**********************************************************
 * purpose:
 *  show calibration result-project point cloud to image.
 *
 * pipeline:
 *    input: pcd and image file, camera intrinsics file, T_base-link_camera and T_base-link_lidar file 
 *    output: save jpg file, show calibration result
 *    
 * usage:
 *    ./test_show_tool /home/danwa/projects/calib_tools/config/config_show_tools.yaml
 * 
 *********************************************************/
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "show_tools.h"
#include "file_io.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: test_show_tools path_to_config_show_tool_file" << std::endl;
    return -1;
  }

  const std::string config_setting_path = std::string(argv[1]);
  cv::FileStorage fSettings(config_setting_path, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_setting_path
              << std::endl;
    return -1;
  }

  std::string input_img_path, input_pcd_path;
  std::string camera_intrinsics_file, camera_extrinsics_file;
  std::string lidar_extrinsics_file;
  std::string save_img_path;
  fSettings["img_path"] >> input_img_path;
  fSettings["pcd_path"] >> input_pcd_path;
  fSettings["camera_intrinsics_file"] >> camera_intrinsics_file;
  fSettings["camera_extrinsics_file"] >> camera_extrinsics_file;
  fSettings["lidar_extrinsics_file"] >> lidar_extrinsics_file;
  fSettings["save_img_path"] >> save_img_path;
  std::cout << "img_path: " << input_img_path << std::endl;
  std::cout << "pcd_path: " << input_pcd_path << std::endl;
  std::cout << "camera_intrinsics_file: " << camera_intrinsics_file << std::endl;
  std::cout << "camera_extrinsics_file: " << camera_extrinsics_file << std::endl;
  std::cout << "lidar_extrinsics_file: " << lidar_extrinsics_file << std::endl;
  std::cout << "save_img_path: " << save_img_path << std::endl;

  cv::Mat raw_img = cv::imread(input_img_path);
  pcl::PointCloud<pcl::PointXYZI> raw_pcd;
  pcl::io::loadPCDFile(input_pcd_path, raw_pcd);

  Eigen::Matrix4d Tx_dr_L;
  file_io::readExtrinsicFromPbFile(lidar_extrinsics_file, Tx_dr_L);
  Eigen::Matrix4d Tx_dr_C;
  file_io::readExtrinsicFromPbFile(camera_extrinsics_file, Tx_dr_C);
  Eigen::Matrix4d Tx_C_L = Tx_dr_C.inverse() * Tx_dr_L;
  std::cout << "Tx_C_L: " << Tx_C_L << std::endl;

  std::string camera_type;
  cv::Mat camera_intrinsic;
  cv::Mat camera_distort;
  int img_height, img_width;
  file_io::readCamInFromXmlFile(camera_intrinsics_file, camera_type,  camera_intrinsic, camera_distort, img_height, img_width);

  cv::Mat res_img = show_tools::getProjectionImg(raw_img, raw_pcd.makeShared(), Tx_C_L, camera_intrinsic, camera_distort);

  cv::imwrite(save_img_path, res_img);

  return 0;
}