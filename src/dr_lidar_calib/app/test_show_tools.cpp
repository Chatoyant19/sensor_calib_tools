#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "show_tools.h"
#include "file_io.h"

int main(int argc, char** argv) {
  std::string input_img_path = "/home/wd/projects/mi-extrinsic-calib-1.0/bin/data/Cam0/image0001.ppm";
  std::string input_pcd_path = "/home/wd/projects/mi-extrinsic-calib-1.0/bin/data/scans/Scan_for_MI_0001.pcd";
  // std::string lidar_extrinsics_file = "/home/wd/datasets/beijing/RESULT/roof_transform.pb.txt";
  // std::string camera_extrinsics_file = "/home/wd/datasets/beijing/RESULT/avm_front_transform.pb.txt";
  // std::string camera_intrinsics_file = "/media/lam_data/标定数据/上海小车/3号/20231124/c样/直线往返/camera_baselink/camera/avm_left_param.xml";

  cv::Mat raw_img = cv::imread(input_img_path);
  pcl::PointCloud<pcl::PointXYZI> raw_pcd;
  pcl::io::loadPCDFile(input_pcd_path, raw_pcd);

  // Eigen::Matrix4d Tx_dr_L;
  // file_io::readExtrinsicFromPbFile(lidar_extrinsics_file, Tx_dr_L);
  // Eigen::Matrix4d Tx_dr_C;
  // file_io::readExtrinsicFromPbFile(camera_extrinsics_file, Tx_dr_C);
  // Eigen::Matrix4d Tx_C_L = Tx_dr_C.inverse() * Tx_dr_L;
  // Eigen::Matrix4d Tx_L_C = Eigen::Matrix4d::Identity();
  // Tx_L_C.block<3, 3>(0, 0) = Eigen::Quaterniond(0.5856330374332144,
  //     -0.8049427527785473,
  //     -0.05777320810071154,
  //     -0.07591684030433596).toRotationMatrix();
  // Tx_L_C.block<3, 1>(0, 3) = Eigen::Vector3d(-0.5400656561363888,
  //     0.44777818471004943,
  //     -0.331944182608915567);    
  // Eigen::Matrix4d Tx_C_L = Tx_L_C.inverse();
  Eigen::Matrix4d T_base_C = Eigen::Matrix4d::Identity();
  T_base_C << -0.00352104, 0.00242484, 0.999991, 0.042152,
              0.000485243, 0.999997,   -0.00242314, -0.001818,
              -0.999994, 0.000476706, -0.00352221, -0.000285,
              0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix4d T_base_L = Eigen::Matrix4d::Identity();
  T_base_L << -0.00455865,    0.999981,  0.00419064, 0.303927,
              -0.999988, -0.00456568,   0.0016699, -0.00234,
              0.001689, -0.00418298,     0.99999, -0.442928, 
              0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix4d Tx_C_L = T_base_C.inverse() * T_base_L;
  std::cout << "Tx_C_L: " << Tx_C_L << std::endl;

  std::string camera_type;
  cv::Mat camera_intrinsic;
  cv::Mat camera_distort;
  int img_height, img_width;
  // file_io::readCamInFromXmlFile(camera_intrinsics_file, camera_type,  camera_intrinsic, camera_distort, img_height, img_width);
  camera_intrinsic = (cv::Mat_<double>(3, 3) << 408.397136, 0.0, 806.586960, 0.0, 408.397136 * 0.5, 315.535008, 0.0, 0.0, 1.0);
  camera_distort = (cv::Mat_<double>(5, 1) << 0., 0., 0., 0., 0.);

  cv::Mat res_img = show_tools::getProjectionImg(raw_img, raw_pcd.makeShared(), Tx_C_L, camera_intrinsic, camera_distort);

  std::string save_path = "/home/wd/datasets/1.png";
  cv::imwrite(save_path, res_img);

  return 0;
}