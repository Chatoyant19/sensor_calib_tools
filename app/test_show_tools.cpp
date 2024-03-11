#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "show_tools.h"
#include "file_io.h"

int main(int argc, char** argv) {
  std::string input_img_path = "/media/lam_data/标定数据/上海小车/3号/20231124/c样/直线往返/lidar_baselink/pre_data/img/avm_left/0.bmp";
  std::string input_pcd_path = "/home/wd/datasets/3/0_visual.pcd";
  // std::string lidar_extrinsics_file = "/home/wd/datasets/beijing/RESULT/roof_transform.pb.txt";
  // std::string camera_extrinsics_file = "/home/wd/datasets/beijing/RESULT/avm_front_transform.pb.txt";
  std::string camera_intrinsics_file = "/media/lam_data/标定数据/上海小车/3号/20231124/c样/直线往返/camera_baselink/camera/avm_left_param.xml";

  cv::Mat raw_img = cv::imread(input_img_path);
  pcl::PointCloud<pcl::PointXYZI> raw_pcd;
  pcl::io::loadPCDFile(input_pcd_path, raw_pcd);

  // Eigen::Matrix4d Tx_dr_L;
  // file_io::readExtrinsicFromPbFile(lidar_extrinsics_file, Tx_dr_L);
  // Eigen::Matrix4d Tx_dr_C;
  // file_io::readExtrinsicFromPbFile(camera_extrinsics_file, Tx_dr_C);
  // Eigen::Matrix4d Tx_C_L = Tx_dr_C.inverse() * Tx_dr_L;
  Eigen::Matrix4d Tx_L_C = Eigen::Matrix4d::Identity();
  Tx_L_C.block<3, 3>(0, 0) = Eigen::Quaterniond(0.5856330374332144,
      -0.8049427527785473,
      -0.05777320810071154,
      -0.07591684030433596).toRotationMatrix();
  Tx_L_C.block<3, 1>(0, 3) = Eigen::Vector3d(-0.5400656561363888,
      0.44777818471004943,
      -0.331944182608915567);    
  Eigen::Matrix4d Tx_C_L = Tx_L_C.inverse();

  std::string camera_type;
  cv::Mat camera_intrinsic;
  cv::Mat camera_distort;
  int img_height, img_width;
  file_io::readCamInFromXmlFile(camera_intrinsics_file, camera_type,  camera_intrinsic, camera_distort, img_height, img_width);

  cv::Mat res_img = show_tools::getProjectionImg(raw_img, raw_pcd.makeShared(), Tx_C_L, camera_intrinsic, camera_distort);

  std::string save_path = "/home/wd/datasets/2.png";
  cv::imwrite(save_path, res_img);
}