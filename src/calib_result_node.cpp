#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
//pcl lib
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

std::string input_pcd_path = "/home/wd/datasets/new_v/pcd/0_visual.pcd";
Eigen::Matrix4d before_T_dr_L_;
Eigen::Matrix4d refine_T_dr_L_;

int main(int argc, char** argv) {
  ros::init(argc, argv, "calib_result_node");
  ros::NodeHandle nh("calib_result_node");
  ros::NodeHandle nh_private("~");

  
  // Eigen::Matrix3d before_R;
  // before_R << 
  // Eigen::Vector3d before_t = Eigen::Vector3d();
  // before_T_dr_L_.block<3, 3>(0, 0) = before_q.toRotationMatrix();
  // before_T_dr_L_.block<3, 1>(0, 3) = before_t; 
  /*before_T_dr_L_ << 0.99948,     0.031479 ,  0.00697896,      1.00836,
  -0.0314761,     0.999504, -0.000520085,    0.0144744,
 -0.00699187,  0.000300144,     0.999976,       1.8938,
           0,            0,            0,            1;

  Eigen::Quaterniond refine_q = Eigen::Quaterniond(0.99987399153739931, -0.0016012319863261662, 0.00115935023477785, -0.015750968548881264);
  Eigen::Vector3d refine_t = Eigen::Vector3d(0.99961697382327819, 0.021454352522907442, 1.8983644295219595);
  refine_T_dr_L_.block<3, 3>(0, 0) = refine_q.toRotationMatrix();
  refine_T_dr_L_.block<3, 1>(0, 3) = refine_t; 

  ros::Publisher pub_transform_pcd_1_ = nh.advertise<sensor_msgs::PointCloud2>("/pcd_1", 100);
  ros::Publisher pub_transform_pcd_2_ = nh.advertise<sensor_msgs::PointCloud2>("/pcd_2", 100);

  pcl::PointCloud<pcl::PointXYZI> raw_pcd;
  pcl::io::loadPCDFile(input_pcd_path, raw_pcd);

  pcl::PointCloud<pcl::PointXYZI> transform_pcd_before;
  pcl::transformPointCloud(raw_pcd, transform_pcd_before, before_T_dr_L_);

  pcl::PointCloud<pcl::PointXYZI> transform_pcd_refine;
  pcl::transformPointCloud(raw_pcd, transform_pcd_refine, refine_T_dr_L_);
  std::cout << "raw_pcd size: " << raw_pcd.size() << std::endl;

  for(int i = 0; i < 10; ++i) {
  sensor_msgs::PointCloud2 transform_pcd_before_ros, transform_pcd_refine_ros;
  pcl::toROSMsg(transform_pcd_before, transform_pcd_before_ros);
  pcl::toROSMsg(transform_pcd_refine, transform_pcd_refine_ros);
  std::cout << "transform_pcd_before size: " << transform_pcd_before.size() << std::endl
            << "transform_pcd_fine size: " << transform_pcd_refine.size() << std::endl;

  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = "base_link";
  transform_pcd_before_ros.header = header;
  transform_pcd_refine_ros.header = header;
  pub_transform_pcd_1_.publish(transform_pcd_before_ros);
  pub_transform_pcd_2_.publish(transform_pcd_refine_ros);
  }*/

  return 0; 
}