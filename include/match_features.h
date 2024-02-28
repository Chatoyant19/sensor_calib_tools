#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

#include "common.h"

class MatchFeatures {
 public:
  MatchFeatures(bool show_residual) : show_residual_(show_residual) {}
  void buildVPnp(const cv::Mat& camera_matrix, const int& camera_width, const int& camera_height,
                 const pcl::PointCloud<pcl::PointXYZ>::Ptr& cam_edge_cloud_2d,
                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
                 const Eigen::Matrix4d& Tx_C_L,
                 const int& dis_threshold,
                 std::vector<VPnPData>& pnp_list,
                 cv::Mat& residual_img);
 private:
  void filterOutViewPcd(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
                        const Eigen::Matrix4d& Tx_C_L,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr& filtered_cloud_3d);
  void calcDirection(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& points,
                     Eigen::Vector2d& direction);                        
 private: 
  bool show_residual_;
  float direction_theta_min_ = cos(DEG2RAD(30.0));
  float direction_theta_max_ = cos(DEG2RAD(150.0));
};