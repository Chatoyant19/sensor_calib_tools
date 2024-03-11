#pragma once 

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace show_tools {
cv::Mat getConnectImg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud,
                      const int& dis_threshold, const int& camera_width, const int& camera_height);
cv::Mat getProjectionImg(const cv::Mat& raw_img, const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_point,
                         const Eigen::Matrix4d& Tx_C_L,
                         const cv::Mat& camera_matrix, cv::Mat& distortion_coeff);                                           
} // namespace show_tools