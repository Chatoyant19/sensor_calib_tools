#pragma once 

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace show_tools {
cv::Mat getConnectImg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud,
                      const int& dis_threshold, const int& camera_width, const int& camera_height);
} // namespace show_tools