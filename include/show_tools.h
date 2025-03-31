#ifndef SHOW_TOOLS
#define SHOW_TOOLS

#include "file_io.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

namespace show_tools {
cv::Mat getConnectImg(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud,
    const int& dis_threshold, const int& camera_width,
    const int& camera_height);
cv::Mat getProjectionImg(const cv::Mat& raw_img,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_point,
                         const Eigen::Matrix4d& Tx_C_L,
                         const cv::Mat& camera_matrix,
                         cv::Mat& distortion_coeff);
void getColorCloud(const std::vector<cv::Mat>& rgb_imgs,
                   const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& lidar_clouds,
                   const std::vector<double>& cam_stamp_vec,
                   const StampedPoseVectorPtr& stamp_pose_vec,
                   const Eigen::Matrix4d& Tx_C_L,
                   const cv::Mat& camera_matrix,
                   const cv::Mat& distortion_coeff,
                   std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& color_cloud);                         
}  // namespace show_tools

#endif