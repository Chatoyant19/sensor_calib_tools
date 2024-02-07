#include <pcl/search/kdtree.h>

#include "show_tools.h"

namespace show_tools {
cv::Mat getConnectImg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud,
                      const int& dis_threshold, const int& camera_width, const int& camera_height) {
  cv::Mat connect_img = cv::Mat(camera_height, camera_width, CV_8UC3, cv::Scalar::all(255));

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
  kdtree->setInputCloud(rgb_edge_cloud);   
  for (auto p : *depth_edge_cloud) {
    cv::Point2d p2(p.x, -p.y);
  }

  int line_count = 0;
  // 指定近邻个数
  int K = 1;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = depth_edge_cloud->points[i];
    if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - rgb_edge_cloud->points[pointIdxNKNSearch[j]].x, 2) +
            pow(searchPoint.y - rgb_edge_cloud->points[pointIdxNKNSearch[j]].y, 2));
        if (distance < dis_threshold) {
          cv::Scalar color = cv::Scalar(0, 255, 0);
          line_count++;
          if ((line_count % 3) == 0) {
            cv::line(connect_img, cv::Point(depth_edge_cloud->points[i].x,
                                            -depth_edge_cloud->points[i].y),
                     cv::Point(rgb_edge_cloud->points[pointIdxNKNSearch[j]].x,
                               -rgb_edge_cloud->points[pointIdxNKNSearch[j]].y),
                     color, 1);
          }
        }
      }
    }
  }

   for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[0] = 255;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[2] = 0;
  }
  for (size_t i = 0; i < depth_edge_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-depth_edge_cloud->points[i].y,
                              depth_edge_cloud->points[i].x)[0] = 0;
    connect_img.at<cv::Vec3b>(-depth_edge_cloud->points[i].y,
                              depth_edge_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-depth_edge_cloud->points[i].y,
                              depth_edge_cloud->points[i].x)[2] = 255;
  }
  int expand_size = 2;
  cv::Mat expand_edge_img;
  expand_edge_img = connect_img.clone();
  for (int x = expand_size; x < connect_img.cols - expand_size; x++) {
    for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
      if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
          }
        }
      } else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
          }
        }
      }
    }
  }
  return connect_img;
} 
} // namespace show_tools