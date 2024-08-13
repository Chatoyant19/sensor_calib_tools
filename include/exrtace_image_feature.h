#ifndef EXTRACT_IMAGE_FEATURE
#define EXTRACT_IMAGE_FEATURE

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

class ExtractImageFeature {
 public:
  ExtractImageFeature(double canny_threshold, int rgb_edge_minLen)
      : canny_threshold_(canny_threshold), rgb_edge_minLen_(rgb_edge_minLen) {}
  void getEdgeFeatures(const cv::Mat& raw_image,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud);

 private:
  void edgeDetector(const cv::Mat& src_img,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& edge_cloud);

 private:
  double canny_threshold_;
  int rgb_edge_minLen_;
};

#endif