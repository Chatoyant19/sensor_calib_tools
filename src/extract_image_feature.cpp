#include "exrtace_image_feature.h"

void ExtractImageFeature::getEdgeFeatures(
    const cv::Mat& raw_image,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud) {
  cv::Mat grey_img;
  cv::cvtColor(raw_image, grey_img, cv::COLOR_BGR2GRAY);

  edgeDetector(grey_img, rgb_edge_cloud);
}

void ExtractImageFeature::edgeDetector(
    const cv::Mat& src_img, pcl::PointCloud<pcl::PointXYZ>::Ptr& edge_cloud) {
  int gaussian_size = 5;
  cv::GaussianBlur(src_img, src_img, cv::Size(gaussian_size, gaussian_size), 0,
                   0);

  int width = src_img.cols;
  int height = src_img.rows;
  cv::Mat canny_result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Canny(src_img, canny_result, canny_threshold_, canny_threshold_ * 3, 3,
            true);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

  cv::Mat edge_img = cv::Mat::zeros(height, width, CV_8UC1);
  for (size_t i = 0; i < contours.size(); i++) {
    if (contours[i].size() < rgb_edge_minLen_) continue;
    for (size_t j = 0; j < contours[i].size(); j++) {
      pcl::PointXYZ p;
      p.x = contours[i][j].x;
      p.y = -contours[i][j].y;
      p.z = 0;
      edge_img.at<uchar>(-p.y, p.x) = 255;
    }
  }

  for (int x = 0; x < edge_img.cols; x++) {
    for (int y = 0; y < edge_img.rows; y++) {
      if (edge_img.at<uchar>(y, x) == 255) {
        pcl::PointXYZ p;
        p.x = x;
        p.y = -y;
        p.z = 0;
        edge_cloud->points.push_back(p);
      }
    }
  }
}