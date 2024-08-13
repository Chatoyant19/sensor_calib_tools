//
// Created by user on 24-1-3.
//
#include "opencv2/opencv.hpp"
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "string"

int main(int argc, char** argv) {
  std::string filename = argv[1];
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::io::loadPCDFile(filename, cloud);

  constexpr float resolution = 0.02;
  Eigen::Vector2f mx(100, -100), my(100, -100);
  for (const auto& p : cloud.points) {
    if (mx.x() > p.x) mx.x() = p.x;
    if (mx.y() < p.x) mx.y() = p.x;
    if (my.x() > p.y) my.x() = p.y;
    if (my.y() < p.y) my.y() = p.y;
  }
  float xl = mx.y() - mx.x();
  float yl = my.y() - my.x();
  int row = int(xl / resolution) + 1;
  int col = int(yl / resolution) + 1;
  cv::Mat intensity_matf(row, col, CV_32F, cv::Scalar(0));

  for (const auto& p : cloud.points) {
    int i = (p.x - mx(0)) / resolution;
    int j = (p.y - my(0)) / resolution;
    intensity_matf.at<float>(i, j) = p.intensity;
  }
  intensity_matf = (intensity_matf > 18) * 255.f;

  cv::Mat im;

  intensity_matf.convertTo(im, CV_8U);
  cv::imwrite("/home/user/im.png", im);

  cv::blur(im, im, {3, 3});

  int canny_threshold = 10;
  cv::Mat canny_result = cv::Mat::zeros(row, col, CV_8UC1);
  cv::Canny(im, canny_result, canny_threshold, canny_threshold * 3, 3, true);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  cv::Mat edge_img = cv::Mat::zeros(row, col, CV_8UC1);

  auto edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < contours.size(); i++) {
    for (size_t j = 0; j < contours[i].size(); j++) {
      pcl::PointXYZ p;
      p.x = contours[i][j].x;
      p.y = -contours[i][j].y;
      p.z = 0;
      edge_img.at<uchar>(-p.y, p.x) = 255;
    }
  }

  cv::imwrite("/home/user/canny_result.png", canny_result);
  cv::imwrite("/home/user/edge_img.png", edge_img);

  return 0;
}
