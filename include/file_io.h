#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace file_io {
  void readExtrinsicFromPbFile(const std::string& pb_file, Eigen::Matrix4d& extrinsic);
  void readCamInFromXmlFile(const std::string& xml_file, std::string& model_type, 
                            cv::Mat& camera_intrinsic, cv::Mat& camera_distort,
                            int& img_height, int& img_width);
  void readCamExFromYmlFile(const std::string& yml_file, Eigen::Matrix4d& cam_extrinsic);   
  void readExtrinsicFromYamlFile(const std::string& file, Eigen::Matrix4d& extrinsic);          
  void writeExtrinsicToPbFile(const Eigen::Matrix4d& extrinsic, std::string& output_file);                        
} // namespace file_io