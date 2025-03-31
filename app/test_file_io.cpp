#include "file_io.h"

#include <opencv2/opencv.hpp>

int main() {
  std::string camera_intrinsic_file = "/media/lam_data/标定数据/2pb65/新方案数据/20231129/line/camera_baselink/camera/avm_left_param.xml";
  std::string model_type; 
  cv::Mat camera_intrinsic;
  cv::Mat camera_distort;
  int img_height;
  int img_width;
  if(file_io::readCamInFromXmlFile(camera_intrinsic_file, model_type, 
                                camera_intrinsic, camera_distort,
                                img_height, img_width)) {
    std::cout << "model_type: " << model_type << std::endl
              << "camera_intrinsic: " << camera_intrinsic << std::endl
              << "camera_distort: " << camera_distort << std::endl
              << "img_height: " << img_height << std::endl
              << "img_width: " << img_width << std::endl;                              
  }

  return 0;
}