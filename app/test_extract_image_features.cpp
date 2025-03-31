#include "exrtace_image_feature.h"

double canny_threshold_ = 10;
int rgb_edge_minLen_ = 50;
std::string input_raw_img_path = "/home/wd/datasets/018/img/avm_front/0.bmp";

int main(int argc, char** argv) {
  std::unique_ptr<ExtractImageFeature> extract_image_feature = 
    std::make_unique<ExtractImageFeature>(canny_threshold_, rgb_edge_minLen_);

  pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_clouds = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>); 
  cv::Mat raw_img = cv::imread(input_raw_img_path);          
  extract_image_feature->getEdgeFeatures(raw_img, rgb_egde_clouds);
  std::cout << "rgb_egde_clouds size: " << rgb_egde_clouds->size() << std::endl;         
}