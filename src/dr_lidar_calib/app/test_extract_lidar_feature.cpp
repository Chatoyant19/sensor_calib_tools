#include <string>
#include <unordered_map>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "extract_lidar_feature.h"

std::string pcd_path_ = "/home/wd/datasets/2/pcd/0.pcd";
double voxel_size_ = 5;
double eigen_ratio_ = 0.05;
double p2line_dis_thre_ = 0.08;
double theta_min_ = 30;
double theta_max_ = 160;

bool use_ada_voxel_ = true;

double ransac_dis_threshold_ = 0.02;
int plane_size_threshold_ = 30;

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud =
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(pcd_path_, *lidar_cloud);
  
  if(use_ada_voxel_) {
    std::unique_ptr<ExtractLidarFeature> extract_lidar_feature = 
      std::make_unique<ExtractLidarFeature>(voxel_size_, eigen_ratio_, p2line_dis_thre_, theta_min_, theta_max_);
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_clouds = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    extract_lidar_feature->getEdgeFeaturesByAdaVoxel(lidar_cloud, line_clouds); 
    std::cout << "line_clouds size: " << line_clouds->size() << std::endl;     
  }
  else {
    std::unique_ptr<ExtractLidarFeature> extract_lidar_feature = 
      std::make_unique<ExtractLidarFeature>(voxel_size_, ransac_dis_threshold_, plane_size_threshold_, 
                                            p2line_dis_thre_,
                                            theta_min_, theta_max_);
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_clouds = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    extract_lidar_feature->getEdgeFeatures(lidar_cloud, line_clouds);  
    std::cout << "line_clouds size: " << line_clouds->size() << std::endl;     
  }
  
  return 0;
}