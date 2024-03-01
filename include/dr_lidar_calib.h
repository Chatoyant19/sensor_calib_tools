#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class ExtractImageFeature;
class ExtractLidarFeature;
class FloorPlaneConstriant;
class MatchFeatures;

class Camera {
 public:
  std::string cam_name_;
  std::string camera_model_ = "fisheye";
  int width_, height_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  Eigen::Matrix4d Tx_dr_C_;

  std::vector<cv::Mat> raw_imgs;
  std::vector<cv::Mat> rgb_imgs;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rgb_edge_clouds_;

  Eigen::Matrix4d Tx_C_L_;

  void update_TxCL(const Eigen::Matrix4d& T)
  {
    Tx_C_L_ << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
               T(1, 0), T(1, 1), T(1, 2), T(1, 3),
               T(2, 0), T(2, 1), T(2, 2), T(2, 3),
               0.0,     0.0,     0.0,    1.0;                 
  }

  void update_TDC(const Eigen::Matrix4d& T)
  {
    Tx_dr_C_ << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
                T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                T(2, 0), T(2, 1), T(2, 2), T(2, 3),
                0.0,     0.0,     0.0,     1.0;
  }
};

class Lidar {
public:
  Eigen::Matrix4d Tx_dr_L_;
  // 存储平面交接点云
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>  plane_line_cloud_vec_; 
  std::vector<pcl::PointCloud<pcl::PointXYZI>> floor_plane_vec_;

  void update_TDL(const Eigen::Matrix4d& T) {
    Tx_dr_L_ << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
                T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                T(2, 0), T(2, 1), T(2, 2), T(2, 3),
                0.0,     0.0,     0.0,     1.0;
  }
};

typedef struct {
  int scene_num;

  /***camera***/
  std::vector<std::vector<cv::Mat>> images;
  std::vector<std::string> cams_name_vec;
  std::vector<std::string> cams_model_vec;
  std::vector<cv::Mat> camera_matrix_vec;
  std::vector<cv::Mat> dist_coeffs_vec;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> camera_extrinsics_vec;
  double canny_threshold;
  int rgb_edge_minLen;

  /***lidar***/
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> map_pcd_vec_;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> visual_pcd_vec_;
  Eigen::Matrix4d init_Tx_dr_L;
  bool use_ada_voxel;
  double voxel_size;
  double eigen_ratio; 
  double p2line_dis_thred;
  double theta_min;
  double theta_max;
  double ransac_dis_threshold;
  int plane_size_threshold;

  bool show_residual;
  std::string result_path;
} DrLidarCalibParam;

class DrLidarCalib{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
  DrLidarCalib(DrLidarCalibParam param) : param_(param) {}
  void init();
  void run(Eigen::Matrix4d& Tx_dr_L, 
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& cams_extrinsics_vec);
 private:
  void initLidar();
  void initCameras();
  void extractImagesFeatures();

 private:
  DrLidarCalibParam param_;

  std::vector<Camera> cams_;
  Lidar lidar_;

  std::unique_ptr<FloorPlaneConstriant> floor_plane_constraint_;
  std::unique_ptr<ExtractImageFeature> extract_image_feature_;
  std::unique_ptr<ExtractLidarFeature> extract_lidar_feature_;
  std::unique_ptr<MatchFeatures> match_features_;
};