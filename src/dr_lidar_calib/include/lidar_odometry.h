#ifndef LIDAR_ODOMETRY
#define LIDAR_ODOMETRY

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class LidarOdometry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LidarOdometry(const bool& use_compensation);

  // ~LidarOdometry();
  void runOdometry(const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd,
                   Eigen::Matrix4d& curr_pose);

 private:
  void addPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd);
  void odomEstimation();
  void getPose(Eigen::Quaterniond& curr_q, Eigen::Vector3d& curr_t);
  void initMapWithPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in);
  void updatePointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in);
  void downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out);
  void addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
                         ceres::Problem& problem,
                         ceres::LossFunction* loss_function);
  void addPointsToMap(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledSurfCloud);
  void pointAssociateToMap(pcl::PointXYZI const* const pi,
                           pcl::PointXYZI* const po);

 private:
  double max_dis_;
  double min_dis_;
  bool use_compensation_ = false;

  // 10 hz
  double scan_period_ = 0.1;

  pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_filtered_;

  bool is_odom_inited_;
  // pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerMap_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfMap_;

  // optimization count
  int optimization_count_;
  Eigen::Isometry3d odom_;
  Eigen::Isometry3d last_odom_;
  // optimization variable
  double parameters_[7] = {0, 0, 0, 1, 0, 0, 0};
  Eigen::Map<Eigen::Quaterniond> q_w_curr_ =
      Eigen::Map<Eigen::Quaterniond>(parameters_);
  Eigen::Map<Eigen::Vector3d> t_w_curr_ =
      Eigen::Map<Eigen::Vector3d>(parameters_ + 4);

  // points downsampling before add to map
  // pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterEdge_;
  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterSurf_;
  double map_resolution_ = 0.2;

  // kd-tree
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfMap_;

  // local map
  pcl::CropBox<pcl::PointXYZI> cropBoxFilter_;
};

#endif