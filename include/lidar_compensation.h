#ifndef LIDAR_COMPENSATION
#define LIDAR_COMPENSATION

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

enum LidarType {Hesai64, Bp32, Helios32, Ft120, Yijing};

namespace Eigen {
template <size_t _row> using VectorNd = Eigen::Matrix<double, _row, 1>;
template <size_t _dem> using MatrixNd = Eigen::Matrix<double, _dem, _dem>;
using Vector6d = VectorNd<6>;
using Matrix6d = MatrixNd<6>;
} // namespace Eigen

class LidarCompensation {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LidarCompensation(const LidarType& lidar_type);
  bool initialize(const int thread_num = 5, const int motion_sample_num = 100);
  void compensate(const Eigen::Matrix4d& delta_pose, 
                  const double& delta_time,  
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& output_pcd);
  bool is_compensate_inited_;
 private:
  void calcPointStamps(const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd, 
                       std::vector<uint16_t>& stamp_vec); 
  std::vector<uint16_t> calcHesai64PointStamps(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pointcloud);
  std::vector<uint16_t> calcBppearl32PointStamps(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pointcloud);
  std::vector<uint16_t> calcHelios32PointStamps(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pointcloud);                       
  void deltaPoseToVelocity(const Eigen::Matrix4d& delta_pose,
                           const double& delta_time, 
                           Eigen::Vector6d* velocity);
  void compensate(const Eigen::Vector6d& velocity,
                  const std::vector<uint16_t>& point_stamp,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& output_pcd);
  void computeMotionSamples(const Eigen::Vector6d& velocity);
  void transformThread(const int start_id, const int stop_id,
                       const std::vector<uint16_t>& points_stamp,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& output_pcd);
  Eigen::AngleAxisd convertQuaternionToAngleAxis(const Eigen::Quaterniond& q);                       


 private:
  LidarType lidar_type_;
  int thread_num_;
  int motion_sample_num_;
  int64_t motion_sample_step_;
  double half_motion_smaple_step_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses_;
  
};

#endif