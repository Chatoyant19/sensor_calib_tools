
#ifndef MULTI_LIDARS_CALIB
#define MULTI_LIDARS_CALIB

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "lidar_odometry.h"

// just for LidarType_enum_type. how to forward declaration?
#include "lidar_compensation.h"

// class LidarOdometry;
// class LidarCompensation;

typedef std::pair<double, Eigen::Matrix4d> StampedPose;
typedef std::vector<StampedPose, Eigen::aligned_allocator<StampedPose>>
    StampedPoseVector;
typedef std::shared_ptr<StampedPoseVector> StampedPoseVectorPtr;

typedef std::pair<double, pcl::PointCloud<pcl::PointXYZI>::Ptr> StampedPcd;
typedef std::vector<StampedPcd, Eigen::aligned_allocator<StampedPcd>>
    StampedPcdVector;
typedef std::shared_ptr<StampedPcdVector> StampedPcdVectorPtr;

typedef std::pair<double, double> Times;
typedef std::vector<Times> TimesVector;

namespace multi_lidars_calib {

struct Lidar {
  double min_dis;
  double max_dis;
  bool use_compensation = false;
  LidarType lidar_type;
  double tz;

  std::unique_ptr<LidarOdometry> odomet;
  std::unique_ptr<LidarCompensation> compens;
  StampedPoseVectorPtr lidar_poses_ptr;
  int cnt;
  // StampedPoseVectorPtr sample_poses_ptr;
  // StampedPcdVectorPtr sample_pcds_ptr;
};

struct RefinePoseParam {
  int max_iter = 20;
  double voxel_size = 1;
  double eigen_thr = 10;
  double downsmp_base = 0.1;
};

typedef struct {
  // int scene_num;
  // int ref_num;
} MultiLidarsCalibParam;

class MultiLidarsCalib {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MultiLidarsCalib(const int& step);
  void initBaseLidar(Lidar& lidar);
  bool processBaseLidar(Lidar& lidar, const double& pcd_stamp,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                        Eigen::Matrix4d& pose,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr& out_pcd);
  Eigen::Matrix4d estimateInitExtrinsics(const StampedPoseVectorPtr& pose_seq1,
                                         const StampedPoseVectorPtr& pose_seq2,
                                         const double& tz);
  void runBaseLidar(const StampedPcdVectorPtr& pcds_seq,
                    StampedPoseVectorPtr& pose_seq, StampedPcd& stamp_map,
                    StampedPcd& stamp_visual);
  TimesVector getCutTimepairs(const int& cut_num,
                              const StampedPoseVectorPtr& pose_seq,
                              const size_t& th_time = 5);
  static bool routeIsOk(const int &cut_num, const StampedPoseVectorPtr &pose_seq,
                        const size_t &th_time);
 private:
  bool processLidar(Lidar& lidar, const double& pcd_stamp,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                    Eigen::Matrix4d& out_pose,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr& out_pcd);
  bool lidarCompensate(Lidar& lidar,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd,
                       const double& timestamp, const Eigen::Matrix4d& pose,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& compensated_pcd);
  void refineBasePose(const StampedPcdVectorPtr& pcds_seq,
                      StampedPoseVectorPtr& pose_seq);
  static bool vehicleIsStatic(const StampedPoseVectorPtr& pose_seq,
                              const Eigen::Matrix4d& element, const size_t& start,
                              const size_t& length);

 private:
  int step_;
  RefinePoseParam refine_pose_param_;
  static TimesVector time_pairs_;
};

}  // namespace multi_lidars_calib

#endif