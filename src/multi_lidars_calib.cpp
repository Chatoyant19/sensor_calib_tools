#include <unordered_map>
#include "multi_lidars_calib.h"
#include "pose_refine.hpp"
#include "handeye.h"
#include "common.h"
#include "eigen_types.hpp"

// #define debug

namespace multi_lidars_calib {
MultiLidarsCalib::MultiLidarsCalib(const int& step) {
  #ifdef debug
  std::cout << "step: " << step << std::endl;
  #endif
  step_ = step;
}

void MultiLidarsCalib::initBaseLidar(Lidar& lidar) {
  #ifdef debug
  std::cout << "lidar.use_compensation: " << lidar.use_compensation << std::endl;
  std::cout << "lidar.lidar_type: " << lidar.lidar_type << std::endl;
  std::cout << "lidar.min_dis: " << lidar.min_dis << std::endl;
  std::cout << "lidar.max_dis: " << lidar.max_dis << std::endl;
  #endif
  lidar.odomet = std::make_unique<LidarOdometry>(lidar.use_compensation);
  lidar.compens = std::make_unique<LidarCompensation>(lidar.lidar_type);
  lidar.lidar_poses_ptr = std::make_shared<StampedPoseVector>();
  // lidar.sample_poses_ptr = std::make_shared<StampedPoseVector>();
  // lidar.sample_pcds_ptr = std::make_shared<StampedPcdVector>();
  lidar.cnt = 0;
}

bool MultiLidarsCalib::processBaseLidar(Lidar& lidar, 
  const double& pcd_stamp, const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
  Eigen::Matrix4d& pose,
  pcl::PointCloud<pcl::PointXYZI>::Ptr& out_pcd) {
  if(processLidar(lidar, pcd_stamp, input_pcd, pose, out_pcd)) {
    ++lidar.cnt;
    lidar.lidar_poses_ptr->emplace_back(StampedPose(pcd_stamp, pose));
    if(lidar.cnt % step_ == 0) {
      // lidar.sample_poses_ptr->emplace_back(StampedPose(pcd_stamp, pose));
      // lidar.sample_pcds_ptr->emplace_back(StampedPcd(pcd_stamp, out_pcd));
      return true;
    }
  }
  return false;
}

bool MultiLidarsCalib::processLidar(Lidar& lidar, 
  const double& pcd_stamp, const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
  Eigen::Matrix4d& out_pose, pcl::PointCloud<pcl::PointXYZI>::Ptr& out_pcd) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZI>);
  for(auto p: *input_pcd) {
    if((!std::isfinite(p.x) || !std::isfinite(p.y) || !isfinite(p.z)) ||
      (std::sqrt(p.x * p.x + p.y * p.y) < lidar.min_dis) ||
      (std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) > lidar.max_dis)) continue;
    pcd->push_back(p); 
  }
  Eigen::Matrix4d pose;
  lidar.odomet->runOdometry(pcd, pose);
  
  if(lidar.lidar_type == LidarType::Hesai64 || lidar.lidar_type == LidarType::Bp32 ||
    lidar.lidar_type == LidarType::Helios32) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr compensated_pcd(new pcl::PointCloud<pcl::PointXYZI>);
    if(!lidarCompensate(lidar, pcd, pcd_stamp, pose, compensated_pcd)) {
      return false;
    }
    *out_pcd = *compensated_pcd; 
  }
  else {// todo: compensate
    *out_pcd = *input_pcd;
  }

  out_pose = pose;
  return true;
}

static double last_timestamp;
static Eigen::Matrix4d last_pose;
bool MultiLidarsCalib::lidarCompensate(Lidar& lidar,
                                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd,
                                       const double& timestamp, const Eigen::Matrix4d& pose,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr& compensated_pcd) {                                     
  if(lidar.compens->is_compensate_inited_ == false) {
    last_timestamp = timestamp;
    last_pose = pose;
    lidar.compens->is_compensate_inited_ = true;
    return false;
  }

  Eigen::Matrix4d delta_pose = last_pose.inverse() * pose;
  double delta_timestamp = timestamp - last_timestamp;

  lidar.compens->compensate(delta_pose, delta_timestamp, origin_pcd, compensated_pcd);

  last_timestamp = timestamp;
  last_pose = pose; 

  return true; 
}

// todo
Eigen::Matrix4d MultiLidarsCalib::estimateInitExtrinsics(const StampedPoseVectorPtr& pose_seq1, 
  const StampedPoseVectorPtr& pose_seq2, const double& tz) {
  std::unique_ptr<HandEyeCalib> init_calib = std::make_unique<HandEyeCalib>(pose_seq2, pose_seq1);
  init_calib->processingPoses();
  Eigen::Matrix4d init_extrinsics = Eigen::Matrix4d::Identity();
  init_calib->estimate(init_extrinsics);
  Eigen::Matrix4d res = init_extrinsics.inverse();
  res(2, 3) = tz;

  return res;  
}

void MultiLidarsCalib::runBaseLidar(const StampedPcdVectorPtr& pcds_seq,
                                    StampedPoseVectorPtr& pose_seq,
                                    StampedPcd& stamp_map,
                                    StampedPcd& stamp_visual) {
  refineBasePose(pcds_seq, pose_seq);
  Eigen::Matrix4d start_pose = pose_seq->at(0).second;
  double stamp = pose_seq->at(0).first;

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_surf(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>);

  for(size_t i = 0; i < pcds_seq->size(); ++i) {
    Eigen::Matrix4d curr_pose = pose_seq->at(i).second;

    Eigen::Matrix4d T_start_curr = start_pose.inverse() * curr_pose;
    pcl::transformPointCloud(*pcds_seq->at(i).second, *pcd_surf, T_start_curr);
    if(i == 0) {
      // stamp_visual = StampedPcd(stamp, pcd_surf);
      stamp_visual.first = stamp;
      stamp_visual.second = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
      *stamp_visual.second = *pcd_surf;
    }
    *map += *pcd_surf;
  }

  stamp_map = StampedPcd(stamp, map);
}

void MultiLidarsCalib::refineBasePose(const StampedPcdVectorPtr& pcds_seq,
  StampedPoseVectorPtr& pose_seq) {
  for(int loop = 0; loop < refine_pose_param_.max_iter; ++loop) {
    std::cout << "---------------------" << std::endl;
    std::cout << "iteration " << loop << std::endl;
    int window_size = pose_seq->size();
    std::unordered_map<VOXEL_LOC, pose_refine::OCTO_TREE*> surf_map;
    pose_refine::LM_OPTIMIZER lm_opt(window_size);

    for(size_t i = 0; i < window_size; ++i) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_surf(new pcl::PointCloud<pcl::PointXYZI>);
      if(refine_pose_param_.downsmp_base > 0) {
        downsample_voxel(*pcds_seq->at(i).second, *pcd_surf, refine_pose_param_.downsmp_base);
      }

      pose_refine::cut_voxel(surf_map, pcd_surf, pose_seq->at(i).second, 
                             i, window_size, refine_pose_param_.voxel_size, refine_pose_param_.eigen_thr);
    }

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();

    Eigen::Quaterniond refine_q;
    Eigen::Vector3d refine_t;
    for(int i = 0; i < window_size; ++i) {
      refine_q = Eigen::Quaterniond(pose_seq->at(i).second.block<3, 3>(0, 0));
      refine_t = Eigen::Vector3d(pose_seq->at(i).second.block<3, 1>(0, 3));
      assign_qt(lm_opt.poses[i], lm_opt.ts[i], refine_q, refine_t);
    }

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->feed_pt(lm_opt);

    lm_opt.optimize(); 

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;     

    std::vector<Eigen::Quaterniond, 
      Eigen::aligned_allocator<Eigen::Quaterniond> > quad_save;
    vector_vec3d trans_save;
    double residual_lm = 0.0;
    double residual_save = 0.0;
    for(int i = 0; i < window_size; ++i) {
      refine_q = Eigen::Quaterniond(pose_seq->at(i).second.block<3, 3>(0, 0));
      refine_t = Eigen::Vector3d(pose_seq->at(i).second.block<3, 1>(0, 3));
      quad_save.emplace_back(refine_q);
      trans_save.emplace_back(refine_t);
    }
    // lm_opt.evaluate_only_residual(quad_save, trans_save, residual_save);
    // std::cout << "####debug residual_save: " << residual_save << std::endl;
    // lm_opt.evaluate_only_residual(lm_opt.poses, lm_opt.ts, residual_lm);
    // std::cout << "####debug residual_lm: " << residual_lm << std::endl;

    for(int i = 0; i < window_size; ++i) {
      assign_qt(refine_q, refine_t, lm_opt.poses[i], lm_opt.ts[i]);
      pose_seq->at(i).second.block<3, 3>(0, 0) = refine_q.toRotationMatrix();
      pose_seq->at(i).second.block<3, 1>(0, 3) = refine_t;
    }

    // if(std::abs(residual_lm - residual_save) < 1e-4) {
    //   break;
    // }  
  }
  std::cout << "---------------------" << std::endl;
  std::cout << "complete" << std::endl;
}

double distance_thred = 7.0;
double radians_thred = 1.0;
TimesVector MultiLidarsCalib::getCutTimepairs(const int& cut_num, const StampedPoseVectorPtr& pose_seq) { 
  TimesVector time_pair;
  time_pair.resize(cut_num);
  int seg_cnt = 0;
  for(size_t i = 0; i < pose_seq->size(); ) { 
    size_t length = 30;
    if(!vehicleIsStatic(pose_seq, pose_seq->at(i).second, i, length)) {
      ++i;
      continue;
    }
    i = i + 20;
    time_pair[seg_cnt].first = pose_seq->at(i).first;
    Eigen::Matrix4d start_pose = pose_seq->at(i).second;
    bool find_traj_end = false;
    for(size_t j = i + 1; j < pose_seq->size(); ++j) {
      Eigen::Matrix4d curr_pose = pose_seq->at(j).second;
      Eigen::Matrix4d delta_pose = start_pose.inverse() * curr_pose;
      Eigen::Matrix3d delta_rot = delta_pose.block<3, 3>(0, 0);
      double delta_yaw = convertRotationToEulerYPR(delta_rot).x();
      double distance = std::sqrt(delta_pose.block<3, 1>(0, 3).x() * delta_pose.block<3, 1>(0, 3).x() +
                                  delta_pose.block<3, 1>(0, 3).y() * delta_pose.block<3, 1>(0, 3).y());
      if((distance > distance_thred || std::abs(delta_yaw) > radians_thred) && !find_traj_end) {
        time_pair[seg_cnt].second = pose_seq->at(j).first;
        find_traj_end = true;
      } 

      if(std::abs(delta_yaw) > 3.0) {
        i = j + 1;
        break;
      }
    }
    find_traj_end = false;

    ++seg_cnt;
    if(seg_cnt == cut_num) return time_pair;
  }
}

bool MultiLidarsCalib::vehicleIsStatic(const StampedPoseVectorPtr& pose_seq,
                                       const Eigen::Matrix4d& element, 
                                       const size_t& start,
                                       const size_t& length) {
  if(start + length > pose_seq->size())
    return false;  

  return std::all_of(pose_seq->begin() + start, pose_seq->begin() + start + length, 
                     [&element](const StampedPose& stamped_mat) { 
                        return stamped_mat.second == element; });                                      
}

} // namespace multi_lidars_calib