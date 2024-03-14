#include "lidar_compensation.h"
#include "ThreadPool.hpp"

std::unique_ptr<ThreadPool> workers_;

LidarCompensation::LidarCompensation(const LidarType& lidar_type) {
  lidar_type_ = lidar_type;
  is_compensate_inited_ = false;
  
  if(!initialize()) {
    std::cerr << "compensation intialize failed!" << std::endl;
  }
}

bool LidarCompensation::initialize(const int thread_num, const int motion_sample_num) {
  thread_num_ = thread_num;
  motion_sample_num_ = motion_sample_num;

  thread_num_ = std::min(10, std::max(1, thread_num_));
  motion_sample_num_ = std::min(1000, std::max(10, motion_sample_num_));

  constexpr int64_t kMillisecond = 1000;
  motion_sample_step_ = 100 * kMillisecond / motion_sample_num_;
  half_motion_smaple_step_ = motion_sample_step_ / 2;
  workers_ = std::unique_ptr<ThreadPool>(new ThreadPool(thread_num_));
  poses_.resize(motion_sample_num_ + 1);

  return true;
}

void LidarCompensation::compensate(const Eigen::Matrix4d& delta_pose, 
                                   const double& delta_time,  
                                   const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr& output_pcd) {
  Eigen::Vector6d velocity;
  deltaPoseToVelocity(delta_pose, delta_time, &velocity);
  std::vector<uint16_t> point_stamp;
  calcPointStamps(input_pcd, point_stamp);
  compensate(velocity, point_stamp, input_pcd, output_pcd);
}

void LidarCompensation::calcPointStamps(const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd, 
                                        std::vector<uint16_t>& stamp_vec) {
  switch(lidar_type_) {
    case LidarType::Hesai64:
      stamp_vec = calcHesai64PointStamps(origin_pcd);
      break;
    case LidarType::Bp32:
      stamp_vec = calcBppearl32PointStamps(origin_pcd);  
      break;
    case LidarType::Helios32:
      stamp_vec = calcHelios32PointStamps(origin_pcd);
      break;
    // case 3:
    //   stamp_vec = calcYijingPointStamps(origin_pcd);
    //   break;
    // case 4:
    //   stamp_vec = calcFT120PointStamps(origin_pcd);
    //   break;
    default:
      std::cerr << "wrong lidar type!" << std::endl;
      break;
  }

}

/*******hesai-64**********/ 
// 频率10Hz
std::vector<uint16_t> LidarCompensation::calcHesai64PointStamps(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pointcloud) { 
  std::vector<uint16_t> points_stamp_vec;

  std::vector<int> line_count_all(64,0);
  std::vector<std::pair<int,int>> point_inLine_index(pointcloud->points.size(),std::make_pair(-1,-1));
  for(size_t i = 0; i < pointcloud->points.size(); ++i) {
    double r = sqrt(pointcloud->at(i).x * pointcloud->at(i).x + pointcloud->at(i).y * pointcloud->at(i).y);
    double angle = atan(pointcloud->at(i).z / r) * 180 / M_PI;
    int ring_id = -1;
    if (angle >= -14 && angle <= -6)  //[-14~-6]  1°
      ring_id = int((angle + 14) + 2.5);
    else if (angle > -6 && angle <= 2) //(-6~2] 0.33° 
      ring_id = int((angle + 6) * 6.0 + 10.5);
    else if (angle > 2) //(2~
      ring_id = int((0.3289 * angle + 34.2368) + 0.5) + 24;
    else // (angle < -14) -19和-25
      ring_id = int((0.18132 * angle + 4.50549) + 0.5);
    if (angle > 15 || angle < -25 || ring_id > 63 || ring_id < 0) {
      continue;
    }

    line_count_all[ring_id]++;
    point_inLine_index[i].first = ring_id;
    point_inLine_index[i].second = line_count_all[ring_id];
  }
  
  // PointXYZIRT new_p;
  for(size_t i = 0; i < pointcloud->points.size(); ++i) {
    double point_stamp = static_cast<float>(point_inLine_index[i].second)/line_count_all[point_inLine_index[i].first]*0.1;
    point_stamp /= 2.0;
    constexpr int64_t kMillisecond = 1000;
    constexpr int64_t kSecond = 1000 * kMillisecond;
    uint16_t timestamp_2us = (uint16_t)static_cast<int64_t>(point_stamp * kSecond + 0.5);
    points_stamp_vec.emplace_back(timestamp_2us);
  }

  return points_stamp_vec;
}

/*******Robosense Bppearl-32**********/ 
// 频率10Hz
std::vector<uint16_t> LidarCompensation::calcBppearl32PointStamps(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pointcloud) {
  std::vector<uint16_t> points_stamp_vec;

  std::vector<int> line_count_all(32,0);
  std::vector<std::pair<int,int>> point_inLine_index(pointcloud->points.size(),std::make_pair(-1,-1));                                                     
  for(size_t i = 0; i < pointcloud->points.size(); ++i) {
    double r = sqrt(pointcloud->at(i).x * pointcloud->at(i).x + pointcloud->at(i).y * pointcloud->at(i).y);
    double angle = atan(pointcloud->at(i).z / r) * 180 / M_PI; // [0-90]
    if(angle < 0 || angle > 90) continue;
    int ring_id = angle / 2.903225806;
    line_count_all[ring_id]++;
    point_inLine_index[i].first = ring_id;
    point_inLine_index[i].second = line_count_all[ring_id];
  }

  for(size_t i = 0; i < pointcloud->points.size(); ++i) {
    double point_stamp = static_cast<float>(point_inLine_index[i].second)/line_count_all[point_inLine_index[i].first]*0.1;
    point_stamp /= 2.0;
    constexpr int64_t kMillisecond = 1000;
    constexpr int64_t kSecond = 1000 * kMillisecond;
    uint16_t timestamp_2us = (uint16_t)static_cast<int64_t>(point_stamp * kSecond + 0.5);
    points_stamp_vec.emplace_back(timestamp_2us);
  }

  return points_stamp_vec; 
}


/*******Robosense Helios-32**********/ 
// 频率10Hz
// todo
std::vector<uint16_t> LidarCompensation::calcHelios32PointStamps(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pointcloud) {
  std::vector<uint16_t> points_stamp_vec;

  std::vector<int> line_count_all(32,0);
  std::vector<std::pair<int,int>> point_inLine_index(pointcloud->points.size(),std::make_pair(-1,-1));                                                     
  for(size_t i = 0; i < pointcloud->points.size(); ++i) {
    double r = sqrt(pointcloud->at(i).x * pointcloud->at(i).x + pointcloud->at(i).y * pointcloud->at(i).y);
    double angle = atan(pointcloud->at(i).z / r) * 180 / M_PI; // [-55-15]
  }

  for(size_t i = 0; i < pointcloud->points.size(); ++i) {
    double point_stamp = double(i) / pointcloud->points.size() * 0.1;
    point_stamp /= 2.0;
    constexpr int64_t kMillisecond = 1000;
    constexpr int64_t kSecond = 1000 * kMillisecond;
    uint16_t timestamp_2us = (uint16_t)static_cast<int64_t>(point_stamp * kSecond + 0.5);
    points_stamp_vec.emplace_back(timestamp_2us);
  }

  return points_stamp_vec; 
}

void LidarCompensation::deltaPoseToVelocity(const Eigen::Matrix4d& delta_pose,
                                            const double& delta_time, 
                                            Eigen::Vector6d* velocity) {
  Eigen::Vector6d tf_vec;  
  Eigen::Quaterniond delta_q = Eigen::Quaterniond(delta_pose.block<3, 3>(0, 0));   
  Eigen::AngleAxisd angle_axis = convertQuaternionToAngleAxis(delta_q);
  tf_vec.head<3>() = angle_axis.angle() * angle_axis.axis();  
  tf_vec.tail<3>() = Eigen::Vector3d(delta_pose.block<3, 1>(0, 3));                                       
  *velocity = tf_vec / delta_time;
}

void LidarCompensation::compensate(const Eigen::Vector6d& velocity,
                                   const std::vector<uint16_t>& point_stamp,
                                   const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr& output_pcd) {
  if (output_pcd == nullptr) return;

  const int step = 10000;
  computeMotionSamples(velocity);
  int point_size = static_cast<int>(input_pcd->points.size());
  output_pcd =
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  output_pcd->resize(point_size);    

  assert(point_size == point_stamp.size());

  std::vector<std::future<void>> results;
  for (int start_id = 0; start_id < point_size; start_id += step) {
    int stop_id = start_id + step;
    if (stop_id > point_size) {
      stop_id = point_size;
    }

    results.emplace_back(workers_->enqueue(&LidarCompensation::transformThread, this,
                                           start_id, stop_id, point_stamp, input_pcd, output_pcd));                                      
  }

  for (auto& result : results) {
    result.wait();
  }
}

void LidarCompensation::computeMotionSamples(const Eigen::Vector6d& velocity) {
  const double motion_sample_step_second = static_cast<double>(motion_sample_step_) * 1e-6;

  double time = 0.0;
  for (int i = 0; i <= motion_sample_num_; i++) {
    Eigen::Vector6d pose_vec = time * velocity;
    // common::geometry::SE3 pose = common::geometry::SE3::exp(pose_vec);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    Eigen::Vector3d rotation_vec = pose_vec.head<3>(); 
    const double half_angle = rotation_vec.norm() / 2.0;
    if (half_angle <= std::numeric_limits<double>::epsilon()) {
      pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    }
    Eigen::Quaterniond q;
    q.w() = cos(half_angle);
    Eigen::Vector3d vec = rotation_vec.normalized();
    q.vec() = sin(half_angle) * vec;
    pose.block<3, 3>(0, 0) = q.toRotationMatrix();


    Eigen::Vector3d t = pose_vec.tail<3>();
    pose.block<3, 1>(0, 3) = t;

    poses_[i] = pose;
    time += motion_sample_step_second;
  }
}

void LidarCompensation::transformThread(const int start_id, const int stop_id,
                                        const std::vector<uint16_t>& points_stamp,
                                        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                                        pcl::PointCloud<pcl::PointXYZI>::Ptr& output_pcd) {
  for (int i = start_id; i < stop_id; i++) {
    auto& pt = input_pcd->points[i];
    auto& pt_out = output_pcd->points[i];

    int motion_sample_id =
        (static_cast<int64_t>(points_stamp[i]) * 2 + half_motion_smaple_step_) / motion_sample_step_;

    if (motion_sample_id >= motion_sample_num_) {
      motion_sample_id = motion_sample_num_ - 1;
    }

    const Eigen::Matrix4d& pose = poses_[motion_sample_id];
    // std::cout << "pose: " << pose << std::endl;

    Eigen::Vector4d pt_before(pt.x, pt.y, pt.z, 1.0);
    Eigen::Vector4d pt_after = pose * pt_before;

    pt_out.x = pt_after(0);
    pt_out.y = pt_after(1);
    pt_out.z = pt_after(2);
    pt_out.intensity = pt.intensity;
  }
}

Eigen::AngleAxisd LidarCompensation::convertQuaternionToAngleAxis(const Eigen::Quaterniond& q) {
  Eigen::AngleAxisd angle_axis;
  const double sin_half_theta = q.vec().norm();
  if (sin_half_theta <= std::numeric_limits<double>::epsilon()) {
    angle_axis.angle() = 0;
    angle_axis.axis() << 1, 0, 0;
    return angle_axis;
  }

  double theta;
  const double cos_half_theta = q.w();
  if (cos_half_theta > 0.0) {
    theta = 2 * atan2(sin_half_theta, cos_half_theta);
  } else {
    theta = -2 * atan2(sin_half_theta, -cos_half_theta);
  }

  angle_axis.angle() = theta;
  angle_axis.axis() = q.vec() / sin_half_theta;
  return angle_axis;
}