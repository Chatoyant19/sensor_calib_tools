// Copyright 2026. All Rights Reserved.
// Author: Dan Wang
/**********************************************************
 * purpose:
 *  show calibration result-project point cloud to image.
 *
 * pipeline:
 *    input: pcd and image file, camera intrinsics file, T_base-link_camera and T_base-link_lidar file 
 *    output: save jpg file, show calibration result
 *    
 * usage:
 *    ./test_handeye_calib /home/danwa/projects/calib_tools/config/config_handeye_calib.yaml
 * 
 *********************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>

#include "file_io.h"
#include "handeye.h"

std::string timestamp_pose_file1_, timestamp_pose_file2_;
std::string aligned_pose_file1_, aligned_pose_file2_;
int motionCount_ = 0;
bool scale_flag_ = false;
double tz_ = 0.0;
std::string calib_result_file_;

bool loadConfigFile(const std::string &config_file);

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: test_handeye_calib path_to_config_handeye_calib_file" << std::endl;
    return -1;
  }

  std::string config_file = argv[1];

  if (!loadConfigFile(config_file)) {
    std::cerr << "failed load config file from: " << config_file << std::endl;
    return -1;
  }

  StampedPoseVectorPtr pose_seq1 = std::make_unique<StampedPoseVector>();
  StampedPoseVectorPtr pose_seq2 = std::make_unique<StampedPoseVector>();
  if (!file_io::readStampPoseFromFile(timestamp_pose_file1_, pose_seq1) ||
      !file_io::readStampPoseFromFile(timestamp_pose_file2_, pose_seq2)) {
    std::cout << "Failed to read pose files!" << std::endl;
    return -1;
  }

  std::cout << "read pose1 size: " << pose_seq1->size() << std::endl;
  std::cout << "read pose2 size: " << pose_seq2->size() << std::endl;

  std::unique_ptr<HandEyeCalib> init_calib =
      std::make_unique<HandEyeCalib>(pose_seq2, pose_seq1);
  init_calib->processingPoses();
  Eigen::Matrix4d init_extrinsics = Eigen::Matrix4d::Identity();
  init_calib->estimate(init_extrinsics);
  Eigen::Matrix4d res = init_extrinsics.inverse();
  res(2, 3) = tz_;

  file_io::writeExtrinsicToPbFile(res, calib_result_file_);

  std::cout << "handeye_calib result: " << std::endl;
  Eigen::Quaterniond qua = Eigen::Quaterniond(res.block<3, 3>(0, 0));
  Eigen::Vector3d trans = Eigen::Vector3d(res.block<3, 1>(0, 3));
  std::cout << "res: " << res << std::endl;
  std::cout << "tx: " << trans.x() << std::endl
            << "ty: " << trans.y() << std::endl
            << "tz: " << trans.z() << std::endl;
  std::cout << "w: " << qua.w() << std::endl
            << "x: " << qua.x() << std::endl
            << "y: " << qua.y() << std::endl
            << "z: " << qua.z() << std::endl;
  return 0;
}

bool loadConfigFile(const std::string &config_file) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    return false;
  }

  
  fSettings["timestamp_pose_file1"] >> timestamp_pose_file1_;
  fSettings["timestamp_pose_file2"] >> timestamp_pose_file2_;
  fSettings["aligned_pose_file1"] >> aligned_pose_file1_;
  fSettings["aligned_pose_file2"] >> aligned_pose_file2_;
  fSettings["motionCount"] >> motionCount_;
  fSettings["scale_flag"] >> scale_flag_;
  fSettings["tz"] >> tz_;
  fSettings["calib_result_file"] >> calib_result_file_;
  std::cout << "timestamp_pose_file1: " << timestamp_pose_file1_ << std::endl;
  std::cout << "timestamp_pose_file2: " << timestamp_pose_file2_ << std::endl;
  std::cout << "aligned_pose_file1: " << aligned_pose_file1_ << std::endl;
  std::cout << "aligned_pose_file2: " << aligned_pose_file2_ << std::endl;
  std::cout << "motionCount: " << motionCount_ << std::endl;
  std::cout << "scale_flag: " << scale_flag_ << std::endl;
  std::cout << "tz: " << tz_ << std::endl;
  std::cout << "calib_result_file: " << calib_result_file_ << std::endl;

  return true;
}