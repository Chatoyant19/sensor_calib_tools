#ifndef FILE_IO
#define FILE_IO

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <queue>
#include <memory>

typedef std::pair<double, Eigen::Matrix4d> StampedPose;
typedef std::vector<StampedPose, Eigen::aligned_allocator<StampedPose>> StampedPoseVector;
typedef std::shared_ptr<StampedPoseVector> StampedPoseVectorPtr;

namespace file_io {
  void readExtrinsicFromPbFile(const std::string& pb_file, Eigen::Matrix4d& extrinsic);
  void readCamInFromXmlFile(const std::string& xml_file, std::string& model_type, 
                            cv::Mat& camera_intrinsic, cv::Mat& camera_distort,
                            int& img_height, int& img_width);
  void readCamExFromYmlFile(const std::string& yml_file, Eigen::Matrix4d& cam_extrinsic);   
  void readExtrinsicFromYamlFile(const std::string& file, Eigen::Matrix4d& extrinsic);          
  void writeExtrinsicToPbFile(const Eigen::Matrix4d& extrinsic, std::string& output_file);   
  void loadPcdFilePath(const std::string& pcd_folder, std::queue<std::string>& pcds_name_vec);
  void readStampPoseFromFile(const std::string& path, StampedPoseVectorPtr& stamp_pose_vec);    
  void writeStampPoseToFile(const StampedPoseVectorPtr& stamp_pose_vec, std::string& path);  
  bool read_timestamp_info(const std::string &input_img_stamp_file, std::vector<double> &img_stamps);
  int getSyncTimestampIdex(const std::vector<double>& timestamp_vec, const double pcd_start_time, 
                        const StampedPoseVectorPtr& stamp_pose_vec);
} // namespace file_io

#endif