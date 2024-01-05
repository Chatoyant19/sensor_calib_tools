#ifndef LIDAR_CAMERA_CALIB_HPP
#define LIDAR_CAMERA_CALIB_HPP

#include "CustomMsg.h"
#include "common.h"
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sstream>
#include <std_msgs/Header.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unordered_map>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <fcntl.h>   // linux open() function
#include <unistd.h>  // linux close() function
#include <google/protobuf/io/coded_stream.h>           // CodedInputStream
#include <google/protobuf/io/zero_copy_stream.h>       // ZeroCopyInputStream,
#include <google/protobuf/io/zero_copy_stream_impl.h>  // FileInputStream, FileOutputStream
#include <google/protobuf/text_format.h>               // google::protobuf::TextFormat
#include <google/protobuf/message.h>

#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "rapidxml_print.hpp"

#include "BA/mypcl.hpp"
#include "BA/ba.hpp"
#include "BA/tools.hpp"

#include "sensor_extrinsic.pb.h"

// #define calib
// #define online
enum CameraName {
    Avm_front = 0,
    Avm_left,
    Avm_rear,
    Avm_right,
    Left_front,
    Left_rear,
    Right_front,
    Right_rear,
    Front_near,
    Rear_medium
  };

class Lidar
{
public:
  Eigen::Matrix4d Tx_dr_L_;
  Eigen::Matrix3d R_dr_L_;
  Eigen::Vector3d t_dr_L_;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcd_vec_;
  // std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> lidar_line_cloud_3d_vec;
  // 存储平面交接点云
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>  plane_line_cloud_vec_; 
  // std::vector<std::vector<int>> plane_line_number_vec_;
  pcl::PointCloud<pcl::PointXYZI> floor_plane_;

  void update_Rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
  {
    R_dr_L_ << R(0, 0), R(0, 1), R(0, 2),
              R(1, 0), R(1, 1), R(1, 2),
              R(2, 0), R(2, 1), R(2, 2);
    t_dr_L_ << t(0), t(1), t(2);
    Tx_dr_L_ << R(0, 0), R(0, 1), R(0, 2), t(0),
               R(1, 0), R(1, 1), R(1, 2), t(1),
               R(2, 0), R(2, 1), R(2, 2), t(2),
               0.0, 0.0, 0.0, 1.0;
  }

  void update_T(const Eigen::Matrix4d& T)
  {
    R_dr_L_ << T(0, 0), T(0, 1), T(0, 2),
              T(1, 0), T(1, 1), T(1, 2),
              T(2, 0), T(2, 1), T(2, 2);
    t_dr_L_ << T(0, 3), T(1, 3), T(2, 3);
    Tx_dr_L_ << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
                T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                T(2, 0), T(2, 1), T(2, 2), T(2, 3),
                0.0,     0.0,     0.0,     1.0;
  }
};

class Camera
{
public:
  std::string cam_name_;
  float fx_, fy_, cx_, cy_, k1_, k2_, k3_, k4_;
  int width_, height_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Mat fixed_ext_;
  Eigen::Matrix4d Tx_dr_C_;
  Eigen::Matrix3d R_dr_C_; // fix rotation
  Eigen::Vector3d t_dr_C_; // fix trans
  std::vector<cv::Mat> rgb_imgs;
  std::vector<cv::Mat> raw_imgs;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rgb_edge_clouds_;

  Eigen::Matrix4d Tx_C_L_;
  Eigen::Matrix3d R_C_L_; //refine value
  Eigen::Vector3d t_C_L_; //refine value

  void update_TxCL(const Eigen::Matrix4d& T)
  {
    Tx_C_L_ << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
               T(1, 0), T(1, 1), T(1, 2), T(1, 3),
               T(2, 0), T(2, 1), T(2, 2), T(2, 3),
               0.0,     0.0,     0.0,    1.0;
    R_C_L_ << T(0, 0), T(0, 1), T(0, 2),
              T(1, 0), T(1, 1), T(1, 2),
              T(2, 0), T(2, 1), T(2, 2);
    t_C_L_ << T(0, 3),
              T(1, 3),
              T(2, 3);                            
  }

  void update_Rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
  {
    R_dr_C_ << R(0, 0), R(0, 1), R(0, 2),
              R(1, 0), R(1, 1), R(1, 2),
              R(2, 0), R(2, 1), R(2, 2);
    t_dr_C_ << t(0), t(1), t(2);
    Tx_dr_C_ << R(0, 0), R(0, 1), R(0, 2), t(0),
               R(1, 0), R(1, 1), R(1, 2), t(1),
               R(2, 0), R(2, 1), R(2, 2), t(2),
               0.0, 0.0, 0.0, 1.0;
  }

  void update_T(const Eigen::Matrix4d& T)
  {
    R_dr_C_ << T(0, 0), T(0, 1), T(0, 2),
              T(1, 0), T(1, 1), T(1, 2),
              T(2, 0), T(2, 1), T(2, 2);
    t_dr_C_ << T(0, 3), T(1, 3), T(2, 3);
    Tx_dr_C_ << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
                T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                T(2, 0), T(2, 1), T(2, 2), T(2, 3),
                0.0,     0.0,     0.0,     1.0;
  }
};

class Calibration {
public:

  enum ProjectionType { DEPTH, INTENSITY, BOTH };
  enum Direction { UP, DOWN, LEFT, RIGHT };
  std::string lidar_topic_name_ = "";
  std::string image_topic_name_ = "";

  std::string image_path_;
  std::string pcd_path_;
  std::string cameras_intrinsic_path_;
  std::string cameras_extrinsic_path_;
  int scene_num_ = 0;
  std::string init_extrinsic_file_;
  std::string result_path_;

  bool is_enhance_rgb_ = false;
  int is_enhance_depth_ = 0;
  bool is_enhance_intensity_ = false;
  bool is_fill_lidar_image_ = true;
  int depth_edge_minLen_ = 80;
  int intensity_edge_minLen_ = 100;
  int rgb_edge_minLen_ = 100;
  int depth_canny_threshold_ = 20;
  int rgb_canny_threshold_ = 20;
  int intensity_canny_threshold_ = 20;
  int min_depth_ = 0.0;
  int max_depth_ = 50;
  float min_cost_ = 1000;
  int plane_max_size_ = 8;
  float detect_line_threshold_ = 0.02;
  int line_number_ = 0;
  int color_intensity_threshold_ = 0;
  cv::Mat connect_img_;
  ProjectionType projection_type_ = DEPTH;
  Eigen::Vector3d optimize_euler_angle_;
  Eigen::Vector3d adjust_euler_angle_;
  int match_type_;
  bool update_camera_extrinsic_;
  bool use_ground_lines_;
  std::string save_lidar_extrinsic_name_;
  std::string img_file_extension_;

  Calibration(const std::string& CalibCfgFile, const std::string& ResultPath, const int& debugMode);
  void loadImgAndPointcloud(const std::vector<std::string>& pcd_paths,
                            const std::vector<std::vector<std::string>>& cams_paths);
  bool loadCalibConfig(const std::string &config_file);

  void loadCamConfigPath_old();
  bool loadCameraConfigFromYmlFile(const std::vector<std::string>& cam_intrinsic_vec, 
                                 const std::vector<std::string>& cam_extrinsic_vec,
                                 std::vector<Camera>& cams);
  void readCamInFromYmlFile(const std::string& yml_file, Camera& cam);  
  void readCamExFromYmlFile(const std::string& yml_file, Camera& cam);                               
  void loadCamConfigPath_new();
  bool loadCameraConfig(const std::vector<std::string>& cam_intrinsic_vec, 
                               const std::vector<std::string>& cam_extrinsic_vec,
                               std::vector<Camera>& cams);
  void readCamInFromXmlFile(const std::string& xml_file, Camera& cam);   
  void readCamExFromPbFile(const std::string& pb_file, Camera& cam);
  void readSensorExFromPbFile(const std::string& pb_file, Eigen::Matrix4d& extrinsic);
  bool readProtoFromTextFile(const std::string& file, google::protobuf::Message* proto);

  void readExtriFromYamlFile(const std::string& file, Eigen::Matrix4d& T_dr_L);

  bool loadConfig(const std::string &configFile);
  bool checkFov(const Camera& cam, const cv::Point2d &p);
  void colorCloud(const Vector6d &extrinsic_params, const int density,
                  const cv::Mat &rgb_img,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &color_cloud);
  void edgeDetector(const int &height, const int &width, 
                    const int &canny_threshold, const int &edge_threshold,
                    const cv::Mat &src_img, cv::Mat &edge_img,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud);
  void projection(const Camera& cam,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
                  const ProjectionType projection_type, const bool is_fill_img,
                  cv::Mat &projection_img);
  void calcLine(const std::vector<Plane> &plane_list, const double voxel_size,
                const Eigen::Vector3d origin,
                std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list);
  cv::Mat fillImg(const cv::Mat &input_img, const Direction first_direct,
                  const Direction second_direct);
  std::vector<pcl::PointCloud<pcl::PointXYZI>> linesFromGroundPlane(const pcl::PointCloud<pcl::PointXYZI>& cloud);
  void buildPnp(const Vector6d &extrinsic_params, const int dis_threshold,
                const bool show_residual,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d,
                std::vector<PnPData> &pnp_list);
  void buildVPnp(const Camera& cam, const int& dis_threshold,
    const bool& show_residual,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cam_edge_cloud_2d_vec,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& lidar_line_cloud_3d_vec,
    /*const std::vector<std::vector<int>>& plane_line_number,*/
    std::vector<std::vector<VPnPData>>& pnp_list_vec);     
  void buildVPnp(const Camera& cam, const int& dis_threshold,
    const int& cnt,
    const bool& show_residual,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cam_edge_cloud_2d_vec,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& lidar_line_cloud_3d_vec,
    /*const std::vector<std::vector<int>>& plane_line_number_vec,*/
    std::vector<std::vector<VPnPData>>& pnp_list_vec);                                         

  cv::Mat getConnectImg(const Camera& cam,
                const int dis_threshold,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud);
  cv::Mat getProjectionImg(const Camera& cam, 
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_lidar_cloud,
                           const cv::Mat& rgb_image);
  cv::Mat showPcdOnImg(const Camera& cam, 
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_lidar_cloud,
                       const cv::Mat& rgb_image);
  void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                 const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map);
  void LiDAREdgeExtraction(
       const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map,
    const float ransac_dis_thre, const int plane_size_threshold,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d/*,
    std::vector<int>& plane_line_number*/);
  void merge_planes(std::vector<Plane>& planes);


  void cut_voxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                double voxel_size, float eigen_ratio);

  void calcDirection(const std::vector<Eigen::Vector2d> &points,
                     Eigen::Vector2d &direction);
  void calcResidual(const Vector6d &extrinsic_params,
                    const std::vector<VPnPData> vpnp_list,
                    std::vector<float> &residual_list);
  void calcCovarance(const Vector6d &extrinsic_params,
                     const VPnPData &vpnp_point, const float pixel_inc,
                     const float range_inc, const float degree_inc,
                     Eigen::Matrix2f &covarance);
  Eigen::Vector3d convertRotationMatrixToEulerYPR(const Eigen::Matrix3d& R);
  Eigen::Matrix3d convertEulerYPRToRotationMatrix(const Eigen::Vector3d& yaw_pitch_roll);
  Eigen::AngleAxisd convertEulerYPRToAngleAxis(const Eigen::Vector3d& yaw_pitch_roll);

  void writeCamExToPbFile(const Eigen::Matrix4d& T_dr_L, std::string& output_file);
  bool writeProtoToTextFile(std::string& file,
                            const google::protobuf::Message& proto);
  void extractFloorPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                         const Eigen::Matrix4d& T_dr_L,
                         pcl::PointCloud<pcl::PointXYZI>& pcd_floor,
                         std::vector<pcl::PointCloud<pcl::PointXYZI>>& lines);
  bool addFloorConstriant(const pcl::PointCloud<pcl::PointXYZI>& pcd_floor,
                          const Eigen::Matrix4d& T_dr_L,
                          Eigen::Matrix4d& T_dr_L_new);                              

  int is_use_custom_msg_;
  float voxel_size_;
  float eigen_ratio_;
  float down_sample_size_;
  float ransac_dis_threshold_;
  float plane_size_threshold_;
  float theta_min_;
  float theta_max_;
  float direction_theta_min_ = cos(DEG2RAD(30.0));
  float direction_theta_max_ = cos(DEG2RAD(150.0));
  float line_dis_threshold_;
  float min_line_dis_threshold_;
  float max_line_dis_threshold_;

  // Camera Settings
  std::vector<Camera> cams;
  CameraName camera_name_;

  // Lidar Settings
  Lidar lidar;
  // // 存储平面交接点云
  // pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
  // std::vector<int> plane_line_number_;

  std::string ResultPath_;
  int debugMode_;
};

Calibration::Calibration(const std::string& CalibCfgFile, const std::string& ResultPath, 
  const int& debugMode) {
  ResultPath_ = ResultPath;
  debugMode_ = debugMode;

  // load image and pcd path, image_name, and some fix parameter
  loadCalibConfig(CalibCfgFile);

  // readExtriFromYamlFile(init_extrinsic_file_, lidar.Tx_dr_L_);
  readSensorExFromPbFile(init_extrinsic_file_, lidar.Tx_dr_L_);
  lidar.update_T(lidar.Tx_dr_L_);
  // std::cout << "init lidar extrinsic: " << lidar.Tx_dr_L_.topLeftCorner(3, 3)
  if(debugMode_) {
    Eigen::Matrix3d init_R = lidar.Tx_dr_L_.topLeftCorner(3, 3);
    Eigen::Quaterniond init_q = Eigen::Quaterniond(init_R);
    Eigen::Vector3d init_t = Eigen::Vector3d(lidar.Tx_dr_L_.topRightCorner(3, 1));
    std::cout << "init_q: " << init_q.coeffs().transpose() << std::endl
              << "init_t: " << init_t.transpose() << std::endl;
  }

  // if old version camera's intrinsic and extrinsic files
  // loadCamConfigPath_old();
  // if old version camera's intrinsic and extrinsic files
  loadCamConfigPath_new();
  std::vector<std::string> pcd_paths;
  std::vector<std::vector<std::string>> cams_paths; 
  for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
    std::string pcd_path = pcd_path_ + "/" + std::to_string(scene_index) + ".pcd"; 
    pcd_paths.emplace_back(pcd_path);
  }
  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    std::string base_img_path = image_path_ + "/" + cams[cam_index].cam_name_;
    std::vector<std::string> cam_paths;
    for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
      std::string cam_path = base_img_path + "/" + std::to_string(scene_index) + "." + img_file_extension_;
      cam_paths.emplace_back(cam_path);
    }
    cams_paths.emplace_back(cam_paths);
  }
  if(debugMode_) {
    std::cout << "pcd path: " << std::endl;
    for(int i = 0; i < pcd_paths.size(); ++i) {
      std::cout << pcd_paths[i] << std::endl;
    }
    std::cout << "img path: " << std::endl;
    for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
      std::cout << "****[" << cams[cam_index].cam_name_ << "******]" << ": " << std::endl;
      for(int i = 0; i < cams_paths[cam_index].size(); ++i) {
        std::cout << cams_paths[cam_index][i] << std::endl;
      }
    }
  }

  // image: img[cam_index][scene_index]
  // pcd: pcd[scene_index]
  loadImgAndPointcloud(pcd_paths, cams_paths);
  
  // add floor plane constraint
  std::string use_pcd_path = pcd_path_ + "/" + std::to_string(0) + ".pcd";
  pcl::PointCloud<pcl::PointXYZI>::Ptr use_pcd(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::io::loadPCDFile(use_pcd_path, *use_pcd);
  std::vector<pcl::PointCloud<pcl::PointXYZI>> ground_lines;
  extractFloorPlane(use_pcd, lidar.Tx_dr_L_, lidar.floor_plane_,ground_lines);

  for(int i=0;i<ground_lines.size();i++)
      pcl::io::savePCDFileBinary(result_path_+"/gournd_line_"+std::to_string(i)+".pcd",ground_lines[i]);

  // for(size_t i = 0 ; i < 10; ++i) {
   Eigen::Matrix4d T_dr_L_new = Eigen::Matrix4d::Identity();
  if(addFloorConstriant(lidar.floor_plane_, lidar.Tx_dr_L_, T_dr_L_new))
    lidar.update_T(T_dr_L_new);
  // }

  // update T_C_L after add_floor_plane constraint
  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    cams[cam_index].Tx_C_L_ = cams[cam_index].Tx_dr_C_.inverse() * lidar.Tx_dr_L_;
    cams[cam_index].R_C_L_ = cams[cam_index].Tx_C_L_.topLeftCorner(3, 3);
    cams[cam_index].t_C_L_ = cams[cam_index].Tx_C_L_.topRightCorner(3, 1);
  }

  // show init calib result
  if(debugMode_) {
    for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
      // Eigen::Matrix4d init_Tx_C_L = cams[cam_index].Tx_C_L_;
      for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
        // pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcd = lidar.pcd_vec_[scene_index];
        // cv::Mat rgb_image = cams[cam_index].rgb_imgs[scene_index];
        cv::Mat rgb_image = cams[cam_index].raw_imgs[scene_index];
        std::string visual_pcd_path = pcd_path_ + "/" + std::to_string(scene_index) + "_visual.pcd";
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_point(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(visual_pcd_path, *raw_point);
      
        cv::Mat init_img = getProjectionImg(cams[cam_index], raw_point, rgb_image);
        // cv::Mat init_img = showPcdOnImg(cams[cam_index], raw_pcd, rgb_image);

        cv::imwrite(result_path_ + "/" + cams[cam_index].cam_name_ + "_sceneID_" + std::to_string(scene_index) + "_init.png", init_img);
      }
    }
  }
  // else {
  //   // pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcd = lidar.pcd_vec_[0];
  //   std::string visual_pcd_path = pcd_path_ + "/" + "0_visual.pcd";
  //   pcl::PointCloud<pcl::PointXYZI>::Ptr raw_point(new pcl::PointCloud<pcl::PointXYZI>());
  //   pcl::io::loadPCDFile(visual_pcd_path, *raw_point);
  //   cv::Mat rgb_image = cams[0].rgb_imgs[0];
  //   cv::Mat init_img = getProjectionImg(cams[0], raw_point, rgb_image);
  //   // cv::Mat init_img = showPcdOnImg(cams[0], raw_pcd, rgb_image);
  //   cv::imwrite(result_path_ + "/" + cams[0].cam_name_ + "_sceneID_" + std::to_string(0) + "_init.png", init_img);
  // }

  time_t t1 = clock();
  // extract lidar pcd lines
  lidar.plane_line_cloud_vec_.resize(scene_num_);
  // lidar.plane_line_number_vec_.resize(scene_num_);

  for(int i = 0; i < scene_num_; ++i) {
    lidar.plane_line_cloud_vec_[i] = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>);
  }
  for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
    std::cout << "*****sceneID: [" << scene_index << "]*****" << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_raw_pcd = lidar.pcd_vec_[scene_index];

    std::unordered_map<VOXEL_LOC, Voxel *> voxel_map;
    // wd todo: refine extract lidar pcd's line features
    initVoxel(input_raw_pcd, voxel_size_, voxel_map);
    // 存储平面交接点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> plane_line_number;
    LiDAREdgeExtraction(voxel_map, ransac_dis_threshold_, plane_size_threshold_,
                      plane_line_cloud/*, plane_line_number*/);
    std::cout << "pcd line feature points num is: " << plane_line_cloud->size() << ", " 
              << "lines num is: " << line_number_ << std::endl;
    plane_line_cloud->width = plane_line_cloud->points.size();
    plane_line_cloud->height = 1;                      
    lidar.plane_line_cloud_vec_[scene_index] = plane_line_cloud; 

    // // add cut_voxel, estimate_edge
    // std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
    // cut_voxel(input_raw_pcd, surf_map, voxel_size_, eigen_ratio_);  
    // std::cout << "surf_map size: " << surf_map.size() << std::endl;
    // for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
    //     iter->second->recut();
    // pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud =
    //   pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);    
    // estimate_edge(surf_map, plane_line_cloud);
    // std::cout << "pcd line feature points num is: " << plane_line_cloud->size() /*<< ", " 
    //           << "lines num is: " << line_number_ */<< std::endl;
    // plane_line_cloud->width = plane_line_cloud->points.size();
    // plane_line_cloud->height = 1;                      
    // lidar.plane_line_cloud_vec_[scene_index] = plane_line_cloud;         
  }
  // // wd to check lidar pcd lines
  // for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
  //   std::vector<int> plane_line_number = lidar.plane_line_number_vec_[scene_index];
  //   std::cout << "wd check plane_line_number: " << std::endl;
  //   for(int i = 0; i < plane_line_number.size(); ++i) {
  //     std::cout << plane_line_number[i] << " ";
  //   }
  //   std::cout << std::endl;

  //   std::cout << "wd check plane_line_cloud: " << std::endl;
  //   pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud = lidar.plane_line_cloud_vec_[scene_index];
  //   std::cout << "size: " << plane_line_cloud->size() << std::endl;
  //   plane_line_cloud->width = plane_line_cloud->points.size();
  //   plane_line_cloud->height = 1;
    // pcl::io::savePCDFile("/home/wd/datasets/65/0817/line.pcd", *plane_line_cloud);
  // }
  
  ROS_INFO_STREAM("Init voxel sucess!");

  if(use_ground_lines_)
      for(auto c:ground_lines)
          *lidar.plane_line_cloud_vec_[0] += c;


    time_t t2 = clock();
  std::cout << "extract all pcd edges use time: " << (double)(t2-t1)/(CLOCKS_PER_SEC) << "s" << std::endl;

  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    for(int scene_index = 0; scene_index < cams[cam_index].rgb_imgs.size(); ++scene_index) {
      if(!cams[cam_index].rgb_imgs[scene_index].data) {
        std::string msg = "Can not load image from " + image_path_ + "/" + cams[cam_index].cam_name_ + 
            "/" + std::to_string(scene_index) + "." + img_file_extension_;
        ROS_ERROR_STREAM(msg.c_str());
        exit(-1);
      }
    }
  }
  ROS_INFO_STREAM("Load all data!");

  time_t t3 = clock();
  // extract image lines
  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    // std::vector<cv::Mat> grey_imgs, rgb_edge_imgs;
    // grey_imgs.resize(scene_num_);
    int height = cams[cam_index].height_;
    int width = cams[cam_index].width_;

    cams[cam_index].rgb_edge_clouds_.resize(scene_num_);
    for(int i = 0; i < scene_num_; ++i) {
      cams[cam_index].rgb_edge_clouds_[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    }

    for(int scene_index = 0; scene_index < cams[cam_index].rgb_imgs.size(); ++scene_index) {
      cv::Mat rgb_edge_img, grey_img;
      cv::cvtColor(cams[cam_index].rgb_imgs[scene_index], grey_img, cv::COLOR_BGR2GRAY);

      pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);

      edgeDetector(height, width, rgb_canny_threshold_, rgb_edge_minLen_, grey_img,
             rgb_edge_img, rgb_egde_cloud); 
                 
      cams[cam_index].rgb_edge_clouds_[scene_index] =  rgb_egde_cloud;
    }
  }
  time_t t4 = clock();
   std::cout << "extract all imgs edges use time: " << (double)(t4-t3)/(CLOCKS_PER_SEC) << "s" << std::endl;

  // // wd to check image lines
  // for(int i = 0; i < cams.size(); ++i) {
  //   for(int j = 0; j < scene_num_; ++j) {
  //     pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_cloud = cams[i].rgb_edge_clouds_[j];
  //     std::cout << "rgb_egde_cloud size: " << rgb_egde_cloud->size() << std::endl;
  //     pcl::io::savePCDFile("/home/wd/datasets/test3.pcd", *rgb_egde_cloud);
  //   }
  // }

  ROS_INFO_STREAM("Initialization complete");
}

bool Calibration::loadCalibConfig(const std::string& config_file) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if(!fSettings.isOpened())
  {
    std::cerr << "Failed to open settings file at: " << config_file << std::endl;
    exit(-1);
  }

  fSettings["ImageFilesPath"] >> image_path_;
  fSettings["LiDARFilesPath"] >> pcd_path_;
  fSettings["CamsIntrinsicPath"] >> cameras_intrinsic_path_;
  fSettings["CamsExtrinsicPath"] >> cameras_extrinsic_path_;
  scene_num_ = fSettings["SceneNum"];

  std::vector<std::string> cams_name_vec;
  fSettings["CameraName"] >> cams_name_vec;
  // init cameras
  cams.resize(cams_name_vec.size());
  std::cout << "init " << cams.size() << " cameras: ";
  
  for(int cam_index = 0; cam_index < cams_name_vec.size(); ++cam_index) {
    // std::cout << "[" << cams_name_vec[cam_index] << "] ";
    cams[cam_index].cam_name_ = cams_name_vec[cam_index];
    /*
    if(cams[cam_index].cam_name_ == "front_near") {
      camera_name_ = CameraName::Front_near;
    }
    else if(cams[cam_index].cam_name_ == "avm_front") {
      camera_name_ = CameraName::Avm_front;
    }
    else if(cams[cam_index].cam_name_ == "avm_left") {
      camera_name_ = CameraName::Avm_left;
    }
    else if(cams[cam_index].cam_name_ == "avm_rear") {
      camera_name_ = CameraName::Avm_rear;
    }
    else if(cams[cam_index].cam_name_ == "avm_right") {
      camera_name_ = CameraName::Avm_right;
    }
    else if(cams[cam_index].cam_name_ == "left_front") {
      camera_name_ = CameraName::Left_front;
    }
    else if(cams[cam_index].cam_name_ == "left_rear") {
      camera_name_ = CameraName::Left_rear;
    } else if(cams[cam_index].cam_name_ == "right_front") {
      camera_name_ = CameraName::Right_front;
    }
    else if(cams[cam_index].cam_name_ == "right_rear") {
      camera_name_ = CameraName::Right_rear;
    }
    else if(cams[cam_index].cam_name_ == "rear_medium") {
      camera_name_ = CameraName::Rear_medium;
    }*/
  }
  // std::cout << std::endl;
  
  fSettings["init_extrinsic_file"] >> init_extrinsic_file_;
  // readExtriFromYamlFile(init_extrinsic_file_, lidar.Tx_dr_L_);

  fSettings["result_path"] >> result_path_;
  if(access(result_path_.c_str(),0) != 0) {
    mkdir(result_path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  // cv::Mat init_Tx_dr_L;
  // fSettings["Tx_dr_L"] >> init_Tx_dr_L;
  // lidar.Tx_dr_L_ << init_Tx_dr_L.at<double>(0, 0), init_Tx_dr_L.at<double>(0, 1),  
  //             init_Tx_dr_L.at<double>(0, 2), init_Tx_dr_L.at<double>(0, 3),
  //             init_Tx_dr_L.at<double>(1, 0), init_Tx_dr_L.at<double>(1, 1),
  //             init_Tx_dr_L.at<double>(1, 2), init_Tx_dr_L.at<double>(1, 3),
  //             init_Tx_dr_L.at<double>(2, 0), init_Tx_dr_L.at<double>(2, 1), 
  //             init_Tx_dr_L.at<double>(2, 2), init_Tx_dr_L.at<double>(2, 3),
  //             0.0, 0.0, 0.0, 1.0;

  rgb_canny_threshold_ = fSettings["Canny.gray_threshold"];
  rgb_edge_minLen_ = fSettings["Canny.len_threshold"];
  voxel_size_ = fSettings["Voxel.size"];
  eigen_ratio_ = fSettings["Voxel.eigen_ratio"];
  down_sample_size_ = fSettings["Voxel.down_sample_size"];
  plane_size_threshold_ = fSettings["Plane.min_points_size"];
  plane_max_size_ = fSettings["Plane.max_size"];
  ransac_dis_threshold_ = fSettings["Ransac.dis_threshold"];
  min_line_dis_threshold_ = fSettings["Edge.min_dis_threshold"];
  max_line_dis_threshold_ = fSettings["Edge.max_dis_threshold"];
  theta_min_ = fSettings["Plane.normal_theta_min"];
  theta_max_ = fSettings["Plane.normal_theta_max"];
  theta_min_ = cos(DEG2RAD(theta_min_));
  theta_max_ = cos(DEG2RAD(theta_max_));
  match_type_ = fSettings["match_type"];
  fSettings["update_camera_extrinsic"] >> update_camera_extrinsic_;
    fSettings["use_ground_lines"] >> use_ground_lines_;
  fSettings["save_lidar_extrinsic_name"] >> save_lidar_extrinsic_name_;
  fSettings["img_file_extension"] >> img_file_extension_;

  if(debugMode_) {
    std::cout << "ImageFilesPath: " << image_path_ << std::endl;
    std::cout << "LiDARFilesPath: " << pcd_path_ << std::endl;
    std::cout << "CamsIntrinsicPath: " << cameras_intrinsic_path_ << std::endl;
    std::cout << "CamsExtrinsicPath: " << cameras_extrinsic_path_ << std::endl;
    std::cout << "SceneNum: " << scene_num_ << std::endl;
    std::cout << "init_extrinsic_file: " << init_extrinsic_file_ << std::endl;
    std::cout << "result_path: " << result_path_ << std::endl;
    std::cout << "Canny.gray_threshold: " << rgb_canny_threshold_ << std::endl;
    std::cout << "Canny.len_threshold: " << rgb_edge_minLen_ << std::endl;
    std::cout << "Voxel.size: " << voxel_size_ << std::endl;
    std::cout << "Voxel.eigen_ratio: " << eigen_ratio_ << std::endl;
    std::cout << "Voxel.down_sample_size: " << down_sample_size_ << std::endl;
    std::cout << "Plane.min_points_size: " << plane_size_threshold_ << std::endl;
    std::cout << "Plane.max_size: " << plane_max_size_ << std::endl;
    std::cout << "Ransac.dis_threshold: " << ransac_dis_threshold_ << std::endl;
    std::cout << "Edge.min_dis_threshold: " << min_line_dis_threshold_ << std::endl;
    std::cout << "Edge.max_dis_threshold: " << max_line_dis_threshold_ << std::endl;
    std::cout << "Plane.normal_theta_min: " << theta_min_ << std::endl;
    std::cout << "Plane.normal_theta_max: " << theta_max_ << std::endl;
    std::cout << "match_type: " << match_type_ << std::endl;
    std::cout << "update_camera_extrinsic: " << update_camera_extrinsic_ << std::endl;
    std::cout << "save_lidar_extrinsic_name: " << save_lidar_extrinsic_name_ << std::endl;
    std::cout << "img_file_extension: " << img_file_extension_ << std::endl;
  }
  return true;
}


void Calibration::readExtriFromYamlFile(const std::string& file, Eigen::Matrix4d& T_dr_L) {
  cv::FileStorage settings(file, cv::FileStorage::READ);
  if (!settings.isOpened()) {
    std::cerr << "# ERROR: Failed to open settings file at: " << file << std::endl;
    return;
  }

  cv::FileNode n = settings["transform"];
  double qx = static_cast<double>(n["qx"]);
  double qy = static_cast<double>(n["qy"]);
  double qz = static_cast<double>(n["qz"]);
  double qw = static_cast<double>(n["qw"]);
  double tx = static_cast<double>(n["tx"]);
  double ty = static_cast<double>(n["ty"]);
  double tz = static_cast<double>(n["tz"]);

  Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz);
  // std::cout << "qua: " << q.coeffs() << std::endl;

  Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Vector3d t = Eigen::Vector3d(tx, ty, tz);
  T_dr_L = T_dr_L.Identity();
  T_dr_L.block<3, 3>(0, 0) = R;
  T_dr_L.block<3, 1>(0, 3) = t;

  if(debugMode_) {
    std::cout << "Tx_dr_L: " << std::endl
              << "q_dr_l: " << Eigen::Quaterniond(R).coeffs()
              << "t_dr_l: " << T_dr_L.topRightCorner(3, 1) << std::endl;
  }
}



void Calibration::loadCamConfigPath_old() {
  std::vector<std::string> cam_intrinsic_vec, cam_extrinsic_vec;
  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    std::string cam_intrinsic_path;
    if(cams[cam_index].cam_name_ == "front_near") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "front_view/" + cams[cam_index].cam_name_ + "_intrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "avm_front") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "around_view/front_intrinsic.yaml";
    }else if(cams[cam_index].cam_name_ == "avm_left") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "around_view/left_intrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "avm_rear") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "around_view/rear_intrinsic.yaml";
    } 
    else if(cams[cam_index].cam_name_ == "avm_right") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "around_view/right_intrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "left_front" ||
            cams[cam_index].cam_name_ == "left_rear" ||
            cams[cam_index].cam_name_ == "right_front" ||
            cams[cam_index].cam_name_ == "right_rear") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "side_view/" + cams[cam_index].cam_name_ + "_intrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "rear_medium") {
      cam_intrinsic_path = cameras_intrinsic_path_ + "/" + "rear_view/" + cams[cam_index].cam_name_ + "_intrinsic.yaml";
    }
    cam_intrinsic_vec.emplace_back(cam_intrinsic_path);
  }

  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    std::string cam_extrinsic_path;
    if(cams[cam_index].cam_name_ == "front_near") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "front_view/" + cams[cam_index].cam_name_ + "_extrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "avm_front") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "around_view/front_extrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "avm_left") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "around_view/left_extrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "avm_rear") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "around_view/rear_extrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "avm_right") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "around_view/right_extrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "left_front" ||
            cams[cam_index].cam_name_ == "left_rear" ||
            cams[cam_index].cam_name_ == "right_front" ||
            cams[cam_index].cam_name_ == "right_rear") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "side_view/" + cams[cam_index].cam_name_ + "_extrinsic.yaml";
    }
    else if(cams[cam_index].cam_name_ == "rear_medium") {
      cam_extrinsic_path = cameras_extrinsic_path_ + "/" + "rear_view/" + cams[cam_index].cam_name_ + "_extrinsic.yaml";
    }
    cam_extrinsic_vec.emplace_back(cam_extrinsic_path);
  }

  assert(cam_intrinsic_vec.size() == cam_extrinsic_vec.size());
  assert(cam_intrinsic_vec.size() == cams.size());

  if(debugMode_) {
    for(int cam_index = 0; cam_index < cam_intrinsic_vec.size(); ++cam_index) {
      std::cout << "in: " << cam_intrinsic_vec[cam_index] << std::endl
                << "ex: " << cam_extrinsic_vec[cam_index] << std::endl;
    }
  }

  // load camera's intrinsic and extrinsic, get them from lam_data directly
  if(!loadCameraConfigFromYmlFile(cam_intrinsic_vec, cam_extrinsic_vec, cams)) {
    return;
  }
}

bool Calibration::loadCameraConfigFromYmlFile(const std::vector<std::string>& cam_intrinsic_vec, 
                        const std::vector<std::string>& cam_extrinsic_vec,
                        std::vector<Camera>& cams) {
  // cams.resize(cam_intrinsic_vec.size());
  for(size_t i = 0; i < cam_intrinsic_vec.size(); ++i) {
    readCamInFromYmlFile(cam_intrinsic_vec[i], cams[i]);
    readCamExFromYmlFile(cam_extrinsic_vec[i], cams[i]);

    if(debugMode_) {
      std::cout << "Camera [" << cams[i].cam_name_ << "] Configuration" << std::endl;
      std::cout << "image_width: " << cams[i].width_ << std::endl
               << "image_height: " << cams[i].height_ << std::endl;
      std::cout << "Camera Matrix: " << std::endl << cams[i].camera_matrix_ << std::endl;
      std::cout << "Distortion Coeffs: " << std::endl << cams[i].dist_coeffs_ << std::endl;
      std::cout << "Tx_dr_C: " << std::endl << cams[i].Tx_dr_C_ << std::endl;
    }
  }

  return true;
}

void Calibration::readCamInFromYmlFile(const std::string& yml_file, Camera& cam) {
   cv::FileStorage fCamInSet(yml_file, cv::FileStorage::READ);
    if(!fCamInSet.isOpened()) {
      std::cerr << "Failed to open cams intrinsic file at " << yml_file << std::endl;
      exit(-1);
    }

    cam.width_ = fCamInSet["image_width"];
    cam.height_ = fCamInSet["image_height"];

    cv::FileNode n = fCamInSet["projection_parameters"];
    cam.fx_ = static_cast<double>(n["fx"]);
    cam.cx_ = static_cast<double>(n["cx"]);
    cam.fy_ = static_cast<double>(n["fy"]);
    cam.cy_ = static_cast<double>(n["cy"]);

    n = fCamInSet["distortion_parameters"];
    cam.k1_ = static_cast<double>(n["k1"]);
    cam.k2_ = static_cast<double>(n["k2"]);
    cam.k3_ = static_cast<double>(n["k3"]);
    cam.k4_ = static_cast<double>(n["k4"]);

    cam.camera_matrix_ = (cv::Mat_<double>(3, 3) << cam.fx_, 0., cam.cx_,
                              0., cam.fy_, cam.cy_,
                              0., 0., 1.);
    cam.dist_coeffs_ = (cv::Mat_<double>(4, 1) << cam.k1_, cam.k2_, cam.k3_, cam.k4_);                                               
}

void Calibration::readCamExFromYmlFile(const std::string& yml_file, Camera& cam) {
  cv::FileStorage fCamExSet(yml_file, cv::FileStorage::READ);
    if(!fCamExSet.isOpened()) {
      std::cerr << "Failed to open cams extrinsic file at " << yml_file  << std::endl;
      exit(-1);
    }

    cv::Mat fixed_ext;
    fCamExSet["Tdc"] >> fixed_ext;
    cam.Tx_dr_C_ << static_cast<double>(fixed_ext.at<float>(0, 0)), static_cast<double>(fixed_ext.at<float>(0, 1)), 
                        static_cast<double>(fixed_ext.at<float>(0, 2)), static_cast<double>(fixed_ext.at<float>(0, 3)),
                        static_cast<double>(fixed_ext.at<float>(1, 0)), static_cast<double>(fixed_ext.at<float>(1, 1)),
                        static_cast<double>(fixed_ext.at<float>(1, 2)), static_cast<double>(fixed_ext.at<float>(1, 3)),
                        static_cast<double>(fixed_ext.at<float>(2, 0)), static_cast<double>(fixed_ext.at<float>(2, 1)), 
                        static_cast<double>(fixed_ext.at<float>(2, 2)), static_cast<double>(fixed_ext.at<float>(2, 3)),
                        0.0, 0.0, 0.0, 1.0;
    cam.R_dr_C_ << cam.Tx_dr_C_.topLeftCorner(3, 3);
    cam.t_dr_C_ << cam.Tx_dr_C_.topRightCorner(3, 1);

    cam.Tx_C_L_ = cam.Tx_dr_C_.inverse() * lidar.Tx_dr_L_;
    cam.R_C_L_ = cam.Tx_C_L_.topLeftCorner(3, 3);
    cam.t_C_L_ = cam.Tx_C_L_.topRightCorner(3, 1);  
}



void Calibration::loadCamConfigPath_new() {
  std::vector<std::string> cam_intrinsic_vec, cam_extrinsic_vec;
  std::string cam_intrinsic_path, cam_extrinsic_path;
  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    cam_intrinsic_path = cameras_intrinsic_path_ + "/" + cams[cam_index].cam_name_ + "_param.xml";
    cam_extrinsic_path = cameras_extrinsic_path_ + "/" + cams[cam_index].cam_name_ + "_transform.pb.txt";
    cam_intrinsic_vec.emplace_back(cam_intrinsic_path);
    cam_extrinsic_vec.emplace_back(cam_extrinsic_path);
  }

  assert(cam_intrinsic_vec.size() == cam_extrinsic_vec.size());
  assert(cam_intrinsic_vec.size() == cams.size());

  if(debugMode_) {
    for(int cam_index = 0; cam_index < cam_intrinsic_vec.size(); ++cam_index) {
      std::cout << "in: " << cam_intrinsic_vec[cam_index] << std::endl
                << "ex: " << cam_extrinsic_vec[cam_index] << std::endl;
    }
  }

  // load camera's intrinsic and extrinsic, get them from lam_data directly
  if(!loadCameraConfig(cam_intrinsic_vec, cam_extrinsic_vec, cams)) {
    return;
  }
}

bool Calibration::loadCameraConfig(const std::vector<std::string>& cam_intrinsic_vec, 
                        const std::vector<std::string>& cam_extrinsic_vec,
                        std::vector<Camera>& cams) {
  for(size_t i = 0; i < cam_intrinsic_vec.size(); ++i) {
    readCamInFromXmlFile(cam_intrinsic_vec[i], cams[i]);
    readCamExFromPbFile(cam_extrinsic_vec[i], cams[i]);

    if(debugMode_) {
      std::cout << "Camera [" << cams[i].cam_name_ << "] Configuration" << std::endl;
      std::cout << "image_width: " << cams[i].width_ << std::endl
               << "image_height: " << cams[i].height_ << std::endl;
      std::cout << "Camera Matrix: " << std::endl << cams[i].camera_matrix_ << std::endl;
      std::cout << "Distortion Coeffs: " << std::endl << cams[i].dist_coeffs_ << std::endl;
      std::cout << "Tx_dr_C: " << std::endl << cams[i].Tx_dr_C_ << std::endl;
    }
	}
  return true; 
}

void Calibration::readCamInFromXmlFile(const std::string& xml_file, Camera& cam) {
  rapidxml::file<> fdoc(xml_file.c_str());
	rapidxml::xml_document<> doc;
	doc.parse<0>(fdoc.data());
	// rapidxml::xml_node<> *param = doc.first_node("param");

  for (rapidxml::xml_node<> *param = doc.first_node("param")->first_node(); param; param = param->next_sibling()) {
		if (param->first_node() != NULL) {
			// std::cout << param->name()/* << " : " << param->value()*/ << std::endl;

      if(std::string(param->name()) == "cx") {
        cam.cx_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "cy") {
        cam.cy_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "fx") {
        cam.fx_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "fy") {
        cam.fy_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "image_height") {
        cam.height_ = std::stoi(param->value());
      }
      else if(std::string(param->name()) == "image_width") {
        cam.width_ = std::stoi(param->value());
      }
      if(std::string(param->name()) == "k1") {
        cam.k1_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "k2") {
        cam.k2_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "k3") {
        cam.k3_ = std::stod(param->value());
      }
      else if(std::string(param->name()) == "k4") {
        cam.k4_ = std::stod(param->value());
      }
      // todo
      else if(std::string(param->name()) == "model_type") {
        std::string model_type = std::string(param->value());
        // std::cout << " model_type: " << model_type << std::endl;
      }
      else if(std::string(param->name()) == "p1") {
        double p1 = std::stod(param->value());
        // std::cout << " p1: " << p1 << std::endl;
      }
      else if(std::string(param->name()) == "p2") {
        double p2 = std::stod(param->value());
        // std::cout << " p2: " << p2 << std::endl;
      }
		}
		else
			std::cout << "name: " << param->name() << "has no value" << std::endl;
	}

  cam.camera_matrix_ = (cv::Mat_<double>(3, 3) << cam.fx_, 0., cam.cx_,
                        0., cam.fy_, cam.cy_,
                        0., 0., 1.);
  cam.dist_coeffs_ = (cv::Mat_<double>(4, 1) << cam.k1_, cam.k2_, cam.k3_, cam.k4_);    
}


void Calibration::readCamExFromPbFile(const std::string& pb_file, Camera& cam) {
  tutorial::SensorExtrinsic T_dr_C_pb;
  if(readProtoFromTextFile(pb_file, &T_dr_C_pb)) {
    Eigen::Vector3d trans_dr_c;
    Eigen::Quaterniond qua_dr_c;
    tutorial::Translation trans = T_dr_C_pb.translation()[0];
    trans_dr_c = Eigen::Vector3d(trans.x(), trans.y(), trans.z());

    tutorial::Rotation rot = T_dr_C_pb.rotation()[0];
    qua_dr_c = Eigen::Quaterniond(rot.w(), rot.x(), rot.y(), rot.z());                

    cam.R_dr_C_ = qua_dr_c.toRotationMatrix();
    cam.t_dr_C_ = trans_dr_c;
    cam.Tx_dr_C_ = Eigen::Matrix4d::Identity();
    cam.Tx_dr_C_.block<3, 3>(0, 0) = cam.R_dr_C_;
    cam.Tx_dr_C_.block<3, 1>(0, 3) = cam.t_dr_C_;

    // cam.Tx_C_L_ = cam.Tx_dr_C_.inverse() * lidar.Tx_dr_L_;
    // cam.R_C_L_ = cam.Tx_C_L_.topLeftCorner(3, 3);
    // cam.t_C_L_ = cam.Tx_C_L_.topRightCorner(3, 1);
  }
  else
    return;
}

void Calibration::readSensorExFromPbFile(const std::string& pb_file, Eigen::Matrix4d& extrinsic) {
  extrinsic = Eigen::Matrix4d::Identity();
  tutorial::SensorExtrinsic extrinsic_pb;
  if(readProtoFromTextFile(pb_file, &extrinsic_pb)) {
    Eigen::Vector3d trans;
    Eigen::Quaterniond qua;
    tutorial::Translation trans_pb = extrinsic_pb.translation()[0];
    trans = Eigen::Vector3d(trans_pb.x(), trans_pb.y(), trans_pb.z());

    tutorial::Rotation qua_pb = extrinsic_pb.rotation()[0];
    qua = Eigen::Quaterniond(qua_pb.w(), qua_pb.x(), qua_pb.y(), qua_pb.z());                

    extrinsic.block<3, 3>(0, 0) = qua.toRotationMatrix();
    extrinsic.block<3, 1>(0, 3) = trans;
  }
  else
    return;
}


bool Calibration::readProtoFromTextFile(const std::string& file, google::protobuf::Message* proto) {
  int fd = open(file.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "ProtoIO: failed to open file(RD): " << file << std::endl;
    return false;
  }

  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
  bool flag = google::protobuf::TextFormat::Parse(input, proto);

  delete input;
  close(fd);

  return flag;
}

// Detect edge by canny, and filter by edge length
void Calibration::edgeDetector(
    const int &height, const int &width, 
    const int &canny_threshold, const int &edge_threshold,
    const cv::Mat &src_img, cv::Mat &edge_img,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud) {
  int gaussian_size = 5;
  cv::GaussianBlur(src_img, src_img, cv::Size(gaussian_size, gaussian_size), 0,
                   0);
  cv::Mat canny_result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Canny(src_img, canny_result, canny_threshold, canny_threshold * 3, 3,
            true);



  // //find straight line
  // cv::Mat line_result = cv::Mat::zeros(height_, width_, CV_8UC1);
  // std::vector<cv::Vec4i> lines;
  // // 检测直线，最小投票为90，线条不短于50，间隙不小于10
  // cv::HoughLinesP(canny_result,lines,1,CV_PI/180,90,40,8);
  // std::vector<cv::Vec4i>::const_iterator it=lines.begin();
  // while(it!=lines.end())
  // {
  //     cv::Point pt1((*it)[0],(*it)[1]);
  //     cv::Point pt2((*it)[2],(*it)[3]);
  //     cv::line(line_result,pt1,pt2,cv::Scalar(255),1); //  线条宽度设置为2
  //     ++it;
  // }

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  edge_img = cv::Mat::zeros(height, width, CV_8UC1);

  edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < contours.size(); i++) {
    if (contours[i].size() > edge_threshold) {
      cv::Mat debug_img = cv::Mat::zeros(height, width, CV_8UC1);
      for (size_t j = 0; j < contours[i].size(); j++) {
        pcl::PointXYZ p;
        p.x = contours[i][j].x;
        p.y = -contours[i][j].y;
        p.z = 0;
        edge_img.at<uchar>(-p.y, p.x) = 255;
      }
    }
  }
  for (int x = 0; x < edge_img.cols; x++) {
    for (int y = 0; y < edge_img.rows; y++) {
      if (edge_img.at<uchar>(y, x) == 255) {
        pcl::PointXYZ p;
        p.x = x;
        p.y = -y;
        p.z = 0;
        edge_cloud->points.push_back(p);
      }
    }
  }
  edge_cloud->width = edge_cloud->points.size();
  edge_cloud->height = 1;
  // cv::imshow("canny result", canny_result);
  // cv::imshow("edge result", edge_img);
  // // cv::imshow("line result", line_result);
  // cv::waitKey(0);
}

void Calibration::projection(
    const Camera& cam,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
    ProjectionType projection_type, const bool is_fill_img,
    cv::Mat &projection_img) {
  std::vector<cv::Point3f> pts_3d;
  std::vector<float> intensity_list;

  // filter out-view pcd;
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
      new pcl::PointCloud<pcl::PointXYZI>); 
  // Eigen::Matrix4d Tx_C_L = cam.Tx_C_L_;
  Eigen::Matrix4d Tx_C_L = cam.Tx_dr_C_.inverse() * lidar.Tx_dr_L_;
  
  pcl::transformPointCloud(*lidar_cloud, *transformed_pcd, Tx_C_L);

  for(auto p: *transformed_pcd) {
    if(p.z < 0) continue;
    tmp_pcd->push_back(p);
  }
  Eigen::Matrix4d Tx_L_C = Tx_C_L.inverse();;
  pcl::transformPointCloud(*tmp_pcd, *tmp_pcd, Tx_L_C);

  // std::cout << "tmp_pcd size: " << tmp_pcd->size() << std::endl;

  // project 3d-points into image view
  Vector6d extrinsic_params;
  Eigen::Matrix3d R_C_L = Tx_C_L.topLeftCorner(3, 3);
  Eigen::Vector3d t_C_L = Tx_C_L.topRightCorner(3, 1);
  Eigen::Vector3d euler = R_C_L.eulerAngles(2, 1, 0);

  extrinsic_params[0] = euler[0];
  extrinsic_params[1] = euler[1];
  extrinsic_params[2] = euler[2];
  extrinsic_params[3] = t_C_L[0];
  extrinsic_params[4] = t_C_L[1];
  extrinsic_params[5] = t_C_L[2];

  Eigen::AngleAxisd rotation_vector3;
  rotation_vector3 =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  // std::cout << "wd check rotation_vector3: " 
  //           << rotation_vector3.angle() * rotation_vector3.axis() << std::endl;    

  cv::Mat r_vec =
      (cv::Mat_<double>(3, 1)
           << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
  // std:cout << "wd check r_vec: " << r_vec << std::endl; 
  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                   extrinsic_params[4], extrinsic_params[5]);

  double max_intensity = 0;
  for (size_t i = 0; i < tmp_pcd->size(); i++) {
    pcl::PointXYZI point_3d = tmp_pcd->points[i];
    
    float depth =
        sqrt(pow(point_3d.x, 2) + pow(point_3d.y, 2) + pow(point_3d.z, 2));
    if (depth > min_depth_ && depth < max_depth_) {
      pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
      intensity_list.emplace_back(tmp_pcd->points[i].intensity);
    }

    if(tmp_pcd->points[i].intensity > max_intensity) 
      max_intensity = tmp_pcd->points[i].intensity;
  }

  if(max_intensity == 0)
    projection_type = DEPTH;
  else
    projection_type = INTENSITY;

  // std::sort(pts_3d.begin(), pts_3d.end(), 
  //   [](const cv::Point3f& a, const cv::Point3f& b) { 
  //     return (sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2)) < sqrt(pow(b.x, 2) + pow(b.y, 2) + pow(b.z, 2))); });
  cv::Mat camera_matrix = cam.camera_matrix_;
  cv::Mat distortion_coeff = cam.dist_coeffs_;   

  // std::vector<std::vector<std::vector<cv::Point3f>>> img_pts_container;
  // for (int y = 0; y < cam.height_; ++y) {
  //   std::vector<std::vector<cv::Point3f>> row_pts_container;
  //   for (int x = 0; x < cam.width_; ++x) {
  //     std::vector<cv::Point3f> col_pts_container;
  //     row_pts_container.emplace_back(col_pts_container);
  //   }
  //   img_pts_container.emplace_back(row_pts_container);
  // }

  // if(debugMode_) {
  //   std::cout << "check projection camera_matrix: " << camera_matrix << std::endl
  //             << "check projection distortion_coeff: " << distortion_coeff << std::endl;
  // }  

  // project 3d-points into image view
  std::vector<cv::Point2f> pts_2d;
  // cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
  //                   pts_2d);                   
  cv::fisheye::projectPoints(pts_3d, pts_2d, r_vec, t_vec, camera_matrix, distortion_coeff);
  // std::cout << "wd check pts_3d size: " << pts_3d.size() << std::endl
  //           << "wd check pts_2d size: " << pts_2d.size() << std::endl
  //           << "wd check height: " << cam.height_ << " " << "width: " << cam.width_ << std::endl;
  cv::Mat image_project = cv::Mat::zeros(cam.height_, cam.width_, CV_16UC1);
  cv::Mat rgb_image_project = cv::Mat::zeros(cam.height_, cam.width_, CV_8UC3);
  for (size_t i = 0; i < pts_2d.size(); ++i) {
    cv::Point2f point_2d = pts_2d[i];
    if (point_2d.x <= 0 || point_2d.x >= cam.width_ || point_2d.y <= 0 ||
        point_2d.y >= cam.height_) {
      continue;
    } else {
      // if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() != 0) {
      //   img_pts_container[pts_2d[i].y][pts_2d[i].x].emplace_back(pts_3d[i]);
      //   continue;
      // }else {
      //   img_pts_container[pts_2d[i].y][pts_2d[i].x].emplace_back(pts_3d[i]);
      // }
      // test depth and intensity both
      if (projection_type == DEPTH) {
        float depth = sqrt(pow(pts_3d[i].x, 2) + pow(pts_3d[i].y, 2) +
                           pow(pts_3d[i].z, 2));
        float intensity = intensity_list[i];
        float depth_weight = 1;
        float grey = depth_weight * depth / max_depth_ * 65535 +
                     (1 - depth_weight) * intensity / 150 * 65535;
        if (image_project.at<ushort>(point_2d.y, point_2d.x) == 0) {
          image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] =
              depth / max_depth_ * 255;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] =
              intensity / 150 * 255;
        } else if (depth < image_project.at<ushort>(point_2d.y, point_2d.x)) {
          image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] =
              depth / max_depth_ * 255;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] =
              intensity / 150 * 255;
        }

      } else {
        float intensity = intensity_list[i];
        if (intensity > 100) {
          intensity = 65535;
        } else {
          intensity = (intensity / 150.0) * 65535;
        }
        image_project.at<ushort>(point_2d.y, point_2d.x) = intensity;
      }
    }
  }
  cv::Mat grey_image_projection;
  cv::cvtColor(rgb_image_project, grey_image_projection, cv::COLOR_BGR2GRAY);

  image_project.convertTo(image_project, CV_8UC1, 1 / 256.0);
  if (is_fill_img) {
    for (int i = 0; i < 5; i++) {
      image_project = fillImg(image_project, UP, LEFT);
    }
  }
  if (is_fill_img) {
    for (int i = 0; i < 5; i++) {
      grey_image_projection = fillImg(grey_image_projection, UP, LEFT);
    }
  }
  projection_img = image_project.clone();
}

// 填补雷达深度图像
cv::Mat Calibration::fillImg(const cv::Mat &input_img,
                             const Direction first_direct,
                             const Direction second_direct) {
  cv::Mat fill_img = input_img.clone();
  for (int y = 2; y < input_img.rows - 2; y++) {
    for (int x = 2; x < input_img.cols - 2; x++) {
      if (input_img.at<uchar>(y, x) == 0) {
        if (input_img.at<uchar>(y - 1, x) != 0) {
          fill_img.at<uchar>(y, x) = input_img.at<uchar>(y - 1, x);
        } else {
          if ((input_img.at<uchar>(y, x - 1)) != 0) {
            fill_img.at<uchar>(y, x) = input_img.at<uchar>(y, x - 1);
          }
        }
      } else {
        int left_depth = input_img.at<uchar>(y, x - 1);
        int right_depth = input_img.at<uchar>(y, x + 1);
        int up_depth = input_img.at<uchar>(y + 1, x);
        int down_depth = input_img.at<uchar>(y - 1, x);
        int current_depth = input_img.at<uchar>(y, x);
        if ((current_depth - left_depth) > 5 &&
            (current_depth - right_depth) > 5 && left_depth != 0 &&
            right_depth != 0) {
          fill_img.at<uchar>(y, x) = (left_depth + right_depth) / 2;
        } else if ((current_depth - up_depth) > 5 &&
                   (current_depth - down_depth) > 5 && up_depth != 0 &&
                   down_depth != 0) {
          fill_img.at<uchar>(y, x) = (up_depth + right_depth) / 2;
        }
      }
    }
  }
  return fill_img;
}


cv::Mat Calibration::getConnectImg(
    const Camera& cam,
    const int dis_threshold,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud) {
  // cv::Mat connect_img = cv::Mat::zeros(cam.height_, cam.width_, CV_8UC3);
  cv::Mat connect_img = cv::Mat(cam.height_, cam.width_, CV_8UC3, cv::Scalar::all(255));
  
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree->setInputCloud(rgb_edge_cloud);
  tree_cloud = rgb_edge_cloud;
  for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
    cv::Point2d p2(depth_edge_cloud->points[i].x,
                   -depth_edge_cloud->points[i].y);
    if (checkFov(cam, p2)) {
      pcl::PointXYZ p = depth_edge_cloud->points[i];
      search_cloud->points.push_back(p);
    }
  }

  int line_count = 0;
  // 指定近邻个数
  int K = 1;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < search_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = search_cloud->points[i];
    if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
            pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
        if (distance < dis_threshold) {
          cv::Scalar color = cv::Scalar(0, 255, 0);
          line_count++;
          if ((line_count % 3) == 0) {
            cv::line(connect_img, cv::Point(search_cloud->points[i].x,
                                            -search_cloud->points[i].y),
                     cv::Point(tree_cloud->points[pointIdxNKNSearch[j]].x,
                               -tree_cloud->points[pointIdxNKNSearch[j]].y),
                     color, 1);
          }
        }
      }
    }
  }

  for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[0] = 255;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[2] = 0;
  }
  for (size_t i = 0; i < search_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                              search_cloud->points[i].x)[0] = 0;
    connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                              search_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                              search_cloud->points[i].x)[2] = 255;
  }
  int expand_size = 2;
  cv::Mat expand_edge_img;
  expand_edge_img = connect_img.clone();
  for (int x = expand_size; x < connect_img.cols - expand_size; x++) {
    for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
      if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
          }
        }
      } else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
          }
        }
      }
    }
  }
  return connect_img;
}


bool Calibration::checkFov(const Camera& cam, const cv::Point2d &p) {
  if (p.x > 0 && p.x < cam.width_ && p.y > 0 && p.y < cam.height_) {
    return true;
  } else {
    return false;
  }
}

void Calibration::initVoxel(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map) {
  ROS_INFO_STREAM("Building Voxel");
  // for voxel test
  srand((unsigned)time(NULL));
  pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
  // std::cout << "wd check input_cloud: " << input_cloud->size() << std::endl;
 
  for (size_t i = 0; i < input_cloud->size(); i++) {
    const pcl::PointXYZI &p_c = input_cloud->points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->cloud->push_back(p_c);
      pcl::PointXYZRGB p_rgb;
      p_rgb.x = p_c.x;
      p_rgb.y = p_c.y;
      p_rgb.z = p_c.z;
      p_rgb.r = voxel_map[position]->voxel_color(0);
      p_rgb.g = voxel_map[position]->voxel_color(0);
      p_rgb.b = voxel_map[position]->voxel_color(0);
      test_cloud.push_back(p_rgb);
    } else {
      Voxel *voxel = new Voxel(voxel_size);
      voxel_map[position] = voxel;
      voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
      voxel_map[position]->voxel_origin[1] = position.y * voxel_size;
      voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
      voxel_map[position]->cloud->push_back(p_c);

      // notice!!!
      int r = rand() % 255;
      int g = rand() % 255;
      int b = rand() % 255;
      voxel_map[position]->voxel_color << r, g, b;
    }
  }

  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->cloud->size() > 20) {
      // down_sampling_voxel(*(iter->second->cloud), 0.03);
      down_sampling_voxel(*(iter->second->cloud), 0.01);
    }
  }
}


void Calibration::cut_voxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                double voxel_size, float eigen_ratio) {
  ROS_INFO_STREAM("cutting Voxel");

  for(size_t i = 0; i < input_cloud->size(); i++) {
    const pcl::PointXYZI &p_c = input_cloud->points[i];
    float loc_xyz[3];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pvec_orig[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], 
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end()) {
      iter->second->all_points.push_back(pvec_orig);
      iter->second->vec_orig.push_back(pvec_orig);        
      iter->second->sig_orig.push(pvec_orig);
    }
    else {
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(eigen_ratio);
      ot->all_points.push_back(pvec_orig);
      ot->vec_orig.push_back(pvec_orig);
      ot->sig_orig.push(pvec_orig);
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

void Calibration::merge_planes(std::vector<Plane>& planes){
    if(planes.size()<2)
        return;

    std::sort(planes.begin(),planes.end(),[](const Plane& p1, const Plane& p2){
        return p1.cloud.size()>p2.cloud.size();
    });
    assert(planes[0].cloud.size()>planes.back().cloud.size());

    for(int i=0;i<int(planes.size())-1;i++){
        Plane & plane1 = planes[i];
        for(int j=i+1;j<int(planes.size());j++){
            Plane& plane2 = planes[j];
            float angle = plane1.normal.dot(plane2.normal);
            if(fabs(angle)<std::cos(DEG2RAD(30)))
                continue;
            auto dist_vec = (plane1.p_center.getVector3fMap() - plane2.p_center.getVector3fMap()).cast<double>();
            float max_dist = std::max(dist_vec.dot(plane1.normal),dist_vec.dot(plane2.normal));
            if(max_dist>0.2)
                continue;
            plane1.p_center.getVector3fMap() = (plane1.p_center.getVector3fMap() * plane1.cloud.size() +
                                                plane2.p_center.getVector3fMap() * plane2.cloud.size())/
                                                        (plane1.cloud.size() + plane2.cloud.size());
            plane1.cloud += plane2.cloud;

            Eigen::Vector4f centroid;
            centroid<<plane1.p_center.getVector3fMap(),1;
            Eigen::Matrix3f cov;
            pcl::computeCovarianceMatrix(plane1.cloud,centroid,cov);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
            assert(eig.eigenvalues().x()<eig.eigenvalues().y() &&
                           eig.eigenvalues().x()<eig.eigenvalues().z());
            plane1.normal = eig.eigenvectors().col(0).cast<double>();
            planes[j] = planes.back();
            planes.pop_back();
            j--;
        }
    }
}

void Calibration::LiDAREdgeExtraction(
    const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map,
    const float ransac_dis_thre, const int plane_size_threshold,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d/*,
    std::vector<int>& plane_line_number*/) {
  ROS_INFO_STREAM("Extracting Lidar Edge");
  // ros::Rate loop(5000);
  lidar_line_cloud_3d =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  int cnt = 0;
    pcl::PointCloud<pcl::PointXYZI> color_planner_cloud;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    // std::cout << "cnt: " << cnt << std::endl;
    if (iter->second->cloud->size() > 50) {
        std::string output_pcd_dir = result_path_ +"/pcds/"+std::to_string(cnt);
        boost::filesystem::create_directories(output_pcd_dir);


        pcl::PointCloud<pcl::PointXYZI> voxel_lines;
      std::vector<Plane> plane_list;
      // 创建一个体素滤波器
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(
          new pcl::PointCloud<pcl::PointXYZI>);
      pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
      //创建一个模型参数对象，用于记录结果
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      // inliers表示误差能容忍的点，记录点云序号
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      //创建一个分割器
      pcl::SACSegmentation<pcl::PointXYZI> seg;
      // Optional,设置结果平面展示的点是分割掉的点还是分割剩下的点
      seg.setOptimizeCoefficients(true);
      // Mandatory-设置目标几何形状
      seg.setModelType(pcl::SACMODEL_PLANE);
      //分割方法：随机采样法
      seg.setMethodType(pcl::SAC_RANSAC);
      //设置误差容忍范围，也就是阈值

      seg.setDistanceThreshold(ransac_dis_thre);


      int plane_index = 0;
      while (cloud_filter->points.size() > 10) {
        pcl::PointCloud<pcl::PointXYZI> planner_cloud;
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        //输入点云
        seg.setInputCloud(cloud_filter);
        seg.setMaxIterations(500);
        //分割点云
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
          ROS_INFO_STREAM(
              "Could not estimate a planner model for the given dataset");
          break;
        }
        extract.setIndices(inliers);
        extract.setInputCloud(cloud_filter);
        extract.filter(planner_cloud);

        if (planner_cloud.size() > plane_size_threshold) {
          pcl::PointXYZ p_center(0, 0, 0);
          for (size_t i = 0; i < planner_cloud.points.size(); i++) {
            pcl::PointXYZRGB p;
            p.x = planner_cloud.points[i].x;
            p.y = planner_cloud.points[i].y;
            p.z = planner_cloud.points[i].z;
            p_center.x += p.x;
            p_center.y += p.y;
            p_center.z += p.z;
          }



          p_center.x = p_center.x / planner_cloud.size();
          p_center.y = p_center.y / planner_cloud.size();
          p_center.z = p_center.z / planner_cloud.size();
          Plane single_plane;
          single_plane.cloud = planner_cloud;
          single_plane.p_center = p_center;
          single_plane.normal << coefficients->values[0],
              coefficients->values[1], coefficients->values[2];
          single_plane.index = plane_index;
          plane_list.emplace_back(single_plane);
          plane_index++;


            pcl::PointCloud<pcl::PointXYZI> voxel_planes_intensity = planner_cloud;
            voxel_planes_intensity.getMatrixXfMap().row(4).array() = cnt;
            color_planner_cloud += voxel_planes_intensity;
        }
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZI> cloud_f;
        extract.filter(cloud_f);
        *cloud_filter = cloud_f;
      }

      //merge planes
        merge_planes(plane_list);
      for(int m=0;m<plane_list.size();m++){
          pcl::RGB color;
          color.r = static_cast<unsigned int>(rand() % 256);
          color.g = static_cast<unsigned int>(rand() % 256);
          color.b = static_cast<unsigned int>(rand() % 256);
            pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
            pcl::copyPointCloud(plane_list[m].cloud,color_cloud);
            for(int l=0;l<color_cloud.size();l++)
                color_cloud.points[l].rgba = color.rgba;
          pcl::io::savePCDFileBinary(output_pcd_dir+"/"+std::to_string(m)+".pcd",color_cloud);
      }


      std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
      calcLine(plane_list, voxel_size_, iter->second->voxel_origin,
               line_cloud_list);
      // ouster 5,normal 3
      if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 8) {

        for (const auto& l: line_cloud_list) {
            *lidar_line_cloud_3d += l;
            voxel_lines += l;
        }

        line_number_ += line_cloud_list.size();
      }



      pcl::io::savePCDFileBinary(output_pcd_dir+"/cloud.pcd",*iter->second->cloud);


      if(!voxel_lines.empty())
        pcl::io::savePCDFileBinary(output_pcd_dir+"/lines.pcd",voxel_lines);


    }
      ++cnt;
  }

  if(!color_planner_cloud.empty())
    pcl::io::savePCDFileBinary(result_path_+"/plane.pcd",color_planner_cloud);
  if(!lidar_line_cloud_3d->empty())
    pcl::io::savePCDFileBinary(result_path_+"/line.pcd",*lidar_line_cloud_3d);
}


void Calibration::calcLine(
    const std::vector<Plane> &plane_list, const double voxel_size,
    const Eigen::Vector3d origin,
    std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list) {
  if (plane_list.size() >= 2 && plane_list.size() <= plane_max_size_) {
    for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1;
         plane_index1++) {
      for (size_t plane_index2 = plane_index1 + 1;
           plane_index2 < plane_list.size(); plane_index2++) {
        float a1 = plane_list[plane_index1].normal[0];
        float b1 = plane_list[plane_index1].normal[1];
        float c1 = plane_list[plane_index1].normal[2];
        float x1 = plane_list[plane_index1].p_center.x;
        float y1 = plane_list[plane_index1].p_center.y;
        float z1 = plane_list[plane_index1].p_center.z;
        float a2 = plane_list[plane_index2].normal[0];
        float b2 = plane_list[plane_index2].normal[1];
        float c2 = plane_list[plane_index2].normal[2];
        float x2 = plane_list[plane_index2].p_center.x;
        float y2 = plane_list[plane_index2].p_center.y;
        float z2 = plane_list[plane_index2].p_center.z;
        float theta = a1 * a2 + b1 * b2 + c1 * c2;
        //
        float point_dis_threshold = 0.00;
        if (theta > theta_max_ && theta < theta_min_) {
          // for (int i = 0; i < 6; i++) {
          if (plane_list[plane_index1].cloud.size() > 0 &&
              plane_list[plane_index2].cloud.size() > 0) {
            float matrix[4][5];
            matrix[1][1] = a1;
            matrix[1][2] = b1;
            matrix[1][3] = c1;
            matrix[1][4] = a1 * x1 + b1 * y1 + c1 * z1;
            matrix[2][1] = a2;
            matrix[2][2] = b2;
            matrix[2][3] = c2;
            matrix[2][4] = a2 * x2 + b2 * y2 + c2 * z2;
            // six types
            std::vector<Eigen::Vector3d> points;
            Eigen::Vector3d point;
            matrix[3][1] = 1;
            matrix[3][2] = 0;
            matrix[3][3] = 0;
            matrix[3][4] = origin[0];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 1;
            matrix[3][3] = 0;
            matrix[3][4] = origin[1];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 0;
            matrix[3][3] = 1;
            matrix[3][4] = origin[2];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 1;
            matrix[3][2] = 0;
            matrix[3][3] = 0;
            matrix[3][4] = origin[0] + voxel_size;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 1;
            matrix[3][3] = 0;
            matrix[3][4] = origin[1] + voxel_size;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 0;
            matrix[3][3] = 1;
            matrix[3][4] = origin[2] + voxel_size;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.emplace_back(point);
            }
            // std::cout << "points size:" << points.size() << std::endl;
            if (points.size() == 2) {
              pcl::PointCloud<pcl::PointXYZI> line_cloud;
              pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);
              pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);
              float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                                  pow(p1.z - p2.z, 2));
              // 指定近邻个数
              int K = 1;
              // 创建两个向量，分别存放近邻的索引值、近邻的中心距
              std::vector<int> pointIdxNKNSearch1(K);
              std::vector<float> pointNKNSquaredDistance1(K);
              std::vector<int> pointIdxNKNSearch2(K);
              std::vector<float> pointNKNSquaredDistance2(K);
              pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(
                  new pcl::search::KdTree<pcl::PointXYZI>());
              pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(
                  new pcl::search::KdTree<pcl::PointXYZI>());
              kdtree1->setInputCloud(
                  plane_list[plane_index1].cloud.makeShared());
              kdtree2->setInputCloud(
                  plane_list[plane_index2].cloud.makeShared());
              Eigen::Vector3f step = (p2.getVector3fMap() - p1.getVector3fMap()) / length;
              float step_size = 0.01;
              float start_inc = -1;
              float end_inc = -1;
              for (float inc = 0; inc <= length; inc += step_size) {
                pcl::PointXYZI p;
                p.getVector3fMap() = p1.getVector3fMap() + step * inc;
                p.intensity = 100;
                if ((kdtree1->nearestKSearch(p, K, pointIdxNKNSearch1,
                                             pointNKNSquaredDistance1) > 0) &&
                    (kdtree2->nearestKSearch(p, K, pointIdxNKNSearch2,
                                             pointNKNSquaredDistance2) > 0)) {
                    float dis1 = (p.getVector3fMap()-plane_list[plane_index1]
                            .cloud.points[pointIdxNKNSearch1[0]].getVector3fMap()).squaredNorm();
                    float dis2 = (p.getVector3fMap()-plane_list[plane_index2]
                            .cloud.points[pointIdxNKNSearch2[0]].getVector3fMap()).squaredNorm();
                    if(std::max(dis1,dis2)<min_line_dis_threshold_ * min_line_dis_threshold_){
                        line_cloud.push_back(p);
                        if(start_inc<0)
                            start_inc = inc;
                        end_inc = inc;
                    }

                }
              }

              if (line_cloud.size() > 10 && line_cloud.size() > 0.0 * (end_inc-start_inc) / step_size) {
                line_cloud_list.emplace_back(line_cloud);
              }
            }
          }
        }
      }
    }
  }
}


void Calibration::buildVPnp(const Camera& cam, const int& dis_threshold,
    const bool& show_residual,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cam_edge_cloud_2d_vec,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& lidar_line_cloud_3d_vec,
    /*const std::vector<std::vector<int>>& plane_line_number_vec,*/
    std::vector<std::vector<VPnPData>>& pnp_list_vec) {
  pnp_list_vec.clear();

  cv::Mat camera_matrix = cam.camera_matrix_;
  // std::cout << "camera_matrix: " << camera_matrix << std::endl;
  // cv::Mat distortion_coeff = cam.dist_coeffs_;
  cv::Mat distortion_coeff = (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0 );
  // std::cout << "distortion_coeff: " << distortion_coeff << std::endl;
  int height = cam.height_;
  int width = cam.width_;

  for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cam_edge_cloud_2d = cam_edge_cloud_2d_vec[scene_index];
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_line_cloud_3d = lidar_line_cloud_3d_vec[scene_index];

    std::vector<VPnPData> pnp_list;
    std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
    for (int y = 0; y < height; ++y) {
      std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
      for (int x = 0; x < width; ++x) {
        std::vector<pcl::PointXYZI> col_pts_container;
        row_pts_container.emplace_back(col_pts_container);
      }
      img_pts_container.emplace_back(row_pts_container);
    }

    // Eigen::Matrix4d extrinsic_params = cam.Tx_C_L_;
    // std::cout << "wd check extrinsic_params: " << extrinsic_params << std::endl;


     // filter out-view pcd;
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
      new pcl::PointCloud<pcl::PointXYZI>);      
  // project 3d-points into image view
  Eigen::Matrix4d Tx_C_L = cam.Tx_C_L_;
  // std::cout << "wd check projection Tx_C_L: " << Tx_C_L << std::endl;
  pcl::transformPointCloud(*lidar_line_cloud_3d, *transformed_pcd, Tx_C_L);

  for(auto p: *transformed_pcd) {
    if(p.z < 0) continue;
    tmp_pcd->push_back(p);
  }
  Eigen::Matrix4d Tx_L_C = Tx_C_L.inverse();;
  pcl::transformPointCloud(*tmp_pcd, *tmp_pcd, Tx_L_C);

  Vector6d extrinsic_params;
  Eigen::Matrix3d R_C_L = Tx_C_L.topLeftCorner(3, 3);
  Eigen::Vector3d t_C_L = Tx_C_L.topRightCorner(3, 1);
  Eigen::Vector3d euler = R_C_L.eulerAngles(2, 1, 0);

  extrinsic_params[0] = euler[0];
  extrinsic_params[1] = euler[1];
  extrinsic_params[2] = euler[2];
  extrinsic_params[3] = t_C_L[0];
  extrinsic_params[4] = t_C_L[1];
  extrinsic_params[5] = t_C_L[2];

  Eigen::AngleAxisd rotation_vector3;
  rotation_vector3 =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  // std::cout << "wd check rotation_vector3: " 
            // << rotation_vector3.angle() * rotation_vector3.axis() << std::endl;    

  cv::Mat r_vec =
      (cv::Mat_<double>(3, 1)
           << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                   extrinsic_params[4], extrinsic_params[5]);

    // project 3d-points into image view
    // pcl::transformPointCloud(*lidar_line_cloud_3d, *lidar_line_cloud_3d, extrinsic_params);
    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < tmp_pcd->size(); ++i) {
      pcl::PointXYZI point_3d = tmp_pcd->points[i];
      pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
    }
    std::vector<cv::Point2f> pts_2d;

    // cv::Vec3d rvec(0, 0, 0);
    // cv::Vec3d tvec(0, 0, 0);

    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                      pts_2d);
    // cv::fisheye::projectPoints(pts_3d, pts_2d, r_vec, t_vec, camera_matrix, distortion_coeff);
  
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_edge_cloud_2d(
        new pcl::PointCloud<pcl::PointXYZ>);
    // std::vector<int> line_edge_cloud_2d_number;
    for (size_t i = 0; i < pts_2d.size(); i++) {
      pcl::PointXYZ p;
      p.x = pts_2d[i].x;
      p.y = -pts_2d[i].y;
      p.z = 0;
      pcl::PointXYZI pi_3d;
      pi_3d.x = pts_3d[i].x;
      pi_3d.y = pts_3d[i].y;
      pi_3d.z = pts_3d[i].z;
      pi_3d.intensity = 1;
      if (p.x > 0 && p.x < width && pts_2d[i].y > 0 && pts_2d[i].y < height) {
        if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) {
          line_edge_cloud_2d->points.push_back(p);
          // line_edge_cloud_2d_number.emplace_back( plane_line_number_vec[scene_index][i]);
          img_pts_container[pts_2d[i].y][pts_2d[i].x].emplace_back(pi_3d);
        } else {
          img_pts_container[pts_2d[i].y][pts_2d[i].x].emplace_back(pi_3d);
        }
      }
    }
    if (show_residual) {
      cv::Mat residual_img =
          getConnectImg(cam, dis_threshold, cam_edge_cloud_2d, line_edge_cloud_2d);
      if(residual_img.cols > 2000) {
        cv::resize(residual_img, residual_img, cv::Size(residual_img.cols/2, residual_img.rows/2));
      }       
      cv::imshow("residual", residual_img);
      cv::waitKey(1000);
    }
    // save init and result residual image

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(
        new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_lidar =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    kdtree->setInputCloud(cam_edge_cloud_2d);
    kdtree_lidar->setInputCloud(line_edge_cloud_2d);
    tree_cloud = cam_edge_cloud_2d;
    tree_cloud_lidar = line_edge_cloud_2d;
    search_cloud = line_edge_cloud_2d;
    // 指定近邻个数
    int K = 5;
    // 创建两个向量，分别存放近邻的索引值、近邻的中心距
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    std::vector<int> pointIdxNKNSearchLidar(K);
    std::vector<float> pointNKNSquaredDistanceLidar(K);
    int match_count = 0;
    double mean_distance;
    int line_count = 0;
    std::vector<cv::Point2d> lidar_2d_list;
    std::vector<cv::Point2d> img_2d_list;
    std::vector<Eigen::Vector2d> camera_direction_list;
    std::vector<Eigen::Vector2d> lidar_direction_list;
    // std::vector<int> lidar_2d_number;
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
      pcl::PointXYZ searchPoint = search_cloud->points[i];
      if ((kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                  pointNKNSquaredDistance) > 0) &&
          (kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar,
                                        pointNKNSquaredDistanceLidar) > 0)) {
        bool dis_check = true;
        for (int j = 0; j < K; j++) {
          float distance = sqrt(
              pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
              pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
          if (distance > dis_threshold) {
            dis_check = false;
          }
        }
        if (dis_check) {
          cv::Point p_l_2d(search_cloud->points[i].x, -search_cloud->points[i].y);
          cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[0]].x,
                           -tree_cloud->points[pointIdxNKNSearch[0]].y);
          Eigen::Vector2d direction_cam(0, 0);
          std::vector<Eigen::Vector2d> points_cam;
          for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
            Eigen::Vector2d p(tree_cloud->points[pointIdxNKNSearch[i]].x,
                              tree_cloud->points[pointIdxNKNSearch[i]].y);
            points_cam.emplace_back(p);
          }
          calcDirection(points_cam, direction_cam);
          Eigen::Vector2d direction_lidar(0, 0);
          std::vector<Eigen::Vector2d> points_lidar;
          for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
            Eigen::Vector2d p(
                tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x,
                tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
            points_lidar.emplace_back(p);
          }
          calcDirection(points_lidar, direction_lidar);
          // direction.normalize();
          if (checkFov(cam, p_l_2d)) {
            lidar_2d_list.emplace_back(p_l_2d);
            img_2d_list.emplace_back(p_c_2d);
            camera_direction_list.emplace_back(direction_cam);
            lidar_direction_list.emplace_back(direction_lidar);
            // lidar_2d_number.emplace_back(line_edge_cloud_2d_number[i]);
          }
        }
      }
    }
    for (size_t i = 0; i < lidar_2d_list.size(); i++) {
      int y = lidar_2d_list[i].y;
      int x = lidar_2d_list[i].x;
      int pixel_points_size = img_pts_container[y][x].size();
      if (pixel_points_size > 0) {
        VPnPData pnp;
        pnp.x = 0;
        pnp.y = 0;
        pnp.z = 0;
        pnp.u = img_2d_list[i].x;
        pnp.v = img_2d_list[i].y;
        for (size_t j = 0; j < pixel_points_size; j++) {
          pnp.x += img_pts_container[y][x][j].x;
          pnp.y += img_pts_container[y][x][j].y;
          pnp.z += img_pts_container[y][x][j].z;
        }
        pnp.x = pnp.x / pixel_points_size;
        pnp.y = pnp.y / pixel_points_size;
        pnp.z = pnp.z / pixel_points_size;
        pnp.direction = camera_direction_list[i];
        pnp.direction_lidar = lidar_direction_list[i];
        // pnp.number = lidar_2d_number[i];
        pnp.number = 0;
        float theta = pnp.direction.dot(pnp.direction_lidar);
        if (theta > direction_theta_min_ || theta < direction_theta_max_) {
          pnp_list.emplace_back(pnp);
        }
      }
    }

    pnp_list_vec.emplace_back(pnp_list);
  }
  // std::cout << "wd check pnp_list_vec size:"  << pnp_list_vec.size() << std::endl;
} 


void Calibration::buildVPnp(const Camera& cam, const int& dis_threshold,
    const int& cnt,
    const bool& show_residual,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cam_edge_cloud_2d_vec,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& lidar_line_cloud_3d_vec,
    /*const std::vector<std::vector<int>>& plane_line_number_vec,*/
    std::vector<std::vector<VPnPData>>& pnp_list_vec) {
  pnp_list_vec.clear();

  cv::Mat camera_matrix = cam.camera_matrix_;
  // cv::Mat distortion_coeff = cam.dist_coeffs_;
  cv::Mat distortion_coeff = (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0 );
  int height = cam.height_;
  int width = cam.width_;

  // if(debugMode_) {
  //   std::cout << "check buildVPnp camera_matrix: " << camera_matrix << std::endl;
  //   std::cout << "check buildVPnp distortion_coeff: " << distortion_coeff << std::endl;
  //   std::cout << "check buildVPnp height: " << height << std::endl;
  //   std::cout << "check buildVPnp width: " << width << std::endl;
  // }

  for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cam_edge_cloud_2d = cam_edge_cloud_2d_vec[scene_index];
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_line_cloud_3d = lidar_line_cloud_3d_vec[scene_index];

    std::vector<VPnPData> pnp_list;
    std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
    for (int y = 0; y < height; ++y) {
      std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
      for (int x = 0; x < width; ++x) {
        std::vector<pcl::PointXYZI> col_pts_container;
        row_pts_container.emplace_back(col_pts_container);
      }
      img_pts_container.emplace_back(row_pts_container);
    }

    /** filter out-view pcd;***/
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);      
    
    Eigen::Matrix4d Tx_C_L = cam.Tx_C_L_;
    // std::cout << "wd check projection Tx_C_L: " << Tx_C_L << std::endl;
    pcl::transformPointCloud(*lidar_line_cloud_3d, *transformed_pcd, Tx_C_L);

    for(auto p: *transformed_pcd) {
      if(p.z < 0) continue;
      tmp_pcd->push_back(p);
    }
    if(tmp_pcd->size() < 10) return;

    Eigen::Matrix4d Tx_L_C = Tx_C_L.inverse();;
    pcl::transformPointCloud(*tmp_pcd, *tmp_pcd, Tx_L_C);

    /*** project 3d-points into image view ***/
    Vector6d extrinsic_params;
    Eigen::Matrix3d R_C_L = Tx_C_L.topLeftCorner(3, 3);
    Eigen::Vector3d t_C_L = Tx_C_L.topRightCorner(3, 1);
    Eigen::Vector3d euler = R_C_L.eulerAngles(2, 1, 0);

    extrinsic_params[0] = euler[0];
    extrinsic_params[1] = euler[1];
    extrinsic_params[2] = euler[2];
    extrinsic_params[3] = t_C_L[0];
    extrinsic_params[4] = t_C_L[1];
    extrinsic_params[5] = t_C_L[2];

    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    // std::cout << "wd check rotation_vector3: " 
    //           << rotation_vector3.angle() * rotation_vector3.axis() << std::endl;    

    cv::Mat r_vec =
        (cv::Mat_<double>(3, 1)
             << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
         rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
         rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    // std:cout << "wd check r_vec: " << r_vec << std::endl; 
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                     extrinsic_params[4], extrinsic_params[5]);

    // wrong!!!
    // pcl::transformPointCloud(*lidar_line_cloud_3d, *lidar_line_cloud_3d, extrinsic_params);
    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < tmp_pcd->size(); ++i) {
      pcl::PointXYZI point_3d = tmp_pcd->points[i];
      pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
    }
    std::vector<cv::Point2f> pts_2d;

    // cv::Vec3d rvec(0, 0, 0);
    // cv::Vec3d tvec(0, 0, 0);
    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                      pts_2d);                                          

    // cv::fisheye::projectPoints(pts_3d, pts_2d, r_vec, t_vec, camera_matrix, distortion_coeff);

    pcl::PointCloud<pcl::PointXYZ>::Ptr line_edge_cloud_2d(
        new pcl::PointCloud<pcl::PointXYZ>);
    // std::vector<int> line_edge_cloud_2d_number;
    for (size_t i = 0; i < pts_2d.size(); i++) {
      pcl::PointXYZ p;
      p.x = pts_2d[i].x;
      p.y = -pts_2d[i].y;
      p.z = 0;
      pcl::PointXYZI pi_3d;
      pi_3d.x = pts_3d[i].x;
      pi_3d.y = pts_3d[i].y;
      pi_3d.z = pts_3d[i].z;
      pi_3d.intensity = 1;
      if (p.x > 0 && p.x < width && pts_2d[i].y > 0 && pts_2d[i].y < height) {
        if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) {
          line_edge_cloud_2d->points.push_back(p);
          // line_edge_cloud_2d_number.push_back( plane_line_number_vec[scene_index][i]);
          img_pts_container[pts_2d[i].y][pts_2d[i].x].emplace_back(pi_3d);
        } else {
          img_pts_container[pts_2d[i].y][pts_2d[i].x].emplace_back(pi_3d);
        }
      }
    }

    // if (show_residual) {
      // to debug
    if (debugMode_) {  
      std::string save_residual_path = result_path_ + "/" + "residual";
      if(access(save_residual_path.c_str(),0) != 0) {
        mkdir(save_residual_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      }

      // if(cam.cam_name_ == "front_near" && scene_index == 0) {
        cv::Mat residual_img =
          getConnectImg(cam, dis_threshold, cam_edge_cloud_2d, line_edge_cloud_2d);
        if(residual_img.cols > 2000) {
          cv::resize(residual_img, residual_img, cv::Size(residual_img.cols/2, residual_img.rows/2));
        } 
        // cv::imshow("residual", residual_img);
        // cv::waitKey(1000);
  
        // save init and result residual image
        if((dis_threshold == 20 && cnt == 0)) {
          cv::imwrite(save_residual_path + "/" + cam.cam_name_ + "_sceneID_" + std::to_string(scene_index) + "_init_residual.jpg", residual_img.clone());
        }
  
        // low_dis_threshold
        if((dis_threshold == 7 && cnt == 1)) {
          cv::imwrite(save_residual_path + "/" + cam.cam_name_ + "_sceneID_" + std::to_string(scene_index) + "_result_residual.jpg", residual_img);
        }
      // }
    }
    

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(
        new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_lidar =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    kdtree->setInputCloud(cam_edge_cloud_2d);
    kdtree_lidar->setInputCloud(line_edge_cloud_2d);
    tree_cloud = cam_edge_cloud_2d;
    tree_cloud_lidar = line_edge_cloud_2d;
    search_cloud = line_edge_cloud_2d;
    // 指定近邻个数
    int K = 5;
    // 创建两个向量，分别存放近邻的索引值、近邻的中心距
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    std::vector<int> pointIdxNKNSearchLidar(K);
    std::vector<float> pointNKNSquaredDistanceLidar(K);
    int match_count = 0;
    double mean_distance;
    int line_count = 0;
    std::vector<cv::Point2d> lidar_2d_list;
    std::vector<cv::Point2d> img_2d_list;
    std::vector<Eigen::Vector2d> camera_direction_list;
    std::vector<Eigen::Vector2d> lidar_direction_list;
    // std::vector<int> lidar_2d_number;
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
      pcl::PointXYZ searchPoint = search_cloud->points[i];
      if ((kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                  pointNKNSquaredDistance) > 0) &&
          (kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar,
                                        pointNKNSquaredDistanceLidar) > 0)) {
        bool dis_check = true;
        for (int j = 0; j < K; j++) {
          float distance = sqrt(
              pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
              pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
          if (distance > dis_threshold) {
            dis_check = false;
          }
        }
        if (dis_check) {
          cv::Point p_l_2d(search_cloud->points[i].x, -search_cloud->points[i].y);
          cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[0]].x,
                           -tree_cloud->points[pointIdxNKNSearch[0]].y);
          Eigen::Vector2d direction_cam(0, 0);
          std::vector<Eigen::Vector2d> points_cam;
          for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
            Eigen::Vector2d p(tree_cloud->points[pointIdxNKNSearch[i]].x,
                              tree_cloud->points[pointIdxNKNSearch[i]].y);
            points_cam.emplace_back(p);
          }
          calcDirection(points_cam, direction_cam);
          Eigen::Vector2d direction_lidar(0, 0);
          std::vector<Eigen::Vector2d> points_lidar;
          for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
            Eigen::Vector2d p(
                tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x,
                tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
            points_lidar.emplace_back(p);
          }
          calcDirection(points_lidar, direction_lidar);
          // direction.normalize();
          if (checkFov(cam, p_l_2d)) {
            lidar_2d_list.emplace_back(p_l_2d);
            img_2d_list.emplace_back(p_c_2d);
            camera_direction_list.emplace_back(direction_cam);
            lidar_direction_list.emplace_back(direction_lidar);
            // lidar_2d_number.push_back(line_edge_cloud_2d_number[i]);
          }
        }
      }
    }
    for (size_t i = 0; i < lidar_2d_list.size(); i++) {
      int y = lidar_2d_list[i].y;
      int x = lidar_2d_list[i].x;
      int pixel_points_size = img_pts_container[y][x].size();
      if (pixel_points_size > 0) {
        VPnPData pnp;
        pnp.x = 0;
        pnp.y = 0;
        pnp.z = 0;
        pnp.u = img_2d_list[i].x;
        pnp.v = img_2d_list[i].y;
        for (size_t j = 0; j < pixel_points_size; j++) {
          pnp.x += img_pts_container[y][x][j].x;
          pnp.y += img_pts_container[y][x][j].y;
          pnp.z += img_pts_container[y][x][j].z;
        }
        pnp.x = pnp.x / pixel_points_size;
        pnp.y = pnp.y / pixel_points_size;
        pnp.z = pnp.z / pixel_points_size;
        pnp.direction = camera_direction_list[i];
        pnp.direction_lidar = lidar_direction_list[i];
        // pnp.number = lidar_2d_number[i];
        pnp.number = 0;
        float theta = pnp.direction.dot(pnp.direction_lidar);
        if (theta > direction_theta_min_ || theta < direction_theta_max_) {
          pnp_list.emplace_back(pnp);
        }
      }
    }

    pnp_list_vec.emplace_back(pnp_list);
  }
  // std::cout << "wd check pnp_list_vec size:"  << pnp_list_vec.size() << std::endl;
} 

void Calibration::loadImgAndPointcloud(const std::vector<std::string>& pcd_paths,
                                       const std::vector<std::vector<std::string>>& cams_paths){
  lidar.pcd_vec_.resize(scene_num_);    
  for(size_t i = 0; i < scene_num_; ++i) {
    lidar.pcd_vec_[i] = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  }   

  for(int scene_index = 0; scene_index < pcd_paths.size(); ++scene_index) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr origin_cloud =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

    std::fstream pcd_file;
    pcd_file.open(pcd_paths[scene_index], ios::in);    
    if(!pcd_file) {
      std::cout << "Pcd file " << pcd_paths[scene_index] << " does not exit" << std::endl;
    }  

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_paths[scene_index], *origin_cloud)== -1){
      ROS_ERROR_STREAM("Couldn't read pcd file\n");
      return;
    }

    // // wd todo
    // // if load mapping in dr coordinate, convert to main lidar coordinate
    // Eigen::Matrix4d Tx_dr_L;
    // // just for 29
    // Tx_dr_L << 0.024205, -0.999358, 0.026431, 1.565430,
    //       0.999624, 0.024534, 0.012197, -0.015458,
    //       -0.012838, 0.026125, 0.999576, 1.911863,
    //       0.000000, 0.000000, 0.000000, 1.000000;             
    // std::cout << "just for 29 vehicle, trandform pcd, Tx_dr_L: " << Tx_dr_L <<  std::endl; 
    // Eigen::Matrix4d Tx_L_dr = Tx_dr_L.inverse();            
    // pcl::transformPointCloud(*origin_cloud, *origin_cloud, Tx_L_dr);

    lidar.pcd_vec_[scene_index] = origin_cloud;
  }

  int img_cnt = 0;
  for(int cam_index = 0; cam_index < cams.size(); ++cam_index) {
    cv::Mat map1, map2;
    cv::Mat new_K = cams[cam_index].camera_matrix_;
    for(int scene_index = 0; scene_index < scene_num_; ++scene_index) {
      std::fstream img_file;
      img_file.open(cams_paths[cam_index][scene_index], ios::in);
      if(!img_file) {
        std::cout << "File " << cams_paths[cam_index][scene_index] << " does not exit" << std::endl;
        return;
      };

      // read image
      cv::Mat raw_img = cv::imread(cams_paths[cam_index][scene_index]);
      cv::Mat save_img;
      raw_img.copyTo(save_img);
      cams[cam_index].raw_imgs.emplace_back(save_img);
      cv::Mat undistort_img;
      //  保存去畸变的图像
      cv::fisheye::initUndistortRectifyMap(cams[cam_index].camera_matrix_, cams[cam_index].dist_coeffs_, cv::Mat(), new_K, (cv::Size(raw_img.cols,raw_img.rows)), CV_16SC2, map1, map2);
      cv::remap(raw_img, undistort_img, map1, map2, cv::INTER_LINEAR);
      // cams[cam_index].rgb_imgs.emplace_back(undistort_img.clone());
      cv::Mat undistort_save_img;
      undistort_img.copyTo(undistort_save_img);
      cams[cam_index].rgb_imgs.emplace_back(undistort_save_img);
     
      ++img_cnt;
    }

    // 去完畸变之后，畸变参数置0
    // cams[cam_index].dist_coeffs_ = (cv::Mat_<double>(4, 1) << 0., 0., 0., 0.);
    // cams[cam_index].camera_matrix_ = new_K;
    cams[cam_index].width_ = cams[cam_index].rgb_imgs[0].cols;
    cams[cam_index].height_ = cams[cam_index].rgb_imgs[0].rows;
  }

  ROS_INFO_STREAM("Sucessfully load " << lidar.pcd_vec_.size() << " Point Clouds and " << img_cnt << " images."); 
}
        
void Calibration::calcDirection(const std::vector<Eigen::Vector2d> &points,
                                Eigen::Vector2d &direction) {
  Eigen::Vector2d mean_point(0, 0);
  for (size_t i = 0; i < points.size(); i++) {
    mean_point(0) += points[i](0);
    mean_point(1) += points[i](1);
  }
  mean_point(0) = mean_point(0) / points.size();
  mean_point(1) = mean_point(1) / points.size();
  Eigen::Matrix2d S;
  S << 0, 0, 0, 0;
  for (size_t i = 0; i < points.size(); i++) {
    Eigen::Matrix2d s =
        (points[i] - mean_point) * (points[i] - mean_point).transpose();
    S += s;
  }
  Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
  Eigen::MatrixXcd evecs = es.eigenvectors();
  Eigen::MatrixXcd evals = es.eigenvalues();
  Eigen::MatrixXd evalsReal;
  evalsReal = evals.real();
  Eigen::MatrixXf::Index evalsMax;
  evalsReal.rowwise().sum().maxCoeff(&evalsMax); //得到最大特征值的位置
  direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
}

cv::Mat Calibration::getProjectionImg(const Camera& cam, 
                                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_lidar_cloud,
                                      const cv::Mat& rgb_image) {
  cv::Mat depth_projection_img;

  projection(cam, raw_lidar_cloud, projection_type_, false,
             depth_projection_img);
  // projection(cam, raw_lidar_cloud, DEPTH, false,
  //            depth_projection_img);               

  cv::Mat map_img = cv::Mat::zeros(cam.height_, cam.width_, CV_8UC3);
  cv::Mat merge_img;
  rgb_image.copyTo(merge_img);
  for (int x = 0; x < map_img.cols; x++) {
    for (int y = 0; y < map_img.rows; y++) {
      uint8_t r, g, b;
      float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
      mapJet(norm, 0, 1, r, g, b);
      map_img.at<cv::Vec3b>(y, x)[0] = b;
      map_img.at<cv::Vec3b>(y, x)[1] = g;
      map_img.at<cv::Vec3b>(y, x)[2] = r;

      if(norm != 0.0) {
        cv::circle(merge_img, cv::Point2i(x, y), 2, cv::Scalar(b, g, r), -1);
      }
    }
  }
  // cv::imshow("map jet", map_img);
  // cv::waitKey();
  // cv::Mat merge_img = 0.5 * map_img + 0.8 * rgb_image;
  return merge_img; 
}

cv::Mat Calibration::showPcdOnImg(const Camera& cam, 
                                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_lidar_cloud,
                                  const cv::Mat& rgb_image) {
  cv::Mat projection_img;
  rgb_image.copyTo(projection_img);

  pcl::PointCloud<pcl::PointXYZI> pcd_camera;
  pcl::transformPointCloud(*raw_lidar_cloud, pcd_camera, cam.Tx_C_L_);

  double max_intensity = 0.0;
  pcl::PointCloud<pcl::PointXYZI> filter_pcd;
  std::vector<cv::Point3f> pt_3d_vec;
  for(auto pcd: pcd_camera) {
    if(pcd.intensity > max_intensity) {
      max_intensity = pcd.intensity;
    }

    // filter out-off fov
    if(pcd.z < 0 || pcd.z >100) continue;
    filter_pcd.push_back(pcd);
    pt_3d_vec.emplace_back(cv::Point3f(pcd.x, pcd.y, pcd.z));
  }
  if(max_intensity == 0) {
    max_intensity = 1; // 或者用depth可视化， todo
  }

  if(pt_3d_vec.size() == 0) { 
    return projection_img;
  }

  cv::Vec3d rvec(0, 0, 0);
  cv::Vec3d tvec(0, 0, 0);
  std::vector<cv::Point2f> pt_2d_vec;
  cv::fisheye::projectPoints(pt_3d_vec, pt_2d_vec, rvec, tvec, cam.camera_matrix_, cam.dist_coeffs_);
  for(int point_index = 0; point_index < pt_2d_vec.size(); ++point_index) {
    if(pt_2d_vec[point_index].x >= 0 && pt_2d_vec[point_index].x < rgb_image.cols && pt_2d_vec[point_index].y >= 0 && pt_2d_vec[point_index].y < rgb_image.rows) {
      cv::circle(projection_img, cv::Point2f(pt_2d_vec[point_index].x, pt_2d_vec[point_index].y), 1, 
                 cv::Scalar(0, filter_pcd[point_index].intensity / max_intensity * 192 + 63,
                 255 - filter_pcd[point_index].intensity / max_intensity * 255), -1);
    }
  }

  return projection_img.clone();
}

Eigen::Vector3d Calibration::convertRotationMatrixToEulerYPR(const Eigen::Matrix3d& R) {
  Eigen::Vector3d ypr;
  ypr(0) = atan2(R(1, 0), R(0, 0));
  ypr(1) = asin(-R(2, 0));
  ypr(2) = atan2(R(2, 1), R(2, 2));
  return ypr;
}   

 Eigen::Matrix3d Calibration::convertEulerYPRToRotationMatrix(const Eigen::Vector3d& yaw_pitch_roll) {
  const double r = yaw_pitch_roll(2);
  const double p = yaw_pitch_roll(1);
  const double y = yaw_pitch_roll(0);
  const double sr = sin(r);
  const double sp = sin(p);
  const double sy = sin(y);
  const double cr = cos(r);
  const double cp = cos(p);
  const double cy = cos(y);
  Eigen::Matrix3d R;
  // clang-format off
  R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
       sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
           -sp,                cp * sr,                cp * cr;
  // clang-format on
  return R;
}

Eigen::AngleAxisd Calibration::convertEulerYPRToAngleAxis(const Eigen::Vector3d& yaw_pitch_roll) {
  // euler to Quanterion
  const double r = yaw_pitch_roll(2) * 0.5;
  const double p = yaw_pitch_roll(1) * 0.5;
  const double y = yaw_pitch_roll(0) * 0.5;
  const double sr = sin(r);
  const double sp = sin(p);
  const double sy = sin(y);
  const double cr = cos(r);
  const double cp = cos(p);
  const double cy = cos(y);

  Eigen::Quaterniond q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  q.w() = cr * cp * cy + sr * sp * sy;
  q.x() = sr * cp * cy - cr * sp * sy;
  q.y() = cr * sp * cy + sr * cp * sy;
  q.z() = cr * cp * sy - sr * sp * cy;

  // Quanterion to angleaxisd
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

void Calibration::writeCamExToPbFile(const Eigen::Matrix4d& T_dr_L, std::string& output_file) {
  tutorial::SensorExtrinsic T_dr_L_pb;
  tutorial::Rotation* rot = T_dr_L_pb.add_rotation();
  Eigen::Matrix3d rotation_matrix = T_dr_L.topLeftCorner(3, 3);
  Eigen::Quaterniond qua = Eigen::Quaterniond(rotation_matrix);
  // std::cout << qua.coeffs() << std::endl;
  rot->set_x(static_cast<double>(qua.x()));
  rot->set_y(static_cast<double>(qua.y()));
  rot->set_z(static_cast<double>(qua.z()));
  rot->set_w(static_cast<double>(qua.w()));

  tutorial::Translation* trans = T_dr_L_pb.add_translation();
  Eigen::Vector3d translation = T_dr_L.topRightCorner(3, 1);
  trans->set_x(translation.x());
  trans->set_y(translation.y());
  trans->set_z(translation.z());
  
  tutorial::Rotation rot_test = T_dr_L_pb.rotation()[0];
  // std::cout << rot_test.x() << std::endl;
  if(!writeProtoToTextFile(output_file, T_dr_L_pb)) {
    return;
  }

  // google::protobuf::ShutdownProtobufLibrary();  
}

bool Calibration::writeProtoToTextFile( std::string& file,
                          const google::protobuf::Message& proto) {
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);  // 0644 is the file permission
  if (fd == -1) {
    std::cerr << "ProtoIO: failed to open file(WR): " << file << std::endl;
    return false;
  }
  google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);
  bool flag = google::protobuf::TextFormat::Print(proto, output);
  // bool flag = google::protobuf::TextFormat::PrintToString(proto, &file);

  delete output;
  close(fd);

  // std::ofstream fw; 
  // fw.open(file, std::ios::out | std::ios::binary | std::ios::app);
  // proto.SerializePartialToOstream(&fw);
  // // google::protobuf::io::OstreamOutputStream *output = new google::protobuf::io::OstreamOutputStream(&fw);
  // // google::protobuf::TextFormat::Print(proto, output);
 
  // // delete output;
  // fw.close();

  // return flag;
  return true;
}

std::vector<pcl::PointCloud<pcl::PointXYZI>> Calibration::linesFromGroundPlane(const pcl::PointCloud<pcl::PointXYZI>& cloud){
    constexpr float resolution = 0.02;
    Eigen::Vector2f mx(100,-100),my(100,-100);
    for(const auto& p:cloud.points){
        if(mx.x()>p.x)
            mx.x() = p.x;
        if(mx.y()<p.x)
            mx.y() = p.x;
        if(my.x()>p.y)
            my.x() = p.y;
        if(my.y()<p.y)
            my.y() = p.y;
    }
    float xl = mx.y()-mx.x();
    float yl = my.y()-my.x();
    int row = int(xl/resolution)+1;
    int col = int(yl/resolution)+1;
    cv::Mat intensity_matf(row,col,CV_32F,cv::Scalar(0));

    for(const auto&p: cloud.points){
        int i = (p.x-mx(0))/resolution;
        int j = (p.y-my(0))/resolution;
        intensity_matf.at<float>(i,j) = p.intensity;

    }
    intensity_matf = (intensity_matf>18)*255.f;
    cv::Mat im;
    intensity_matf.convertTo(im,CV_8U);


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // You can change the kernel size as needed

    // Perform erosion and dilation
    for (int i = 0; i < 3; ++i) {
        cv::dilate(im, im, kernel, cv::Point(-1, -1), 1);
        cv::erode(im, im, kernel, cv::Point(-1, -1), 1);
    }

    for (int i = 0; i < 3; ++i) {
        cv::erode(im, im, kernel, cv::Point(-1, -1), 1);
        cv::dilate(im, im, kernel, cv::Point(-1, -1), 1);
    }

    // Apply edge detection (Canny edge detector)
    cv::Mat edges;
    cv::Canny(im, edges, 50, 150, 3);


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // Draw contours on a blank image (for visualization)

    //debug
//    cv::Mat contourImage = cv::Mat::zeros(edges.size(), CV_8UC3);

//    for (size_t i = 0; i < contours.size(); ++i) {
//        if(contours[i].size()<20)
//            continue;
//        cv::Scalar color = cv::Scalar(0, 255, 0); // Green color for contours
//        cv::drawContours(contourImage, contours, static_cast<int>(i), color, 2, cv::LINE_8, hierarchy);
//    }

    // Display the image with detected contours
//    cv::imshow("Contours", contourImage);
//    cv::imshow("im", im);
//    cv::imshow("edges", edges);
//    cv::waitKey(0);
//    cv::destroyAllWindows();


    std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud;
    for(auto contour:contours){
        if(contour.size()<20)
            continue;
        pcl::PointCloud<pcl::PointXYZI> line;
        for(auto p:contour){
            pcl::PointXYZI pl;
            pl.getVector3fMap()<<float(p.y)*resolution+mx(0),
                    float(p.x)*resolution+my(0),
                                 0;
            line.push_back(pl);
        }
        line_cloud.push_back(line);
    }


    return line_cloud;
}

// ref:https://zhuanlan.zhihu.com/p/580163189?utm_id=0 
// ref: https://github.com/koide3/hdl_graph_slam
// ref:https://zhuanlan.zhihu.com/p/470690484?utm_medium=social&utm_oi=980540876585635840&utm_psn=1687572596030328832&utm_source=wechat_session&utm_id=0
void Calibration::extractFloorPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_pcd,
                                    const Eigen::Matrix4d& T_dr_L,
                                    pcl::PointCloud<pcl::PointXYZI>& pcd_floor,
                                    std::vector<pcl::PointCloud<pcl::PointXYZI>>& lines) {
  pcl::PointCloud<pcl::PointXYZI> pcd_transformed;
  pcl::transformPointCloud(*input_pcd, pcd_transformed, T_dr_L);
  // extract floor
  // pcl::PointCloud<pcl::PointXYZI> pcd_floor;
  for(auto p: pcd_transformed) {
    if(p.z >= -0.25 && p.z <= 0.25 &&
     p.x >= -30 && p.x <= 30 && 
     p.y >= -15 && p.y <= 15) {
     pcd_floor.push_back(p);
    }
  }
    lines = linesFromGroundPlane(pcd_floor);
    pcl::io::savePCDFileBinary(result_path_+"/floor.pcd",pcd_floor);

  // too few points for RANSAC
  double floor_pts_thresh = 100;
  
  if (pcd_floor.size() < floor_pts_thresh) {
    std::cerr << "too few points for RANSAC" << std::endl;
    std::cout << "pcd_floor size: " << pcd_floor.size() << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZI> pcd_floor_tmp;
  Eigen::Matrix4d T_L_dr = T_dr_L.inverse();
  pcl::transformPointCloud(pcd_floor, pcd_floor_tmp, T_L_dr);
  pcd_floor = pcd_floor_tmp;

  for(auto& c:lines)
      pcl::transformPointCloud(c,c,T_L_dr);
  
}

bool Calibration::addFloorConstriant(const pcl::PointCloud<pcl::PointXYZI>& pcd_floor,
                                     const Eigen::Matrix4d& T_dr_L,
                                     Eigen::Matrix4d& T_dr_L_new) {
  pcl::PointCloud<pcl::PointXYZI> pcd_floor_tmp;
  pcl::transformPointCloud(pcd_floor, pcd_floor_tmp, T_dr_L);
  
  // too few points for RANSAC
  double floor_normal_thresh = 10;

  // 使用RANSAC算法拟合地平面参数
  pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr model_p(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(pcd_floor_tmp.makeShared()));
  pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_p);
  ransac.setDistanceThreshold(0.05);
  ransac.computeModel();

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  ransac.getInliers(inliers->indices);
  // too few inliers
  double floor_pts_thresh = 100;
  if (inliers->indices.size() < floor_pts_thresh) {
    std::cerr << "too few inliers" << std::endl;
    return false;
  }

  pcl::PointCloud<pcl::PointXYZI> floor_plane;
  assert(inliers->indices.size() <= pcd_floor_tmp.size());
  for(size_t i = 0; i < inliers->indices.size(); ++i) {
    floor_plane.push_back(pcd_floor_tmp[i]);
  }

  // 验证算出的法线是否是真正的地面法线：是否和z轴夹角小于10°
  Eigen::Vector4f reference = Eigen::Vector4f::UnitZ();
  Eigen::VectorXf coeffs;
  ransac.getModelCoefficients(coeffs);
  double dot = coeffs.head<3>().dot(reference.head<3>());
  if (std::abs(dot) < std::cos(floor_normal_thresh*M_PI/180.0)) {
    // the normal is not vertical
    std::cerr << "the normal is not vertical!" << std::endl;
    return false;
  }
  // make the normal upward
  if (coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
    coeffs *= -1.0f;
  }

  double pitch = 0, roll = 0, yaw = 0;
  pitch = atan2((double)coeffs.head<3>()(1), (double)coeffs.head<3>()(2));
  roll = -atan2((double)coeffs.head<3>()(0), (double)coeffs.head<3>()(2));
  std::cout << "pitch: " << pitch * 180 / M_PI  << ", " << "roll: " << roll * 180 / M_PI  << std::endl;
  if(pitch > (5.0 * M_PI / 180) || pitch > (5.0 * M_PI / 180)) {
    std::cout << "calibration result may be a larger error! " << std::endl;
  }

  // Eigen::Matrix3d R_yp = convertEulerYPRToRotationMatrix(Eigen::Vector3d(yaw, pitch, roll));
  
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  // rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
  //            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitY()) *
  //            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX());
  Eigen::Matrix3d rotation_pitch;
  Eigen::Matrix3d rotation_roll;
  Eigen::Matrix3d rotation_yaw;
  rotation_pitch << 1, 0, 0, 
                    0, cos(pitch), -sin(pitch), 
                    0, sin (pitch), cos (pitch);
  rotation_roll << cos (roll), 0, sin(roll),
                     0, 1, 0,
                     -sin (roll), 0, cos (roll);
  rotation_yaw << cos (yaw), -sin(yaw), 0, 
                     sin (yaw), cos (yaw), 0, 
                     0, 0, 1;
  rotation = rotation_yaw * rotation_roll * rotation_pitch;
  Eigen::Matrix4d T_dr_pr = Eigen::Matrix4d::Identity();
  T_dr_pr.block<3, 3>(0, 0) = rotation;
  T_dr_pr.block<3, 1>(0, 3) = Eigen::Vector3d::Zero();

  Eigen::Vector4d coeffs_double;
  coeffs_double[0] = (double)coeffs[0];
  coeffs_double[1] = (double)coeffs[1];
  coeffs_double[2] = (double)coeffs[2];
  coeffs_double[3] = (double)coeffs[3];
  Eigen::Vector4d new_norm = T_dr_pr * coeffs_double;
  if((new_norm.head<3>().normalized() - Eigen::Vector3d::UnitZ()).norm() < 10e-3) {
    std::cout << "successful to constraint pitch and roll" << std::endl;
    std::cout << "new_norm: " << new_norm.head<3>() << std::endl;

    T_dr_L_new = T_dr_pr * T_dr_L;

    pcl::PointCloud<pcl::PointXYZI> before_floor_plane;
    Eigen::Matrix4d T_L_dr = T_dr_L.inverse();
    pcl::transformPointCloud(floor_plane, before_floor_plane, T_L_dr);
    pcl::PointCloud<pcl::PointXYZI> new_floor_plane_transformed;
    pcl::transformPointCloud(before_floor_plane, new_floor_plane_transformed, T_dr_L_new);
    double z_mean = 0.0;
    for(auto p: new_floor_plane_transformed) {
      z_mean += p.z;
    }
    z_mean /= double(floor_plane.size());
    std::cout << "z_mean: " << z_mean << std::endl;
   
    T_dr_L_new(2, 3) -= z_mean;
    pcl::transformPointCloud(before_floor_plane, new_floor_plane_transformed, T_dr_L_new);
    double refine_z_mean = 0.0;
    for(auto p: new_floor_plane_transformed) {
      refine_z_mean += p.z;
    }
    refine_z_mean /= double(new_floor_plane_transformed.size());
    std::cout << "after z_mean: " << refine_z_mean << std::endl;
    return true;
  }
  else {
    std::cerr << "failed to constraint pitch and roll" << std::endl;
    std::cout << "new_norm: " << new_norm.head<3>() << std::endl;
    return false;
  }

}


#endif