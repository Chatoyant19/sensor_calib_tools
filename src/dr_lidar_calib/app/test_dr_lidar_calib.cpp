#include <pcl/io/pcd_io.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "dr_lidar_calib.h"
#include "file_io.h"
#include "extract_lidar_feature.h"
#include "exrtace_image_feature.h"
#include "match_features.h"
#include "floor_plane_constriant.h"
#include "show_tools.h"

#define debug
using namespace dr_lidar_calib;

std::string pcd_path_;
std::vector<std::vector<cv::Mat>> imgs_vec_;
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> visual_pcd_vec_;
Eigen::Matrix4d init_Tx_dr_L_;

std::string save_lidar_extrinsic_name_;
void loadConfigFile(const std::string& calib_setting_path, DrLidarCalibParam& calib_param);

int main(int argc, char** argv) {
  if ( argc != 2 ) {
    std::cout<<"Usage: test_dr_lidar_calib path_to_config_file"<< std::endl;
    return -1;
  }
  const std::string calib_setting_path = std::string(argv[1]);
  DrLidarCalibParam calib_param;
  loadConfigFile(calib_setting_path, calib_param);

  #ifdef debug
  // show init 
  for(size_t cam_index = 0; cam_index < calib_param.cams_name_vec.size(); ++cam_index) {
    Eigen::Matrix4d Tx_C_L = calib_param.camera_extrinsics_vec[cam_index].inverse() * init_Tx_dr_L_;
    for(size_t scene_index = 0; scene_index < calib_param.scene_num; ++scene_index) {
      cv::Mat init_img = show_tools::getProjectionImg(imgs_vec_[scene_index][cam_index], visual_pcd_vec_[scene_index],
        Tx_C_L, calib_param.camera_matrix_vec[cam_index], calib_param.dist_coeffs_vec[cam_index]);
      cv::imwrite(calib_param.result_path + "/" + calib_param.cams_name_vec[cam_index] + 
        "_sceneID_" + std::to_string(scene_index) + "_init.png", init_img);
    }
  }
  #endif

  std::unique_ptr<DrLidarCalib> dr_lidar_calib = std::make_unique<DrLidarCalib>(calib_param);
  for(size_t scene_index = 0; scene_index < calib_param.scene_num; ++scene_index) {
    std::string map_points_path = pcd_path_ + "/" + std::to_string(scene_index) + ".pcd"; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_pcd = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(map_points_path, *map_pcd);
    dr_lidar_calib->processLidar(map_pcd);
  }

  // dr_lidar_calib->init();

  Eigen::Matrix4d Tx_dr_L;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> cams_extrinsics_vec;
  dr_lidar_calib->run(init_Tx_dr_L_, visual_pcd_vec_, imgs_vec_, Tx_dr_L, cams_extrinsics_vec, false);

  /***write extrinsics to files***/
  std::string lidar_extrinsics_folder = calib_param.result_path + "/lidar";
  if(access(lidar_extrinsics_folder.c_str(),0) != 0) {
    mkdir(lidar_extrinsics_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
  std::string lidar_extrinsics_path = lidar_extrinsics_folder + "/" + save_lidar_extrinsic_name_;      
  file_io::writeExtrinsicToPbFile(Tx_dr_L, lidar_extrinsics_path);

  std::string camera_extrinsics_folder = calib_param.result_path + "/camera";
  if(access(camera_extrinsics_folder.c_str(),0) != 0) {
    mkdir(camera_extrinsics_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
  for(size_t cam_index = 0; cam_index < calib_param.cams_name_vec.size(); ++cam_index) {
    std::string camera_extrinsics_path = camera_extrinsics_folder + "/" + calib_param.cams_name_vec[cam_index] + "_transform.pb.txt";
    file_io::writeExtrinsicToPbFile(cams_extrinsics_vec[cam_index], camera_extrinsics_path);
  }

  // show res
  for(size_t cam_index = 0; cam_index < calib_param.cams_name_vec.size(); ++cam_index) {
    Eigen::Matrix4d Tx_C_L = cams_extrinsics_vec[cam_index].inverse() * Tx_dr_L;
    for(size_t scene_index = 0; scene_index < calib_param.scene_num; ++scene_index) {
      cv::Mat res_img = show_tools::getProjectionImg(imgs_vec_[scene_index][cam_index], visual_pcd_vec_[scene_index],
        Tx_C_L, calib_param.camera_matrix_vec[cam_index], calib_param.dist_coeffs_vec[cam_index]);
      cv::imwrite(calib_param.result_path + "/" + calib_param.cams_name_vec[cam_index] + 
        "_sceneID_" + std::to_string(scene_index) + "_res.png", res_img);
    }
  }

  return 0;
}

void loadConfigFile(const std::string& calib_setting_path, DrLidarCalibParam& calib_param) {
  cv::FileStorage fSettings(calib_setting_path, cv::FileStorage::READ);
  if(!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << calib_setting_path << std::endl;
    exit(-1);
  }

  std::string image_path;
  fSettings["ImageFilesPath"] >> image_path;
  fSettings["CameraName"] >> calib_param.cams_name_vec;
  calib_param.scene_num = fSettings["SceneNum"];
  std::string img_file_extension;
  fSettings["img_file_extension"] >> img_file_extension;
  fSettings["LiDARFilesPath"] >> pcd_path_;
  std::string cameras_intrinsic_path;
  std::string cameras_extrinsic_path;
  fSettings["CamsIntrinsicPath"] >> cameras_intrinsic_path;
  fSettings["CamsExtrinsicPath"] >> cameras_extrinsic_path;
  calib_param.canny_threshold = fSettings["Canny.gray_threshold"];
  calib_param.rgb_edge_minLen = fSettings["Canny.len_threshold"];
  fSettings["UseAdaVoxel"] >> calib_param.use_ada_voxel;
  calib_param.voxel_size = fSettings["Voxel.size"];
  calib_param.eigen_ratio = fSettings["Voxel.eigen_ratio"];
  calib_param.plane_size_threshold = fSettings["Plane.min_points_size"];
  calib_param.ransac_dis_threshold = fSettings["Ransac.dis_threshold"];
  calib_param.p2line_dis_thred = fSettings["Edge.min_dis_threshold"];
  calib_param.theta_min = fSettings["Plane.normal_theta_min"];
  calib_param.theta_max = fSettings["Plane.normal_theta_max"];
  fSettings["ResultPath"] >> calib_param.result_path;
  if(access(calib_param.result_path.c_str(),0) != 0) {
    mkdir(calib_param.result_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
  std::string init_extrinsics_file;
  fSettings["init_extrinsics_file"] >> init_extrinsics_file;
  fSettings["save_lidar_extrinsic_name"] >> save_lidar_extrinsic_name_;

  #ifdef debug
  std::cout << "ImageFilesPath: " << image_path << std::endl;
  for(auto name: calib_param.cams_name_vec)
    std::cout << name << ", ";
  std::cout << std::endl;
  std::cout << "SceneNum: " << calib_param.scene_num << std::endl;
  std::cout << "img_file_extension: " << img_file_extension << std::endl;
  std::cout << "LiDARFilesPath: " << pcd_path_ << std::endl;
  std::cout << "CamsIntrinsicPath: " << cameras_intrinsic_path << std::endl;
  std::cout << "CamsExtrinsicPath: " << cameras_extrinsic_path << std::endl;
  std::cout << "Canny.gray_threshold: " << calib_param.canny_threshold << std::endl;
  std::cout << "Canny.len_threshold: " << calib_param.rgb_edge_minLen << std::endl;
  std::cout << "UseAdaVoxel: " << calib_param.use_ada_voxel << std::endl;
  std::cout << "Voxel.size: " << calib_param.voxel_size << std::endl;
  std::cout << "Voxel.eigen_ratio: " << calib_param.eigen_ratio << std::endl;
  std::cout << "Plane.min_points_size: " << calib_param.plane_size_threshold << std::endl;
  std::cout << "Ransac.dis_threshold: " << calib_param.ransac_dis_threshold << std::endl;
  std::cout << "Edge.min_dis_threshold: " << calib_param.p2line_dis_thred << std::endl;
  std::cout << "Plane.normal_theta_min: " << calib_param.theta_min << std::endl;
  std::cout << "Plane.normal_theta_max: " << calib_param.theta_max << std::endl;
  std::cout << "ResultPath: " << calib_param.result_path << std::endl;
  std::cout << "init_extrinsics_file: " << init_extrinsics_file << std::endl;
  std::cout << "save_lidar_extrinsic_name: " << save_lidar_extrinsic_name_ << std::endl;
  #endif

  int cam_num = calib_param.cams_name_vec.size();
  imgs_vec_.resize(calib_param.scene_num);
  calib_param.cams_model_vec.resize(cam_num);
  calib_param.camera_matrix_vec.resize(cam_num);
  calib_param.dist_coeffs_vec.resize(cam_num);
  calib_param.camera_extrinsics_vec.resize(cam_num);
  for(size_t cam_index = 0; cam_index < cam_num; ++cam_index) {
    std::string cam_intrinsics_path = cameras_intrinsic_path + "/" + calib_param.cams_name_vec[cam_index] + "_param.xml";
    int img_height, img_width;
    std::string camera_model;
    file_io::readCamInFromXmlFile(cam_intrinsics_path, camera_model, calib_param.camera_matrix_vec[cam_index],
    calib_param.dist_coeffs_vec[cam_index], img_height, img_width);
    if(camera_model == "FISHEYE")
      calib_param.cams_model_vec[cam_index] = CameraModel::Fisheye;
    
    std::string cam_extrinsics_path = cameras_extrinsic_path + "/" + calib_param.cams_name_vec[cam_index] + "_transform.pb.txt";
    file_io::readExtrinsicFromPbFile(cam_extrinsics_path, calib_param.camera_extrinsics_vec[cam_index]);
    
    std::string cam_path = image_path + "/" + calib_param.cams_name_vec[cam_index];
    for(size_t scene_index = 0; scene_index < calib_param.scene_num; ++scene_index) {
      std::string img_path = cam_path + "/" + std::to_string(scene_index) + "." + img_file_extension;
      cv::Mat raw_img = cv::imread(img_path);
      imgs_vec_[scene_index].emplace_back(raw_img);
    }
  }

  visual_pcd_vec_.resize(calib_param.scene_num);
  for(size_t scene_index = 0; scene_index < calib_param.scene_num; ++scene_index) {
    std::string visual_points_path = pcd_path_ + "/" + std::to_string(scene_index) + "_visual.pcd"; 
    visual_pcd_vec_[scene_index] = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(visual_points_path, *visual_pcd_vec_[scene_index]);
  }

  file_io::readExtrinsicFromPbFile(init_extrinsics_file, init_Tx_dr_L_);
  #ifdef debug
  Eigen::Quaterniond init_q_dr_l = Eigen::Quaterniond(init_Tx_dr_L_.block<3, 3>(0, 0));
  std::cout << "init_Tx_dr_L: " << init_q_dr_l.coeffs().transpose() << std::endl 
            << init_Tx_dr_L_.block<3, 1>(0, 3).transpose() << std::endl;
  #endif
}
