#include <pcl/io/pcd_io.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "multi_lidars_calib.h"
#include "dr_lidar_calib.h"
#include "file_io.h"
#include "extract_lidar_feature.h"
#include "exrtace_image_feature.h"
#include "match_features.h"
#include "floor_plane_constriant.h"
#include "show_tools.h"
#include "lidar_compensation.h"

// #define debug
// #define test

int ref_num_;
int step_;
std::string pcd_path_;
std::queue<std::string> base_pcds_name_que_;
StampedPoseVectorPtr dr_poses_ptr_;
std::string image_paths_;
std::string cameras_intrinsic_path_;
std::string cameras_extrinsic_path_;
std::string around_cam_stamp_file_;
Eigen::Matrix4d init_Tx_dr_L_;

std::string save_lidar_extrinsic_name_;
void loadConfigFile(const std::string& calib_setting_path, multi_lidars_calib::Lidar& base_lidar, 
  dr_lidar_calib::DrLidarCalibParam& calib_param);
void getImg(const std::string& img_path, const int& image_index, cv::Mat& img); 
std::vector<cv::Mat> getSyncImgs(const dr_lidar_calib::DrLidarCalibParam& calib_param,
  const double& lidar_stamp);

int main(int argc, char** argv) {
  if ( argc != 2 ) {
    std::cout<<"Usage: test_main path_to_config_file"<< std::endl;
    return -1;
  }
  const std::string calib_setting_path = std::string(argv[1]);
  multi_lidars_calib::Lidar base_lidar;
  dr_lidar_calib::DrLidarCalibParam calib_param;
  loadConfigFile(calib_setting_path, base_lidar, calib_param);
  #ifdef debug
  std::cout << "base_pcds_name_que: " << base_pcds_name_que_.size() << std::endl;
  #endif

  std::unique_ptr<dr_lidar_calib::DrLidarCalib> dr_lidar_calib = 
      std::make_unique<dr_lidar_calib::DrLidarCalib>(calib_param); 
  std::vector<std::vector<cv::Mat>> imgs_vec;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> visual_pcd_vec;
  if(ref_num_ > 0) {
    // todo: multi lidars calib
  }
  else {
    std::unique_ptr<multi_lidars_calib::MultiLidarsCalib> multi_lidars_calib = 
      std::make_unique<multi_lidars_calib::MultiLidarsCalib>(base_lidar, step_);    

    TimesVector time_pairs;
    if (calib_param.scene_num > 0) {
      time_pairs = multi_lidars_calib->getCutTimepairs(calib_param.scene_num, dr_poses_ptr_);
      #ifdef debug
      for(auto time: time_pairs)
        std::cout << "time: " << time.first << ", " << time.second << std::endl;
      #endif
    }

    int scene_index = 0;
    StampedPoseVectorPtr sample_pcd_poses_ptr = std::make_shared<StampedPoseVector>();
    StampedPcdVectorPtr sample_pcds_ptr = std::make_shared<StampedPcdVector>();
    while(!base_pcds_name_que_.empty()) {
      std::string stamp_pcd_path = base_pcds_name_que_.front();
      int start = stamp_pcd_path.rfind('/') + 1;
      int end = stamp_pcd_path.rfind('.');
      std::string tmp_str;
      tmp_str = std::string(stamp_pcd_path.substr(start, end - start));
      double stamp = std::stod(tmp_str);
      pcl::PointCloud<pcl::PointXYZI>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::io::loadPCDFile(stamp_pcd_path, *pcd);
      
      Eigen::Matrix4d pose;
      pcl::PointCloud<pcl::PointXYZI>::Ptr out_pcd = 
        pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
      if(multi_lidars_calib->processBaseLidar(base_lidar, stamp, pcd, pose, out_pcd)) {
        if (calib_param.scene_num > 0 && scene_index < calib_param.scene_num) {
          if(stamp >= time_pairs[scene_index].first && stamp <= time_pairs[scene_index].second) {
            sample_pcd_poses_ptr->emplace_back(StampedPose(stamp, pose));
            sample_pcds_ptr->emplace_back(StampedPcd(stamp, out_pcd));  
          }
          if(stamp > time_pairs[scene_index].second) {
            // process sample pcd and pose
            StampedPcd stamp_map;
            StampedPcd stamp_visual;
            multi_lidars_calib->runBaseLidar(sample_pcds_ptr, sample_pcd_poses_ptr, stamp_map, stamp_visual);
            #ifdef test
            std::cout << "sample_pcds_ptr size: " << sample_pcds_ptr->size() << std::endl;
            std::cout << "sample_pcd_poses_ptr size: " << sample_pcd_poses_ptr->size() << std::endl;
            std::string pose_path = "/home/wd/datasets/3/" + std::to_string(scene_index) + "_pose.txt";
            file_io::writeStampPoseToFile(sample_pcd_poses_ptr, pose_path);
            std::string map_pcd_path = "/home/wd/datasets/3/" + std::to_string(scene_index) + "_map.pcd";
            pcl::io::savePCDFile(map_pcd_path, *stamp_map.second);
            std::string visual_pcd_path = "/home/wd/datasets/3/" + std::to_string(scene_index) + "_visual.pcd";
            pcl::io::savePCDFile(visual_pcd_path, *stamp_visual.second);
            #endif
            
            sample_pcd_poses_ptr->clear();
            sample_pcds_ptr->clear();

            visual_pcd_vec.emplace_back(stamp_visual.second);
            dr_lidar_calib->processLidar(stamp_map.second);
            std::vector<cv::Mat> img_vec = getSyncImgs(calib_param, stamp_visual.first);
            imgs_vec.emplace_back(img_vec);
            ++scene_index;
          }
        }
      }
      base_pcds_name_que_.pop();
    }
    #ifdef debug
    std::cout << "base lidar pose traj size: " << base_lidar.lidar_poses_ptr->size() << std::endl;
    // todo: save lidar pose to check
    #endif

    // preCameras(calib_param, imgs_vec);
    #ifdef test
    std::cout << "imgs_vec size: " << imgs_vec.size() << std::endl;  
    for(auto imgs: imgs_vec) {
      for(auto img: imgs) {
        cv::imshow("img", img);
        cv::waitKey(0);
      }
    }
    #endif

    init_Tx_dr_L_ =
      multi_lidars_calib->estimateInitExtrinsics(dr_poses_ptr_, base_lidar.lidar_poses_ptr, base_lidar.tz);
  }

  #ifdef debug
  // show init 
  for(size_t cam_index = 0; cam_index < calib_param.cams_name_vec.size(); ++cam_index) {
    Eigen::Matrix4d Tx_C_L = calib_param.camera_extrinsics_vec[cam_index].inverse() * init_Tx_dr_L_;
    for(size_t scene_index = 0; scene_index < calib_param.scene_num; ++scene_index) {
      cv::Mat init_img = show_tools::getProjectionImg(imgs_vec[scene_index][cam_index], visual_pcd_vec[scene_index],
        Tx_C_L, calib_param.camera_matrix_vec[cam_index], calib_param.dist_coeffs_vec[cam_index]);
      cv::imwrite(calib_param.result_path + "/" + calib_param.cams_name_vec[cam_index] + 
        "_sceneID_" + std::to_string(scene_index) + "_init.png", init_img);
    }
  }
  #endif

  Eigen::Matrix4d Tx_dr_L;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> cams_extrinsics_vec;
  dr_lidar_calib->run(init_Tx_dr_L_, visual_pcd_vec, imgs_vec, Tx_dr_L, cams_extrinsics_vec);

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
      cv::Mat res_img = show_tools::getProjectionImg(imgs_vec[scene_index][cam_index], visual_pcd_vec[scene_index],
        Tx_C_L, calib_param.camera_matrix_vec[cam_index], calib_param.dist_coeffs_vec[cam_index]);
      cv::imwrite(calib_param.result_path + "/" + calib_param.cams_name_vec[cam_index] + 
        "_sceneID_" + std::to_string(scene_index) + "_res.png", res_img);
    }
  }

  return 0;
}

void loadConfigFile(const std::string& calib_setting_path, multi_lidars_calib::Lidar& base_lidar, 
  dr_lidar_calib::DrLidarCalibParam& calib_param) {
  cv::FileStorage fSettings(calib_setting_path, cv::FileStorage::READ);
  if(!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << calib_setting_path << std::endl;
    exit(-1);
  }

  std::string base_pcds_file;
  fSettings["BaseLidar"]["pcds_file"] >> base_pcds_file;
  fSettings["BaseLidar"]["max_dis"] >> base_lidar.max_dis;
  fSettings["BaseLidar"]["min_dis"] >> base_lidar.min_dis;
  std::string base_lidar_type;
  fSettings["BaseLidar"]["lidar_type"] >> base_lidar_type;
  if(base_lidar_type == "hesai64") {
    base_lidar.lidar_type = LidarType::Hesai64;
  }
  else if(base_lidar_type == "bp32") {
    base_lidar.lidar_type = LidarType::Bp32;
  }
  else if(base_lidar_type == "helios32") {
    base_lidar.lidar_type = LidarType::Helios32;
  }
  else if(base_lidar_type == "ft120") {
    base_lidar.lidar_type = LidarType::Ft120;
  }
  else if(base_lidar_type == "yijing") {
    base_lidar.lidar_type = LidarType::Yijing;
  }
  else {
    std::cerr << base_lidar_type << " cannot support" << std::endl;
    exit(-1);
  }
  fSettings["BaseLidar"]["tz"] >> base_lidar.tz;
  fSettings["RefNum"] >> ref_num_;
  fSettings["Step"] >> step_;
  std::string dr_pose_file;
  fSettings["DrPosesFile"] >> dr_pose_file;
  calib_param.scene_num = fSettings["SceneNum"];

  fSettings["ImageFilesPath"] >> image_paths_;
  fSettings["CameraName"] >> calib_param.cams_name_vec;
  
  fSettings["CamsIntrinsicPath"] >> cameras_intrinsic_path_;
  fSettings["CamsExtrinsicPath"] >> cameras_extrinsic_path_;
  fSettings["AroudCameraStampFile"] >> around_cam_stamp_file_;
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

  fSettings["save_lidar_extrinsic_name"] >> save_lidar_extrinsic_name_;

  #ifdef debug
  std::cout << "BaseLidar/pcds_file: " << base_pcds_file << std::endl;
  std::cout << "BaseLidar/max_dis: " << base_lidar.max_dis << std::endl;
  std::cout << "BaseLidar/min_dis: " << base_lidar.min_dis << std::endl;
  std::cout << "BaseLidar/lidar_type: " << base_lidar_type << std::endl;
  std::cout << "BaseLidar/tz: " << base_lidar.tz << std::endl;
  std::cout << "RefNum: " << ref_num_ << std::endl;
  std::cout << "Step: " << step_ << std::endl;
  std::cout << "DrPosesFile: " << dr_pose_file << std::endl;
  std::cout << "SceneNum: " << calib_param.scene_num << std::endl;
  std::cout << "ImageFilesPath: " << image_paths_ << std::endl;
  for(auto name: calib_param.cams_name_vec)
    std::cout << name << ", ";
  std::cout << std::endl;
  
  std::cout << "CamsIntrinsicPath: " << cameras_intrinsic_path_ << std::endl;
  std::cout << "CamsExtrinsicPath: " << cameras_extrinsic_path_ << std::endl;
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
  std::cout << "save_lidar_extrinsic_name: " << save_lidar_extrinsic_name_ << std::endl;
  #endif

  file_io::loadPcdFilePath(base_pcds_file, base_pcds_name_que_);
  dr_poses_ptr_ = std::make_unique<StampedPoseVector>();
  file_io::readStampPoseFromFile(dr_pose_file, dr_poses_ptr_);
  #ifdef debug
  std::cout << "dr pose traj size: " << dr_poses_ptr_->size() << std::endl;
  #endif

  int cam_num = calib_param.cams_name_vec.size();
  // calib_param.images.resize(cam_num);
  calib_param.cams_model_vec.resize(cam_num);
  calib_param.camera_matrix_vec.resize(cam_num);
  calib_param.dist_coeffs_vec.resize(cam_num);
  calib_param.camera_extrinsics_vec.resize(cam_num);

  for(size_t cam_index = 0; cam_index < cam_num; ++cam_index) {
    std::string cam_intrinsics_path = cameras_intrinsic_path_ + "/" + calib_param.cams_name_vec[cam_index] + "_param.xml";
    int img_height, img_width;
    std::string camera_model;
    file_io::readCamInFromXmlFile(cam_intrinsics_path, camera_model, calib_param.camera_matrix_vec[cam_index],
    calib_param.dist_coeffs_vec[cam_index], img_height, img_width);
    if(camera_model == "FISHEYE")
      calib_param.cams_model_vec[cam_index] = CameraModel::Fisheye;
    
    std::string cam_extrinsics_path = cameras_extrinsic_path_ + "/" + calib_param.cams_name_vec[cam_index] + "_transform.pb.txt";
    file_io::readExtrinsicFromPbFile(cam_extrinsics_path, calib_param.camera_extrinsics_vec[cam_index]);
  }
}

void getImg(const std::string& img_path, const int& image_index, cv::Mat& img) {
  // cv::VideoCapture capture(video_path);
  // if (!capture.isOpened()) {
  //   std::cerr << "# ERROR: Open video failed!" << std::endl;
  //   return;
  // }

  // int image_num = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  std::vector<cv::String> files;
  cv::glob(img_path, files);
  sort(files.begin(), files.end(), [&files](cv::String a, cv::String b){
    return(std::stod(cv::String(a.substr(a.rfind('/') + 1, a.rfind('.') - a.rfind('/') + 1))) 
    <= std::stod(cv::String(b.substr(b.rfind('/') + 1, b.rfind('.') - b.rfind('/') + 1))));}); 

  for(int i = 0; i < files.size(); ++i) {
    if(i == image_index) {
      cv::Mat image;
      image = cv::imread(files[i]);
      image.copyTo(img);
      break;
    }
  }
}

std::vector<cv::Mat> getSyncImgs(const dr_lidar_calib::DrLidarCalibParam& calib_param,
  const double& lidar_stamp) {
  std::vector<cv::Mat> img_vec;
  // just aroud camera
  std::vector<double> around_timestamp_vec;
  int around_camera_start_index = -1;
  if(file_io::read_timestamp_info(around_cam_stamp_file_, around_timestamp_vec)) {
    around_camera_start_index = file_io::getSyncTimestampIdex(around_timestamp_vec, lidar_stamp, dr_poses_ptr_);
  }

  for(size_t cam_index = 0; cam_index < calib_param.cams_name_vec.size() ; ++cam_index) {
    std::string cam_path = image_paths_ + "/" + calib_param.cams_name_vec[cam_index];
    cv::Mat img;
    getImg(cam_path, around_camera_start_index, img);
    img_vec.emplace_back(img);
  }

  return img_vec;
}