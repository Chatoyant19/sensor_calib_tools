#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include "dr_lidar_calib.h"
#include "exrtace_image_feature.h"
#include "extract_lidar_feature.h"
#include "floor_plane_constriant.h"
#include "match_features.h"
#include "cost_function.hpp"

// DrLidarCalib::DrLidarCalib() {

// }

void DrLidarCalib::init() {
  initLidar();

  initCameras();

  floor_plane_constraint_ = std::make_unique<FloorPlaneConstriant>();

  match_features_ = std::make_unique<MatchFeatures>(param_.show_residual);
}

void DrLidarCalib::initLidar() {
  lidar_.update_TDL(param_.init_Tx_dr_L);

  assert(param_.map_pcd_vec_.size() == param_.scene_num);
  std::cout << "successfule load " << param_.scene_num << "point clouds." << std::endl;
}

void DrLidarCalib::initCameras() {
  cams_.resize(param_.cams_name_vec.size());
  std::cout << "init " << cams_.size() << " cameras: ";

  assert(cams_.size() == param_.images.size());
  int img_cnt = 0;
  for(size_t cam_index = 0; cam_index < cams_.size(); ++cam_index) {
    cams_[cam_index].cam_name_ = param_.cams_name_vec[cam_index];
    cams_[cam_index].camera_model_ = param_.cams_model_vec[cam_index];
    cams_[cam_index].camera_matrix_ = param_.camera_matrix_vec[cam_index];
    cams_[cam_index].dist_coeffs_ = param_.dist_coeffs_vec[cam_index];
 
    assert(param_.scene_num == param_.images[cam_index].size());
    cv::Mat new_K = cams_[cam_index].camera_matrix_;
    cv::Mat map1, map2;
    for(size_t scene_index = 0; scene_index < param_.scene_num; ++scene_index) {
      cams_[cam_index].raw_imgs.emplace_back(param_.images[cam_index][scene_index]);

      cv::Mat raw_img = param_.images[cam_index][scene_index];
      cv::Mat undistort_img;
      if(cams_[cam_index].camera_model_ == "fisheye") {
        cv::fisheye::initUndistortRectifyMap(cams_[cam_index].camera_matrix_, cams_[cam_index].dist_coeffs_, 
          cv::Mat(), new_K, (cv::Size(raw_img.cols,raw_img.rows)), CV_16SC2, map1, map2);
        cv::remap(raw_img, undistort_img, map1, map2, cv::INTER_LINEAR);
        cams_[cam_index].rgb_imgs.emplace_back(undistort_img);

        ++img_cnt;
      }

      cams_[cam_index].width_ = cams_[cam_index].rgb_imgs[0].cols;
      cams_[cam_index].height_ = cams_[cam_index].rgb_imgs[0].rows;
    }
  }
  std::cout << "successful load " << img_cnt << " images." << std::endl;
    

  extract_image_feature_ = 
    std::make_unique<ExtractImageFeature>(param_.canny_threshold, param_.rgb_edge_minLen);
}



void DrLidarCalib::run(Eigen::Matrix4d& Tx_dr_L,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& cams_extrinsics_vec) {
  // floor plane constraint
  Eigen::Matrix4d Tx_dr_L_new = Eigen::Matrix4d::Identity();
  // todo: compare results
  floor_plane_constraint_->addFloorConstriant(param_.visual_pcd_vec_[0], lidar_.Tx_dr_L_, Tx_dr_L_new);
  // floor_plane_constraint_->addFloorConstriantRac(param_.visual_pcd_vec_[0], lidar_.Tx_dr_L_, Tx_dr_L_new);
  lidar_.update_TDL(Tx_dr_L_new);

  for(int cam_index = 0; cam_index < cams_.size(); ++cam_index) {
    cams_[cam_index].Tx_C_L_ = cams_[cam_index].Tx_dr_C_.inverse() * lidar_.Tx_dr_L_;
  }

  // extract lidar line features
  lidar_.plane_line_cloud_vec_.resize(param_.scene_num);
  for(int scene_index = 0; scene_index < param_.scene_num; ++scene_index) {
    lidar_.plane_line_cloud_vec_[scene_index] = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>);
    if(param_.use_ada_voxel) {
      extract_lidar_feature_ = 
        std::make_unique<ExtractLidarFeature>(param_.voxel_size, param_.eigen_ratio, param_.p2line_dis_thred, 
                                              param_.theta_min, param_.theta_max);
      pcl::PointCloud<pcl::PointXYZI>::Ptr line_clouds = 
        pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
      extract_lidar_feature_->getEdgeFeaturesByAdaVoxel(param_.map_pcd_vec_[scene_index], line_clouds); 
      std::cout << "line_clouds size: " << line_clouds->size() << std::endl; 

      lidar_.plane_line_cloud_vec_[scene_index] = line_clouds;
    }
    else {
      extract_lidar_feature_ = 
      std::make_unique<ExtractLidarFeature>(param_.voxel_size, param_.ransac_dis_threshold, param_.plane_size_threshold, 
                                            param_.p2line_dis_thred, param_.theta_min, param_.theta_max);
      pcl::PointCloud<pcl::PointXYZI>::Ptr line_clouds = 
        pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
      extract_lidar_feature_->getEdgeFeatures(param_.map_pcd_vec_[scene_index], line_clouds);  
      std::cout << "line_clouds size: " << line_clouds->size() << std::endl; 

      lidar_.plane_line_cloud_vec_[scene_index] = line_clouds;
    }
  }

  // extrace images line features
  for(int cam_index = 0; cam_index < cams_.size(); ++cam_index) {
    cams_[cam_index].rgb_edge_clouds_.resize(param_.scene_num);
    for(int scene_index = 0; scene_index < param_.scene_num; ++scene_index) {
      cams_[cam_index].rgb_edge_clouds_[scene_index] = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
      
      pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_clouds = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
      extract_image_feature_->getEdgeFeatures(cams_[cam_index].rgb_imgs[scene_index], rgb_egde_clouds);
      cams_[cam_index].rgb_edge_clouds_[scene_index] =  rgb_egde_clouds;
    }
  }

  // match features
  int iter = 0;
  int low_dis_threshold = 4;
  int high_dis_threshold = 30;
  for (int dis_threshold = high_dis_threshold; dis_threshold > low_dis_threshold; dis_threshold -= 1) {
    for (int cnt = 0; cnt < 5; ++cnt) {
      Eigen::Matrix3d R_dr_L = lidar_.Tx_dr_L_.block<3, 3>(0, 0);
      Eigen::Vector3d t_dr_l = lidar_.Tx_dr_L_.block<3, 1>(0, 3);
      Eigen::Quaterniond q(R_dr_L);
      double ext[7];
      ext[0] = q.x();
      ext[1] = q.y();
      ext[2] = q.z();
      ext[3] = q.w();
      ext[4] = t_dr_l[0];
      ext[5] = t_dr_l[1];
      ext[6] = t_dr_l[2];
      Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
      Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);
      ceres::LocalParameterization *q_parameterization =
              new ceres::EigenQuaternionParameterization();
      ceres::Problem problem;
      problem.AddParameterBlock(ext, 4, q_parameterization);
      problem.AddParameterBlock(ext + 4, 3);

      int total_vpnp_size = 0;
      for(int cam_index = 0; cam_index < cams_.size(); ++cam_index) {
        Eigen::Matrix3d inner;
        inner << cams_[cam_index].camera_matrix_.at<double>(0, 0), 0.0, cams_[cam_index].camera_matrix_.at<double>(0, 2),
                 0.0, cams_[cam_index].camera_matrix_.at<double>(1, 1), cams_[cam_index].camera_matrix_.at<double>(1, 2),
                 0.0, 0.0, 1.0;
        Eigen::Vector4d distor;
        distor << 0.0, 0.0, 0.0, 0.0;
        Eigen::Matrix3d R_dr_C = cams_[cam_index].Tx_dr_C_.block<3, 3>(0, 0);
        Eigen::Vector3d t_dr_c = cams_[cam_index].Tx_dr_C_.block<3, 1>(0, 3);
        
        for(int scene_index = 0; scene_index < param_.scene_num; ++scene_index) {
          std::vector<VPnPData> vpnp_list;
          cv::Mat residual_img;
          match_features_->buildVPnp(cams_[cam_index].camera_matrix_, cams_[cam_index].width_, cams_[cam_index].height_,
                                     cams_[cam_index].rgb_edge_clouds_[scene_index], lidar_.plane_line_cloud_vec_[scene_index],
                                     cams_[cam_index].Tx_C_L_, dis_threshold, vpnp_list, residual_img);
          if(param_.show_residual) {
            std::string save_residual_path = param_.result_path + "/residual";
            if(access(save_residual_path.c_str(),0) != 0) {
              mkdir(save_residual_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }

            if(cnt == 0 && dis_threshold == high_dis_threshold) {
              cv::imwrite(save_residual_path + "/" + cams_[cam_index].cam_name_ + "_sceneID_" + 
                std::to_string(scene_index) + "_init_residual.jpg", residual_img);
            }

            if(cnt == 4 && dis_threshold == low_dis_threshold + 1) {
              cv::imwrite(save_residual_path + "/" + cams_[cam_index].cam_name_ + "_sceneID_" + 
               std::to_string(scene_index) + "_result_residual.jpg", residual_img);
            }
          }

          if(vpnp_list.size() == 0) {
            std::cout << "not enough measurement!" << std::endl;
              continue;
          }
          total_vpnp_size += vpnp_list.size();

          ceres::CostFunction *cost_function;
          for(auto val: vpnp_list)  {
            cost_function = vpnp_calib_pin_auto::Create(val, R_dr_C, t_dr_c, inner, distor);
            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.3), ext, ext + 4);
          }
        }
                 
      }

      std::cout << "Iteration:" << iter++ << " Dis:" << dis_threshold
                  << " pnp size: " << total_vpnp_size << std::endl;
      ceres::Solver::Options options;
      // options.preconditioner_type = ceres::JACOBI;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = true;
      options.trust_region_strategy_type = ceres::DOGLEG;
      // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options.max_num_iterations = 500;
      options.use_nonmonotonic_steps = true;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.BriefReport() << std::endl;

      lidar_.Tx_dr_L_.block<3, 3>(0, 0) = m_q.toRotationMatrix();
      lidar_.Tx_dr_L_.block<3, 1>(0, 3) = m_t;
      lidar_.update_TDL(lidar_.Tx_dr_L_);

      for (size_t cam_index = 0; cam_index < cams_.size(); ++cam_index) {
        Eigen::Matrix4d Tx_C_L = cams_[cam_index].Tx_dr_C_.inverse() * lidar_.Tx_dr_L_;
        cams_[cam_index].update_TxCL(Tx_C_L);
      }
    }
  }

  // floor plane constrint again
  Eigen::Matrix4d T_dr_L_new = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_dr_pr = Eigen::Matrix4d::Identity();
  floor_plane_constraint_->addFloorConstriant(param_.visual_pcd_vec_[0], lidar_.Tx_dr_L_, T_dr_L_new);
  T_dr_pr = T_dr_L_new * (lidar_.Tx_dr_L_.inverse());
  lidar_.update_TDL(T_dr_L_new);

  Tx_dr_L = lidar_.Tx_dr_L_;
  cams_extrinsics_vec.resize(cams_.size());
  for (size_t cam_index = 0; cam_index < cams_.size(); ++cam_index) {
    cams_extrinsics_vec[cam_index] = T_dr_pr * cams_[cam_index].Tx_dr_C_;
  }

}

