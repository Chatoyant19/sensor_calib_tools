#include "dr_lidar_calib.h"

#include <sys/stat.h>
#include <unistd.h>

#include <iostream>

#include "cost_function.hpp"
#include "exrtace_image_feature.h"
#include "extract_lidar_feature.h"
#include "extract_lidar_feature.hpp"
#include "floor_plane_constriant.h"
#include "match_features.h"
#include "file_io.h"

namespace dr_lidar_calib {

DrLidarCalib::DrLidarCalib(const DrLidarCalibParam &param) {
  param_ = param;

  if (param_.use_ada_voxel) {
    extract_lidar_feature_ = std::make_unique<ExtractLidarFeature>(
        param_.voxel_size, param_.eigen_ratio, param_.p2line_dis_thred,
        param_.theta_min, param_.theta_max);
  } else {
    extract_lidar_feature_ = std::make_unique<ExtractLidarFeature>(
        param_.voxel_size, param_.ransac_dis_threshold,
        param_.plane_size_threshold, param_.p2line_dis_thred, param_.theta_min,
        param_.theta_max);
  }
}

void DrLidarCalib::init(const Eigen::Matrix4d &init_Tx_DR_Lidar,
                        const std::vector<std::vector<cv::Mat>> &imgs_vec) {
  initLidar(init_Tx_DR_Lidar);

  initCameras(imgs_vec);
}

void DrLidarCalib::initLidar(const Eigen::Matrix4d &init_Tx_DR_Lidar) {
  lidar_.update_TxDL(init_Tx_DR_Lidar);
}

void DrLidarCalib::initCameras(
    const std::vector<std::vector<cv::Mat>> &imgs_vec) {
  cams_.resize(param_.cams_name_vec.size());
  std::cout << "init " << cams_.size() << " cameras." << std::endl;

  std::unique_ptr<ExtractImageFeature> extract_image_feature =
      std::make_unique<ExtractImageFeature>(param_.canny_threshold,
                                            param_.rgb_edge_minLen);

  for (size_t cam_index = 0; cam_index < cams_.size(); ++cam_index) {
    cams_[cam_index].cam_name_ = param_.cams_name_vec[cam_index];
    cams_[cam_index].camera_model_ = param_.cams_model_vec[cam_index];
    cams_[cam_index].camera_matrix_ = param_.camera_matrix_vec[cam_index];
    cams_[cam_index].dist_coeffs_ = param_.dist_coeffs_vec[cam_index];
    cams_[cam_index].Tx_dr_C_ = param_.camera_extrinsics_vec[cam_index];
    cams_[cam_index].Tx_C_L_ =
        cams_[cam_index].Tx_dr_C_.inverse() * lidar_.Tx_dr_L_;
    cv::Mat new_K = cams_[cam_index].camera_matrix_;
    cv::Mat map1, map2;
    for (size_t scene_index = 0; scene_index < param_.scene_num;
         ++scene_index) {
      cv::Mat raw_img = imgs_vec[scene_index][cam_index];
      cv::Mat undistort_img;
      if (cams_[cam_index].camera_model_ == CameraModel::Fisheye) {
        cv::fisheye::initUndistortRectifyMap(
            cams_[cam_index].camera_matrix_, cams_[cam_index].dist_coeffs_,
            cv::Mat(), new_K, (cv::Size(raw_img.cols, raw_img.rows)), CV_16SC2,
            map1, map2);
        cv::remap(raw_img, undistort_img, map1, map2, cv::INTER_LINEAR);
      }

      cams_[cam_index].width_ = undistort_img.cols;
      cams_[cam_index].height_ = undistort_img.rows;

      // extrace images line features
      pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_clouds =
          pcl::PointCloud<pcl::PointXYZ>::Ptr(
              new pcl::PointCloud<pcl::PointXYZ>);
      extract_image_feature->getEdgeFeatures(undistort_img, rgb_egde_clouds);
      cams_[cam_index].rgb_edge_clouds_.emplace_back(rgb_egde_clouds);
    }
  }
}

// extract lidar line features
void DrLidarCalib::processLidar(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_lidar_cloud) {
  std::cerr << "========== processLidar-begin ==========" << std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr line_clouds =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  if (param_.use_ada_voxel) {
    extract_lidar_feature_->getEdgeFeaturesByAdaVoxel(input_lidar_cloud,
                                                      line_clouds);
  } else {
    extract_lidar_feature_->getEdgeFeatures(input_lidar_cloud, line_clouds);
  }
  std::cout << "line_clouds size: " << line_clouds->size() << std::endl;

  lidar_.plane_line_cloud_vec_.emplace_back(line_clouds);
  std::cerr << "========== processLidar-end ==========" << std::endl;
}

void DrLidarCalib::run(
    const Eigen::Matrix4d &init_Tx_DR_Lidar,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &visual_pcd_vec,
    const std::vector<std::vector<cv::Mat>> &imgs_vec,
    Eigen::Matrix4d &Tx_dr_L,
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &
    cams_extrinsics_vec,
    bool has_CAD_prior) {
  std::cerr << "===== Estimate DR-Lidar =====" << std::endl;

  init(init_Tx_DR_Lidar, imgs_vec);

  std::cerr << "planar constraints - 1" << std::endl;
  // floor plane constraint
  Eigen::Matrix4d new_Tx_DR_Lidar = Eigen::Matrix4d::Identity();
  // todo: compare results
  // todo: 考虑地面约束失败的情况
  std::unique_ptr<FloorPlaneConstriant> floor_plane_constraint =
      std::make_unique<FloorPlaneConstriant>();
  // 根据平面信息优化Tx_DR_Lidar，并更新Tx_Camera_Lidar[根据Tx_Camera_DR * Tx_DR_Lidar]
  if (floor_plane_constraint->addFloorConstraint(
      visual_pcd_vec[0], lidar_.Tx_dr_L_, new_Tx_DR_Lidar)) {
    // if (floor_plane_constraint->addFloorConstraintRac(visual_pcd_vec[0],
    // lidar_.Tx_dr_L_, new_Tx_DR_Lidar)) {
    lidar_.update_TxDL(new_Tx_DR_Lidar);

    for (int cam_index = 0; cam_index < cams_.size(); ++cam_index) {
      cams_[cam_index].Tx_C_L_ =
          cams_[cam_index].Tx_dr_C_.inverse() * lidar_.Tx_dr_L_;
    }
  }
  std::cout << "Tx_DR_Lidar_1: " << new_Tx_DR_Lidar << std::endl;
  std::string Tx_DR_Lidar_1_file = param_.result_path + "/" + "Tx_DR_Lidar_1.pb.txt";
  file_io::writeExtrinsicToPbFile(new_Tx_DR_Lidar, Tx_DR_Lidar_1_file);

  assert(lidar_.plane_line_cloud_vec_.size() == param_.scene_num);
  std::cout << "successfully load " << param_.scene_num << " point clouds."
            << std::endl;
  for (size_t cam_index = 0; cam_index < cams_.size(); ++cam_index)
    assert(cams_[cam_index].rgb_edge_clouds_.size() == param_.scene_num);
  std::cout << "successfully load " << param_.scene_num * cams_.size()
            << " images." << std::endl;

  // match features
  int iter = 0;
  int low_dis_threshold = 4;
  int high_dis_threshold = 30;
  // Tx_dr_L_, Tx_dr_C_
  for (int dis_threshold = high_dis_threshold;
       dis_threshold > low_dis_threshold; dis_threshold -= 1) {
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
      // prior
      if (has_CAD_prior) {
        auto translation_prior_cost = new ceres::AutoDiffCostFunction<TranslationError, 3, 3>(
            new TranslationError(m_t, 1e10)
        );
        problem.AddResidualBlock(translation_prior_cost, nullptr, m_t.data());
      }

      int total_vpnp_size = 0;
      for (int cam_index = 0; cam_index < cams_.size(); ++cam_index) {
        Eigen::Matrix3d inner;
        inner << cams_[cam_index].camera_matrix_.at<double>(0, 0), 0.0,
            cams_[cam_index].camera_matrix_.at<double>(0, 2), 0.0,
            cams_[cam_index].camera_matrix_.at<double>(1, 1),
            cams_[cam_index].camera_matrix_.at<double>(1, 2), 0.0, 0.0, 1.0;
        Eigen::Vector4d distor;
        distor << 0.0, 0.0, 0.0, 0.0;
        Eigen::Matrix3d R_dr_C = cams_[cam_index].Tx_dr_C_.block<3, 3>(0, 0);
        Eigen::Vector3d t_dr_c = cams_[cam_index].Tx_dr_C_.block<3, 1>(0, 3);

        for (int scene_index = 0; scene_index < param_.scene_num;
             ++scene_index) {
          std::vector<VPnPData> vpnp_list;
          cv::Mat residual_img;
          std::unique_ptr<MatchFeatures> match_features =
              std::make_unique<MatchFeatures>(param_.show_residual);
          match_features->buildVPnp(
              cams_[cam_index].camera_matrix_, cams_[cam_index].width_,
              cams_[cam_index].height_,
              cams_[cam_index].rgb_edge_clouds_[scene_index],
              lidar_.plane_line_cloud_vec_[scene_index],
              cams_[cam_index].Tx_C_L_, dis_threshold, vpnp_list, residual_img);
          if (vpnp_list.size() == 0) {
            std::cout << "not enough measurement!" << std::endl;
            continue;
          }
          if (param_.show_residual) {
            std::string save_residual_path = param_.result_path + "/residual";
            if (access(save_residual_path.c_str(), 0) != 0) {
              mkdir(save_residual_path.c_str(),
                    S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }

            if (cnt == 0 && dis_threshold == high_dis_threshold) {
              cv::imwrite(save_residual_path + "/" +
                              cams_[cam_index].cam_name_ + "_sceneID_" +
                              std::to_string(scene_index) +
                              "_init_residual.jpg",
                          residual_img);
            }

            if (cnt == 4 && dis_threshold == low_dis_threshold + 1) {
              cv::imwrite(save_residual_path + "/" +
                              cams_[cam_index].cam_name_ + "_sceneID_" +
                              std::to_string(scene_index) +
                              "_result_residual.jpg",
                          residual_img);
            }
          }

          total_vpnp_size += vpnp_list.size();

          ceres::CostFunction *cost_function;
          for (auto val : vpnp_list) {
            cost_function =
                vpnp_calib_pin_auto::Create(val, R_dr_C, t_dr_c, inner, distor);
            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.3),
                                     ext, ext + 4);
          }
        }
      }

      std::cout << "Iteration: " << iter++ << ", Dis: " << dis_threshold
                << ", PnP size: " << total_vpnp_size << std::endl;
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
      // std::cout << summary.BriefReport() << std::endl;

      lidar_.Tx_dr_L_.block<3, 3>(0, 0) = m_q.toRotationMatrix();
      lidar_.Tx_dr_L_.block<3, 1>(0, 3) = m_t;
      lidar_.update_TxDL(lidar_.Tx_dr_L_);

      for (size_t cam_index = 0; cam_index < cams_.size(); ++cam_index) {
        Eigen::Matrix4d Tx_C_L =
            cams_[cam_index].Tx_dr_C_.inverse() * lidar_.Tx_dr_L_;
        cams_[cam_index].update_TxCL(Tx_C_L);
      }
    }
  }
  std::cout << "Tx_DR_Lidar_2: " << lidar_.Tx_dr_L_ << std::endl;
  std::string Tx_DR_Lidar_2_file = param_.result_path + "/" + "Tx_DR_Lidar_2.pb.txt";
  file_io::writeExtrinsicToPbFile(lidar_.Tx_dr_L_, Tx_DR_Lidar_2_file);

  std::cerr << "planar constraints - 2" << std::endl;
  // floor plane constraints again
  Eigen::Matrix4d T_dr_L_new = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_dr_pr = Eigen::Matrix4d::Identity();
  if (floor_plane_constraint->addFloorConstraint(visual_pcd_vec[0],
                                                 lidar_.Tx_dr_L_, T_dr_L_new)) {
    T_dr_pr = T_dr_L_new * (lidar_.Tx_dr_L_.inverse());
    lidar_.update_TxDL(T_dr_L_new);
  }
  std::cout << "Tx_DR_Lidar_3: " << T_dr_L_new << std::endl;
  std::string Tx_DR_Lidar_3_file = param_.result_path + "/" + "Tx_DR_Lidar_3.pb.txt";
  file_io::writeExtrinsicToPbFile(T_dr_L_new, Tx_DR_Lidar_3_file);

  // update results
  Tx_dr_L = lidar_.Tx_dr_L_;
  cams_extrinsics_vec.resize(cams_.size());
  for (size_t cam_index = 0; cam_index < cams_.size(); ++cam_index) {
    cams_extrinsics_vec[cam_index] = T_dr_pr * cams_[cam_index].Tx_dr_C_;
  }

  std::cerr << "===== Estimate DR-Lidar Done! =====" << std::endl;
}

}  // namespace dr_lidar_calib
