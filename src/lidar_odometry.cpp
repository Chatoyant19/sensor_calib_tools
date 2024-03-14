#include "lidar_odometry.h"
#include "pose_optimization.h"

LidarOdometry::LidarOdometry(const bool& use_compensation) {
  // todo: add compensate pcd
  use_compensation_ = use_compensation;

  pointcloud_filtered_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

  is_odom_inited_ = false;
    
  //init local map
  // laserCloudCornerMap_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
  laserCloudSurfMap_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

  odom_ = Eigen::Isometry3d::Identity();
  last_odom_ = Eigen::Isometry3d::Identity();
  optimization_count_=2;

  // downSizeFilterEdge_.setLeafSize(map_resolution_, map_resolution_, map_resolution_);
  downSizeFilterSurf_.setLeafSize(map_resolution_ * 2, map_resolution_ * 2, map_resolution_ * 2);

  //kd-tree
  kdtreeSurfMap_ = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZI>());
}

void LidarOdometry::runOdometry(const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd, 
    Eigen::Matrix4d& curr_pose) {
  curr_pose = Eigen::Matrix4d::Identity();
  addPointCloud(origin_pcd);   
  odomEstimation();
  Eigen::Quaterniond curr_q;
  Eigen::Vector3d curr_t;
  getPose(curr_q, curr_t); 
  curr_pose.block<3, 3>(0, 0) = curr_q.toRotationMatrix();
  curr_pose.block<3, 1>(0, 3) = curr_t; 
}

void LidarOdometry::addPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& origin_pcd) {
    pointcloud_filtered_ = origin_pcd;

    if(use_compensation_) {
      std::cout << "test: compensation?" << std::endl;
      // todo compensation pointcloud_filtered_ and output pointcloud_filtered_
    }
}

void LidarOdometry::odomEstimation() {
  if(is_odom_inited_ == false){
    initMapWithPoints(pointcloud_filtered_);
    is_odom_inited_ = true;
  }
  else {
    updatePointsToMap(pointcloud_filtered_);
  }
}

void LidarOdometry::getPose(Eigen::Quaterniond& curr_q, Eigen::Vector3d& curr_t) {
  curr_q = Eigen::Quaterniond(odom_.rotation());
  curr_t = Eigen::Vector3d(odom_.translation());
}

void LidarOdometry::initMapWithPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in) {
  *laserCloudSurfMap_ += *surf_in;
  optimization_count_ = 12;
}

void LidarOdometry::updatePointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in) {
  if(optimization_count_ > 2)
    optimization_count_--;

  Eigen::Isometry3d odom_prediction = odom_ * (last_odom_.inverse() * odom_);
  last_odom_ = odom_;
  odom_ = odom_prediction;  

  q_w_curr_ = Eigen::Quaterniond(odom_.rotation());
  t_w_curr_ = odom_.translation();

  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZI>());
  downSamplingToMap(surf_in,downsampledSurfCloud);

  if(laserCloudSurfMap_->points.size() > 50){
    kdtreeSurfMap_->setInputCloud(laserCloudSurfMap_);
    for (int iterCount = 0; iterCount < optimization_count_; iterCount++){
      ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);

      problem.AddParameterBlock(parameters_, 7, new PoseSE3Parameterization());

      addSurfCostFactor(downsampledSurfCloud, laserCloudSurfMap_, problem, loss_function);

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.max_num_iterations = 4;
      options.minimizer_progress_to_stdout = false;
      options.check_gradients = false;
      options.gradient_check_relative_precision = 1e-4;
      ceres::Solver::Summary summary;

      ceres::Solve(options, &problem, &summary);

    }
  }
  else {
    printf("not enough points in map to associate, map error");
  }
  odom_ = Eigen::Isometry3d::Identity();
  odom_.linear() = q_w_curr_.toRotationMatrix();
  odom_.translation() = t_w_curr_;
  addPointsToMap(downsampledSurfCloud);
}

void LidarOdometry::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in, 
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out){
    downSizeFilterSurf_.setInputCloud(surf_pc_in);
    downSizeFilterSurf_.filter(*surf_pc_out);       
}

void LidarOdometry::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, 
                                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in, 
                                      ceres::Problem& problem, ceres::LossFunction *loss_function){
  int surf_num=0;
  for (int i = 0; i < (int)pc_in->points.size(); i++) {
    pcl::PointXYZI point_temp;
    pointAssociateToMap(&(pc_in->points[i]), &point_temp);
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    kdtreeSurfMap_->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

    Eigen::Matrix<double, 5, 3> matA0;
    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
    if (pointSearchSqDis[4] < 1.0) {     
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
        matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
        matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
      }
      // find the norm of plane
      Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
      double negative_OA_dot_norm = 1 / norm.norm();
      norm.normalize();

      bool planeValid = true;
      for (int j = 0; j < 5; j++)
      {
          // if OX * n > 0.2, then plane is not fit well
          if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                   norm(1) * map_in->points[pointSearchInd[j]].y +
                   norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
          {
              planeValid = false;
              break;
          }
      }
      Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
      if (planeValid)
      {
          ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);    
          problem.AddResidualBlock(cost_function, loss_function, parameters_);
          surf_num++;
      }
    }

  }
  if(surf_num<20){
    printf("not enough correct points");
  }                                      
}

void LidarOdometry::pointAssociateToMap(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po) {
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr_ * point_curr + t_w_curr_;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
    //po->intensity = 1.0;
}

void LidarOdometry::addPointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledSurfCloud){
    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap_->push_back(point_temp);
    }
    
    double x_min = +odom_.translation().x()-100;
    double y_min = +odom_.translation().y()-100;
    double z_min = +odom_.translation().z()-100;
    double x_max = +odom_.translation().x()+100;
    double y_max = +odom_.translation().y()+100;
    double z_max = +odom_.translation().z()+100;
    
    //ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter_.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter_.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter_.setNegative(false);    

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZI>());
    cropBoxFilter_.setInputCloud(laserCloudSurfMap_);
    cropBoxFilter_.filter(*tmpSurf);
    downSizeFilterSurf_.setInputCloud(tmpSurf);
    downSizeFilterSurf_.filter(*laserCloudSurfMap_);
}