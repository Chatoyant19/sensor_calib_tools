#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/io/pcd_io.h>
#include "floor_plane_constriant.h"
#include "extract_lidar_feature.hpp"
#include "eigen_types.hpp"
// #define test

bool FloorPlaneConstriant::addFloorConstriant(const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_pcd, 
                                              const Eigen::Matrix4d& Tx_dr_L,
                                              Eigen::Matrix4d& update_Tx_dr_L) {
  pcl::PointCloud<pcl::PointXYZI> transformed_pcd;
  pcl::transformPointCloud(*raw_pcd, transformed_pcd, Tx_dr_L);  

  pcl::PointCloud<pcl::PointXYZI> floor_pcd;
  filteredPcd(transformed_pcd, floor_pcd);  
  // #ifdef test
  // std::string path = "/home/wd/datasets/beijing/2/floor_pcd.pcd";
  // pcl::io::savePCDFile(path, floor_pcd);
  // #endif
  assert(floor_pcd.size() > 50);

  pcl::PointCloud<pcl::PointXYZI> floor_plane_cloud;
  Eigen::Vector3d plane_normal;
  if(extractFloorPlane(floor_pcd, floor_plane_cloud, plane_normal)) {
    // #ifdef test
    // std::string path_1 = "/home/wd/datasets/beijing/2/floor_plane_cloud.pcd";
    // pcl::io::savePCDFile(path_1, floor_plane_cloud);
    // std::cout << "plane_normal: " << plane_normal.transpose() << std::endl;
    // #endif
    update_Tx_dr_L = computeRollPitchAndZ(floor_plane_cloud, Tx_dr_L, plane_normal);
    return true;
  }
  else {
    return false;
  }
}

void FloorPlaneConstriant::filteredPcd(const pcl::PointCloud<pcl::PointXYZI>& raw_pcd,
                                       pcl::PointCloud<pcl::PointXYZI>& cloud) {
  for(const auto& p : raw_pcd) {
    if(std::abs(p.x) < 20 && std::abs(p.y) < 20 && std::abs(p.z) < 0.25)
      cloud.push_back(p);
  }                
}

// todo: compute plane normal vector!
bool FloorPlaneConstriant::extractFloorPlane(const pcl::PointCloud<pcl::PointXYZI>& cloud, 
                                             pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud,
                                             Eigen::Vector3d& plane_normal) {
  std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
  pcl::PointCloud<pcl::PointXYZI> tmp_cloud = cloud;
  cutVoxel(surf_map, tmp_cloud); 
  for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
    iter->second->recut();     

  for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
  {
    std::vector<Plane*> plane_list;
    iter->second->get_plane_list(plane_list);

    std::vector<Plane*> merge_plane_list;
    if(plane_list.size() > 1) {
      mergePlane(plane_list, merge_plane_list);
      plane_list = merge_plane_list;
    }

    for(auto plane: plane_list)
    {
      std::vector<unsigned int> colors;
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));

      if(plane->plane_points.size() < 1000) continue;
      if(std::abs(plane->normal.dot(Eigen::Vector3d::UnitZ())) > 0.98) {
        for(auto pv: plane->plane_points) {
          pcl::PointXYZI pi;
          pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
          // pi.intensity = todo
          floor_plane_cloud.push_back(pi);
        }
      }
    }
  }

  if(floor_plane_cloud.size() < 0) {
    std::cout << "cannot extract floor plane cloud!" << std::endl;
    return false;
  }         
  plane_normal = computePlaneNormal(floor_plane_cloud);   
  return true;                                      
}

void FloorPlaneConstriant::cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                                    pcl::PointCloud<pcl::PointXYZI>& pl_feat)
{
  float loc_xyz[3];
  printf("extract_floor_plane/total point size %ld\n", pl_feat.points.size());
  for(pcl::PointXYZI& p_c: pl_feat.points)
  {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);

    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_orig[j] / voxel_size_;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->all_points.push_back(pvec_orig);
      iter->second->vec_orig.push_back(pvec_orig);        
      iter->second->sig_orig.push(pvec_orig);
    }
    else
    {
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(eigen_ratio_, layer_limit_, what_);
      ot->all_points.push_back(pvec_orig);
      ot->vec_orig.push_back(pvec_orig);
      ot->sig_orig.push(pvec_orig);
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size_;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size_;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size_;
      ot->quater_length = voxel_size_ / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

void FloorPlaneConstriant::mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list)
{
  for(size_t i = 0; i < origin_list.size(); i++)
    origin_list[i]->id = 0; // 初始化
  int current_id = 1; // 平面id
  for(auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--)
  {
    for(auto iter2 = origin_list.begin(); iter2 != iter; iter2++)
    {
      Eigen::Vector3d normal_diff = (*iter)->normal - (*iter2)->normal; // 发向量同向
      Eigen::Vector3d normal_add = (*iter)->normal + (*iter2)->normal; // 发向量反向
      double dis1 = fabs((*iter)->normal(0) * (*iter2)->center(0) +
                         (*iter)->normal(1) * (*iter2)->center(1) +
                         (*iter)->normal(2) * (*iter2)->center(2) + (*iter)->d);
      double dis2 = fabs((*iter2)->normal(0) * (*iter)->center(0) +
                         (*iter2)->normal(1) * (*iter)->center(1) +
                         (*iter2)->normal(2) * (*iter)->center(2) + (*iter2)->d);
      if(normal_diff.norm() < 0.2 || normal_add.norm() < 0.2) // 11.3度
        if(dis1 < 0.05 && dis2 < 0.05)
        {
          if((*iter)->id == 0 && (*iter2)->id == 0)
          {
            (*iter)->id = current_id;
            (*iter2)->id = current_id;
            current_id++;
          }
          else if((*iter)->id == 0 && (*iter2)->id != 0)
            (*iter)->id = (*iter2)->id;
          else if((*iter)->id != 0 && (*iter2)->id == 0)
            (*iter2)->id = (*iter)->id;
        }
    }
  }
  std::vector<int> merge_flag;
  for(size_t i = 0; i < origin_list.size(); i++)
  {
    auto it = std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id);
    if(it != merge_flag.end()) continue; // 已经merge过的平面，直接跳过
    
    if(origin_list[i]->id == 0) // 没有merge的平面
    {
      if(origin_list[i]->points_size > 100)
        merge_list.push_back(origin_list[i]);
      continue;
    }
    Plane* merge_plane = new Plane;
    (*merge_plane) = (*origin_list[i]);
    for(size_t j = 0; j < origin_list.size(); j++)
    {
      if(i == j) continue;
      if(origin_list[i]->id != 0)
        if(origin_list[j]->id == origin_list[i]->id)
          for(auto pv: origin_list[j]->plane_points)
            merge_plane->plane_points.push_back(pv); // 跟当前平面id相同的都merge
    }
    merge_plane->covariance = Eigen::Matrix3d::Zero();
    merge_plane->center = Eigen::Vector3d::Zero();
    merge_plane->normal = Eigen::Vector3d::Zero();
    merge_plane->points_size = merge_plane->plane_points.size();
    merge_plane->radius = 0;
    for(auto pv: merge_plane->plane_points)
    {
      merge_plane->covariance += pv * pv.transpose();
      merge_plane->center += pv;
    }
    merge_plane->center = merge_plane->center / merge_plane->points_size;
    merge_plane->covariance = merge_plane->covariance / merge_plane->points_size -
                              merge_plane->center * merge_plane->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    merge_plane->id = origin_list[i]->id;
    merge_plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    merge_plane->min_eigen_value = evalsReal(evalsMin);
    merge_plane->radius = sqrt(evalsReal(evalsMax));
    merge_plane->d = -(merge_plane->normal(0) * merge_plane->center(0) +
                      merge_plane->normal(1) * merge_plane->center(1) +
                      merge_plane->normal(2) * merge_plane->center(2));
    merge_plane->p_center.x = merge_plane->center(0);
    merge_plane->p_center.y = merge_plane->center(1);
    merge_plane->p_center.z = merge_plane->center(2);
    merge_plane->p_center.normal_x = merge_plane->normal(0);
    merge_plane->p_center.normal_y = merge_plane->normal(1);
    merge_plane->p_center.normal_z = merge_plane->normal(2);
    merge_plane->is_plane = true;
    merge_flag.push_back(merge_plane->id);
    merge_list.push_back(merge_plane);
  }
}

Eigen::Vector3d FloorPlaneConstriant::computePlaneNormal(const pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud) {
  // 构造数据矩阵
  Eigen::MatrixXd A(floor_plane_cloud.size(), 3);
  for (size_t i = 0; i < floor_plane_cloud.size(); ++i) {
      A(i, 0) = floor_plane_cloud[i].x;
      A(i, 1) = floor_plane_cloud[i].y;
      A(i, 2) = 1; // 添加常数项
  }

  // 构造观测值向量
  Eigen::VectorXd b(floor_plane_cloud.size());
  for (size_t i = 0; i < floor_plane_cloud.size(); ++i) {
    b(i) = floor_plane_cloud[i].z;
  }

  // 使用最小二乘法求解
  Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(b);

  // 构造平面法向量
  Eigen::Vector3d normal;
  normal << coeffs(0), coeffs(1), -1.0; // 平面方程为 ax + by + cz + d = 0，c为-1

  return normal.normalized(); // 归一化法向量
}

Eigen::Matrix4d FloorPlaneConstriant::computeRollPitchAndZ(const pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud,
                                                           const Eigen::Matrix4d& Tx_dr_L,
                                                           Eigen::Vector3d& plane_normal) {
  Eigen::Matrix4d update_Tx_dr_L = Eigen::Matrix4d::Identity();

  pcl::PointCloud<pcl::PointXYZI> floor_plane_cloud_lidar;
  Eigen::Matrix4d Tx_L_dr = Tx_dr_L.inverse();
  pcl::transformPointCloud(floor_plane_cloud, floor_plane_cloud_lidar, Tx_L_dr); 
  // make the normal upward
  if (plane_normal.dot(Eigen::Vector3d::UnitZ()) < 0.0f) {
    plane_normal *= -1.0f;
  }  
  Eigen::Vector3d v = plane_normal.head<3>().cross(Eigen::Vector3d::UnitZ());
  double c = plane_normal.head<3>().dot(Eigen::Vector3d::UnitZ()); 
  // if(c < 0.98) {
  //   std::cerr << "the normal is not vertical!" << std::endl;
  // }
  Eigen::Matrix3d v_mat;
  v_mat << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  double tmp_rod = 1.0 / (1.0 + c);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity() + v_mat + v_mat * v_mat * tmp_rod; 
  Eigen::Vector3d ypr = convertRotationMatrixToEulerYPR(rotation);
  std::cout << "yaw: " << ypr[0] * 180 / M_PI  << ", "
            << "pitch: " << ypr[1] * 180 / M_PI  << ", " 
            << "roll: " << ypr[2] * 180 / M_PI  << std::endl;                              
  ypr[0] = 0.0;
  rotation = convertEulerYPRToRotationMatrix(ypr);

  Eigen::Matrix4d Tx_dr_pr = Eigen::Matrix4d::Identity();
  Tx_dr_pr.block<3, 3>(0, 0) = rotation;
  Tx_dr_pr.block<3, 1>(0, 3) = Eigen::Vector3d::Zero();

  // test
  Eigen::Vector3d new_norm = rotation * plane_normal;
  if((new_norm.normalized() - Eigen::Vector3d::UnitZ()).norm() < 10e-3) {
    std::cout << "successful to constraint pitch and roll" << std::endl;
    std::cout << "new_norm: " << new_norm << std::endl;

    pcl::PointCloud<pcl::PointXYZI> floor_plane_cloud_truth;
    pcl::transformPointCloud(floor_plane_cloud, floor_plane_cloud_truth, Tx_dr_pr);
    double z_mean = 0.0;
    for(auto p: floor_plane_cloud_truth) {
      z_mean += p.z;
    }
    z_mean /= double(floor_plane_cloud_truth.size());
    std::cout << "z_mean: " << z_mean << std::endl;

    update_Tx_dr_L = Tx_dr_pr * Tx_dr_L;
    update_Tx_dr_L(2, 3) -= z_mean;

    // test translation z
    pcl::transformPointCloud(floor_plane_cloud_lidar, floor_plane_cloud_truth, update_Tx_dr_L);
    #ifdef test
    std::string path_1 = "/home/wd/datasets/beijing/2/floor_plane_cloud.pcd";
    pcl::io::savePCDFile(path_1, floor_plane_cloud_truth);
    #endif
    double refine_z_mean = 0.0;
    for(auto p: floor_plane_cloud_truth) {
      refine_z_mean += p.z;
    }
    refine_z_mean /= double(floor_plane_cloud_truth.size());
    std::cout << "after z_mean: " << refine_z_mean << std::endl;
  }
  else {
    std::cerr << "failed to constraint pitch and roll!!!" << std::endl;
    std::cout << "new_norm: " << new_norm.head<3>() << std::endl;
    update_Tx_dr_L = Tx_dr_L;
  }

   return update_Tx_dr_L;
}

bool FloorPlaneConstriant::addFloorConstriantRac(const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_pcd, 
                                                 const Eigen::Matrix4d& Tx_dr_L,
                                                 Eigen::Matrix4d& update_Tx_dr_L) {
  pcl::PointCloud<pcl::PointXYZI> transformed_pcd;
  pcl::transformPointCloud(*raw_pcd, transformed_pcd, Tx_dr_L);  

  pcl::PointCloud<pcl::PointXYZI> floor_pcd;
  filteredPcd(transformed_pcd, floor_pcd);  

  pcl::PointCloud<pcl::PointXYZI> floor_plane_cloud;
  Eigen::Vector3d plane_normal;
  if(extractFloorPlaneRac(floor_pcd, floor_plane_cloud, plane_normal)) {
    update_Tx_dr_L = computeRollPitchAndZ(floor_plane_cloud, Tx_dr_L, plane_normal);
    return true;
  }
  else {
    return false;
  }
}

bool FloorPlaneConstriant::extractFloorPlaneRac(const pcl::PointCloud<pcl::PointXYZI>& cloud, 
                                                pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud,
                                                Eigen::Vector3d& plane_normal) {
  pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr model_p(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(cloud.makeShared()));
  pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_p);
  ransac.setDistanceThreshold(plane_ransac_thred_);
  ransac.computeModel();

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  ransac.getInliers(inliers->indices);
  double floor_pts_thresh = 100;
  if (inliers->indices.size() < floor_pts_thresh) {
    std::cerr << "too few inliers" << std::endl;
    return false;
  } 
  for(size_t i = 0; i < inliers->indices.size(); ++i)
    floor_plane_cloud.push_back(cloud[i]);

  Eigen::VectorXf coeffs;
  ransac.getModelCoefficients(coeffs);  
  plane_normal = Eigen::Vector3d((double)coeffs.head<3>()(0),
                                 (double)coeffs.head<3>()(1),
                                 (double)coeffs.head<3>()(2));      
}