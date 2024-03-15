
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>

#include "extract_lidar_feature.h"
#include "extract_lidar_feature.hpp"

ExtractLidarFeature::ExtractLidarFeature(const double& voxel_size, const double& eigen_ratio,
                                         const double& p2line_dis_thre,
                                         const double& theta_min, const double& theta_max) {
  std::cout << "use adaptive voxel" << std::endl;
  voxel_size_ = voxel_size;
  eigen_ratio_ = eigen_ratio;
  theta_min_ = cos(DEG2RAD(theta_min));
  theta_max_ = cos(DEG2RAD(theta_max));
  p2line_dis_thre_ = p2line_dis_thre;

  layer_limit_ = 3;
  what_ = 0.98;
  similarityThreshold_ = 0.98;
  line_points_nums_ = 300;
}

void ExtractLidarFeature::getEdgeFeaturesByAdaVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_lidar_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud) {
  std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
  cutVoxel(surf_map, *input_lidar_cloud);

  for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();  

  estimateEdgeByAdaVoxel(surf_map, lidar_edge_cloud);          
}

void ExtractLidarFeature::cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                                   pcl::PointCloud<pcl::PointXYZI>& pl_feat)
{
  float loc_xyz[3];
  printf("extract_total point size %ld\n", pl_feat.points.size());
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

void ExtractLidarFeature::estimateEdgeByAdaVoxel(const std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map, 
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud) 
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes =
         pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

  for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
  {
    pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
    pcl::PointCloud<pcl::PointXYZI> input_cloud;
    // how to get all_points
    for(auto pv: iter->second->all_points)
    {
      pcl::PointXYZI p;
      p.x = pv(0); p.y = pv(1); p.z = pv(2);
      input_cloud.push_back(p);
    }
    kd_tree.setInputCloud(input_cloud.makeShared());

    std::vector<Plane*> merge_plane_list;
    estimatePlaneByAdaVoxel(iter->second, merge_plane_list);
    if(merge_plane_list.size() <= 1) continue;

    pcl::PointCloud<pcl::PointXYZI>::Ptr lines =
         pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    extractLineByAdaVoxel(merge_plane_list, kd_tree, input_cloud, lines);
    *lidar_edge_cloud += *lines;

    /****show planes*****/
    for(auto plane: merge_plane_list)
    {
      std::vector<unsigned int> colors;
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      for(auto pv: plane->plane_points)
      {
        pcl::PointXYZRGB pi;
        pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
        pi.r = colors[0]; pi.g = colors[1]; pi.b = colors[2];
        planes->points.push_back(pi);
      }
    }
  }

  // if(planes->size() > 0) {
  //   std::cout << "planes size: " << planes->size() << std::endl;
  //   std::string path = "/home/wd/datasets/2/all_planes.pcd";
  //   planes->height = 1;
  //   planes->width = planes->size();
  //   pcl::io::savePCDFile(path, *planes);
  // }
} 

void ExtractLidarFeature::estimatePlaneByAdaVoxel(OCTO_TREE_ROOT* surf, std::vector<Plane*>& merge_plane_list) 
{
  std::vector<Plane*> plane_list;
  surf->get_plane_list(plane_list);

  if(plane_list.size() <= 1) return;
  
  mergePlane(plane_list, merge_plane_list);
}

void ExtractLidarFeature::mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list)
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

void ExtractLidarFeature::extractLineByAdaVoxel(const std::vector<Plane*>& merge_plane_list, 
                                      const pcl::KdTreeFLANN<pcl::PointXYZI>& kd_tree,
                                      const pcl::PointCloud<pcl::PointXYZI>& input_cloud,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr& lines) {
  std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
  for(size_t p1_index = 0; p1_index < merge_plane_list.size()-1; p1_index++)
  {  
    for(size_t p2_index = p1_index+1; p2_index < merge_plane_list.size(); p2_index++)
    {
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> line_point;
      projectLine(merge_plane_list[p1_index], merge_plane_list[p2_index], line_point);
      if(line_point.size() == 0) continue;

      pcl::PointCloud<pcl::PointXYZI> line_cloud;
      for(size_t j = 0; j < line_point.size(); j++)
      {
        pcl::PointXYZI p;
        p.x = line_point[j][0]; p.y = line_point[j][1]; p.z = line_point[j][2];
        // debug_cloud.points.push_back(p);
        int K = 5;
        // 创建两个向量，分别存放近邻的索引值、近邻的中心距
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        if(kd_tree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) == K)
        {
          Eigen::Vector3d tmp(input_cloud.points[pointIdxNKNSearch[K-1]].x,
                              input_cloud.points[pointIdxNKNSearch[K-1]].y,
                              input_cloud.points[pointIdxNKNSearch[K-1]].z);
    
          if((tmp - line_point[j]).norm() < p2line_dis_thre_)
           line_cloud.points.push_back(p);
        }
      }

      if(line_cloud.size() > 0) {
        line_cloud_list.emplace_back(line_cloud);
      }
    }
  }

  std::vector<pcl::PointCloud<pcl::PointXYZI>> merge_line_list;
  mergeLine(line_cloud_list, merge_line_list);
  for(const auto& line: merge_line_list)
    for(const auto& p: line)
    {
      lines->push_back(p);
    }
}

void ExtractLidarFeature::projectLine(const Plane* plane1, const Plane* plane2, 
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& line_point)
{
  float theta = plane1->normal.dot(plane2->normal);
  if(!(theta > theta_max_ && theta < theta_min_)) return;

    Eigen::Vector3d c1 = plane1->center;
    Eigen::Vector3d c2 = plane2->center;
    Eigen::Vector3d n1 = plane1->normal;
    Eigen::Vector3d n2 = plane2->normal;

    Eigen::Matrix3d A;
    Eigen::Vector3d d = n1.cross(n2).normalized();
    A.row(0) = n1.transpose();
    A.row(1) = d.transpose();
    A.row(2) = n2.transpose();
    Eigen::Vector3d b(n1.dot(c1), d.dot(c1), n2.dot(c2));
    Eigen::Vector3d O = A.colPivHouseholderQr().solve(b);

    double c1_to_line = (c1 - O).norm();
    double c2_to_line = ((c2 - O) - (c2 - O).dot(d) * d).norm();

    if(c1_to_line/c2_to_line > 8 || c2_to_line/c1_to_line > 8) return;
    
    if(plane1->points_size < plane2->points_size)
      for(auto pt: plane1->plane_points)
      {
        Eigen::Vector3d p = (pt - O).dot(d) * d + O;
        line_point.push_back(p);
      }
    else
      for(auto pt: plane2->plane_points)
      {
        Eigen::Vector3d p = (pt - O).dot(d) * d + O;
        line_point.push_back(p);
      }
    
    return;
}

void ExtractLidarFeature::mergeLine(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& origin_line_list,  
                                    std::vector<pcl::PointCloud<pcl::PointXYZI>>& merge_line_list) {
  std::vector<pcl::PointCloud<pcl::PointXYZI>> merge_tmp;
  for(size_t index1 = 0; index1 < origin_line_list.size(); ++index1) {
    bool merged = false;

    for (size_t index2 = 0; index2 < merge_tmp.size(); ++index2) {
      Eigen::Vector3d line1_direct = computeLineDirection(origin_line_list[index1]);
      Eigen::Vector3d line2_direct = computeLineDirection(merge_tmp[index2]);
      double similarity = cosineSimilarity(line1_direct, line2_direct);
      if((std::abs(similarity) > similarityThreshold_)) {
        merge_tmp[index2] += origin_line_list[index1];
        merged = true;
        break;
      }
    }

    if (!merged) {
      merge_tmp.emplace_back(origin_line_list[index1]);
    }
  }   

  for(const auto& line: merge_tmp) {
    if(line.size() > line_points_nums_) 
      merge_line_list.emplace_back(line);
  }     
}

Eigen::Vector3d ExtractLidarFeature::computeLineDirection(const pcl::PointCloud<pcl::PointXYZI>& line) {

  Eigen::Vector3d direct_vec = {0.0, 0.0, 0.0};
  direct_vec.x() = line[0].x - line[line.size()-1].x;
  direct_vec.y() = line[0].y - line[line.size()-1].y;
  direct_vec.z() = line[0].z - line[line.size()-1].z;
  direct_vec.normalize();
  
  return direct_vec;
}

double ExtractLidarFeature::cosineSimilarity(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  double dotProduct = v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
  double magnitude1 = std::sqrt(v1.x() * v1.x() + v1.y() * v1.y() + v1.z() * v1.z());
  double magnitude2 = std::sqrt(v2.x() * v2.x() + v2.y() * v2.y() + v2.z() * v2.z());

  return dotProduct / (magnitude1 * magnitude2);
}

ExtractLidarFeature::ExtractLidarFeature(const double& voxel_size, const double& ransac_dis_threshold,
                                         const int& plane_size_threshold,
                                         const double& p2line_dis_thre,
                                         const double& theta_min, const double& theta_max) {
  voxel_size_ = voxel_size;
  ransac_dis_threshold_ = ransac_dis_threshold;
  plane_size_threshold_ = plane_size_threshold;
  theta_min_ = cos(DEG2RAD(theta_min));
  theta_max_ = cos(DEG2RAD(theta_max));
  p2line_dis_thre_ = p2line_dis_thre;
}

void ExtractLidarFeature::getEdgeFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_lidar_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud) {
  std::unordered_map<VOXEL_LOC, Voxel*> voxel_map;
  initVoxel(input_lidar_cloud, voxel_map);
  // std::cout << "voxel_map size: " << voxel_map.size() << std::endl;
  estimateEdge(voxel_map, lidar_edge_cloud); 
}

void ExtractLidarFeature::initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, Voxel*> &voxel_map) 
{
  for (size_t i = 0; i < input_cloud->size(); i++) {
    const pcl::PointXYZI &p_c = input_cloud->points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size_;
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
    } else {
      Voxel *voxel = new Voxel(voxel_size_);
      voxel_map[position] = voxel;
      voxel_map[position]->voxel_origin[0] = position.x * voxel_size_;
      voxel_map[position]->voxel_origin[1] = position.y * voxel_size_;
      voxel_map[position]->voxel_origin[2] = position.z * voxel_size_;
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
      down_sampling_voxel(*(iter->second->cloud), 0.01);
    }
  }
}

void ExtractLidarFeature::estimateEdge(const std::unordered_map<VOXEL_LOC, Voxel*> &voxel_map, 
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes =
         pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  for(auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if(iter->second->cloud->size() < 50) continue;

    std::vector<SinglePlane> merge_plane_list;
    estimatePlane(iter->second, merge_plane_list);

    std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
    calcLine(merge_plane_list, iter->second->voxel_origin,
              line_cloud_list);
    // ouster 5,normal 3
    if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 8) {
      for(const auto& line: line_cloud_list) 
        *lidar_edge_cloud += line;
    }
  
    /****show planes*****/
    for(auto plane: merge_plane_list)
    {
      std::vector<unsigned int> colors;
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      for(auto pv: plane.cloud)
      {
        pcl::PointXYZRGB pi;
        pi.x = pv.x; pi.y = pv.y; pi.z = pv.z;
        pi.r = colors[0]; pi.g = colors[1]; pi.b = colors[2];
        planes->points.push_back(pi);
      }
    }

  }  

  // if(planes->size() > 0) {
  //   std::cout << "planes size: " << planes->size() << std::endl;
  //   std::string path = "/home/wd/datasets/2/all_planes.pcd";
  //   planes->height = 1;
  //   planes->width = planes->size();
  //   pcl::io::savePCDFile(path, *planes);
  // }     
}

void ExtractLidarFeature::estimatePlane(Voxel* voxel, std::vector<SinglePlane>& merge_plane_list) {
  // 创建一个体素滤波器
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud(*voxel->cloud, *cloud_filter);
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

  seg.setDistanceThreshold(ransac_dis_threshold_);

  // std::vector<SinglePlane> plane_list;
  while (cloud_filter->points.size() > 10) {
    //输入点云
    seg.setInputCloud(cloud_filter);
    seg.setMaxIterations(500);
    //分割点云
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
      std::cerr << "Could not estimate a planner model for the given dataset" << std::endl;
      break;
    }

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    pcl::PointCloud<pcl::PointXYZI> planner_cloud;
    extract.setIndices(inliers);
    extract.setInputCloud(cloud_filter);
    extract.filter(planner_cloud);
    
    if (planner_cloud.size() > plane_size_threshold_) {
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
      SinglePlane single_plane;
      single_plane.cloud = planner_cloud;
      single_plane.p_center = p_center;
      single_plane.normal << coefficients->values[0],
          coefficients->values[1], coefficients->values[2];
      merge_plane_list.emplace_back(single_plane);        
    }

    extract.setNegative(true);
    pcl::PointCloud<pcl::PointXYZI> cloud_f;
    extract.filter(cloud_f);
    *cloud_filter = cloud_f;
  }

  mergePlane_1(merge_plane_list);
}

void ExtractLidarFeature::mergePlane_1(std::vector<SinglePlane>& merge_list) {
  if(merge_list.size() < 2) return;

  std::sort(merge_list.begin(),merge_list.end(),[](const SinglePlane& p1, const SinglePlane& p2){
        return p1.cloud.size() > p2.cloud.size();});

  for(size_t plane_index1 = 0; plane_index1 < merge_list.size(); ++plane_index1) {
    SinglePlane plane1 = merge_list[plane_index1];
    for(size_t plane_index2 = plane_index1 + 1; plane_index2 < merge_list.size(); ++plane_index2) {
      SinglePlane plane2 = merge_list[plane_index2];
      float angle = plane1.normal.dot(plane2.normal);
      if(fabs(angle) < std::cos(DEG2RAD(30))) continue;

      auto dist_vec = (plane1.p_center.getVector3fMap() - plane2.p_center.getVector3fMap()).cast<double>();
      float max_dist = std::max(dist_vec.dot(plane1.normal),dist_vec.dot(plane2.normal));
      if(max_dist > 0.2) continue;

      plane1.p_center.getVector3fMap() = (plane1.p_center.getVector3fMap() * plane1.cloud.size() +
                                               plane2.p_center.getVector3fMap() * plane2.cloud.size())/
                                                       (plane1.cloud.size() + plane2.cloud.size());
      plane1.cloud += plane2.cloud;

      Eigen::Vector4f centroid;
      centroid<<plane1.p_center.getVector3fMap(),1;
      Eigen::Matrix3f cov;
      pcl::computeCovarianceMatrix(plane1.cloud, centroid, cov);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
      assert(eig.eigenvalues().x()<eig.eigenvalues().y() &&
                     eig.eigenvalues().x()<eig.eigenvalues().z());
      plane1.normal = eig.eigenvectors().col(0).cast<double>();
      merge_list[plane_index2] = merge_list.back();
      merge_list.pop_back();
      plane_index2--;
    }
  }
} 

void ExtractLidarFeature::calcLine(const std::vector<SinglePlane> &plane_list, 
    const Eigen::Vector3d origin,
    std::vector<pcl::PointCloud<pcl::PointXYZI>>& line_cloud_list) {
  if (plane_list.size() >= 2 && plane_list.size() <= 8) {
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
            std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
            Eigen::Vector3d point;
            matrix[3][1] = 1;
            matrix[3][2] = 0;
            matrix[3][3] = 0;
            matrix[3][4] = origin[0];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 1;
            matrix[3][3] = 0;
            matrix[3][4] = origin[1];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 0;
            matrix[3][3] = 1;
            matrix[3][4] = origin[2];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 1;
            matrix[3][2] = 0;
            matrix[3][3] = 0;
            matrix[3][4] = origin[0] + voxel_size_;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 1;
            matrix[3][3] = 0;
            matrix[3][4] = origin[1] + voxel_size_;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
              points.emplace_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 0;
            matrix[3][3] = 1;
            matrix[3][4] = origin[2] + voxel_size_;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
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
                  if(std::max(dis1,dis2) < p2line_dis_thre_ * p2line_dis_thre_){
                    line_cloud.push_back(p);
                    if(start_inc < 0)
                      start_inc = inc;
                    end_inc = inc;
                  }

                }
              }

              if (line_cloud.size() > 10 && line_cloud.size() > 0.5 * (end_inc-start_inc) / step_size) {
                line_cloud_list.emplace_back(line_cloud);
              }
            }
          }
        }
      }
    }
  }
}





