#include <string>
#include <unordered_map>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "extract_lidar_feature.h"

std::string pcd_path_ = "";
float voxel_size_ = 10;
float eigen_ratio_ = 0.2;
int layer_limit_ = 5; // origin: 3 
double what_ = 0.95; // origin: 0.98, what???
float theta_min_ = cos(DEG2RAD(10));
float theta_max_ = cos(DEG2RAD(170)); 
float maxDisThre_ = 0.1;
float similarityThreshold_ = 0.95;
int line_points_nums_ = 30;

void cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
              pcl::PointCloud<pcl::PointXYZI>& pl_feat,
              float voxel_size, float eigen_ratio, int layer_limit, double what);
void estimateEdge(const std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map, 
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud);
void estimatePlane(OCTO_TREE_ROOT* surf, std::vector<Plane*>& merge_plane_list, 
                   pcl::KdTreeFLANN<pcl::PointXYZI>& kd_tree,
                   pcl::PointCloud<pcl::PointXYZI>& input_cloud);                  
void mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list);
void extractLine(const std::vector<Plane*>& merge_plane_list, 
                 const pcl::KdTreeFLANN<pcl::PointXYZI>& kd_tree,
                 const pcl::PointCloud<pcl::PointXYZI>& input_cloud,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr& lines);
void projectLine(const Plane* plane1, const Plane* plane2, std::vector<Eigen::Vector3d>& line_point);
void mergeLine(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& origin_line_list,  
               std::vector<pcl::PointCloud<pcl::PointXYZI>>& merge_line_list);
Eigen::Vector3d computeLineDirection(const pcl::PointCloud<pcl::PointXYZI>& line);
double cosineSimilarity(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
         
int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud =
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(pcd_path_, *lidar_cloud);
  std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
  cutVoxel(surf_map, *lidar_cloud, voxel_size_, eigen_ratio_, layer_limit_, what_);

  for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_edge_cloud;       
  estimateEdge(surf_map, lidar_edge_cloud);
  return 0;
}

void cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
              pcl::PointCloud<pcl::PointXYZI>& pl_feat,
              float voxel_size, float eigen_ratio, int layer_limit, double what)
{
  float loc_xyz[3];
  printf("total point size %ld\n", pl_feat.points.size());
  for(pcl::PointXYZI& p_c: pl_feat.points)
  {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);

    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_orig[j] / voxel_size;
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
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(eigen_ratio, layer_limit, what);
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

void estimateEdge(const std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map, 
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud) {
  for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
  {
    std::vector<Plane*> merge_plane_list;
    pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
    pcl::PointCloud<pcl::PointXYZI> input_cloud;
    estimatePlane(iter->second, merge_plane_list, kd_tree, input_cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr lines;
    extractLine(merge_plane_list, kd_tree, input_cloud, lines);

    *lidar_edge_cloud += *lines;
  }

  
}

void estimatePlane(OCTO_TREE_ROOT* surf, std::vector<Plane*>& merge_plane_list, 
                   pcl::KdTreeFLANN<pcl::PointXYZI>& kd_tree,
                   pcl::PointCloud<pcl::PointXYZI>& input_cloud) {
  std::vector<Plane*> plane_list;
  surf->get_plane_list(plane_list);

  if(plane_list.size() <= 1) return;
  
  
  
  for(auto pv: surf->all_points)
  {
    pcl::PointXYZI p;
    p.x = pv(0); p.y = pv(1); p.z = pv(2);
    input_cloud.push_back(p);
  }
  // todo: check input_cloud
  kd_tree.setInputCloud(input_cloud.makeShared());
  
  mergePlane(plane_list, merge_plane_list);
  // todo: plane_list vs merge_plane_list
  if(merge_plane_list.size() <= 1) return;

  // for(auto plane: merge_plane_list)
  // {
  //   pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
  //   std::vector<unsigned int> colors;
  //   colors.push_back(static_cast<unsigned int>(rand() % 255));
  //   colors.push_back(static_cast<unsigned int>(rand() % 255));
  //   colors.push_back(static_cast<unsigned int>(rand() % 255));
  //   for(auto pv: plane->plane_points)
  //   {
  //     pcl::PointXYZRGB pi;
  //     pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
  //     pi.r = colors[0]; pi.g = colors[1]; pi.b = colors[2];
  //     color_cloud.points.push_back(pi);
  //   }
  // }
}

 void mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list)
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

void extractLine(const std::vector<Plane*>& merge_plane_list, 
                 const pcl::KdTreeFLANN<pcl::PointXYZI>& kd_tree,
                 const pcl::PointCloud<pcl::PointXYZI>& input_cloud,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr& lines) {
  std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
  for(size_t p1_index = 0; p1_index < merge_plane_list.size()-1; p1_index++)
  {  
    for(size_t p2_index = p1_index+1; p2_index < merge_plane_list.size(); p2_index++)
    {
      std::vector<Eigen::Vector3d> line_point;
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
          // if(pointNKNSquaredDistance[K-1] < 0.01)
          // std::cout << "dis: " << (tmp - line_point[j]).norm() << std::endl;
          // todo: set maxDisThre = 0.1
          if((tmp - line_point[j]).norm() < maxDisThre_)
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

void projectLine(const Plane* plane1, const Plane* plane2, std::vector<Eigen::Vector3d>& line_point)
{
  float theta = plane1->normal.dot(plane2->normal);
  theta_min_ = cos(DEG2RAD(10));
  theta_max_ = cos(DEG2RAD(170));
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

void mergeLine(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& origin_line_list,  
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

Eigen::Vector3d computeLineDirection(const pcl::PointCloud<pcl::PointXYZI>& line) {

  Eigen::Vector3d direct_vec = {0.0, 0.0, 0.0};
  direct_vec.x() = line[0].x - line[line.size()-1].x;
  direct_vec.y() = line[0].y - line[line.size()-1].y;
  direct_vec.z() = line[0].z - line[line.size()-1].z;
  direct_vec.normalize();
  
  return direct_vec;
}

double cosineSimilarity(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  double dotProduct = v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
  double magnitude1 = std::sqrt(v1.x() * v1.x() + v1.y() * v1.y() + v1.z() * v1.z());
  double magnitude2 = std::sqrt(v2.x() * v2.x() + v2.y() * v2.y() + v2.z() * v2.z());

  return dotProduct / (magnitude1 * magnitude2);
}
