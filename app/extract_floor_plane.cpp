#include <unordered_map>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include "extract_lidar_feature.hpp"

         

// plane param
float voxel_size_ = 40;
float eigen_ratio_ = 0.025;
int layer_limit_ = 3; // origin: 3 
double what_ = 0.98; // origin: 0.98, what???

double floor_thred_ = 0.025;
double floor_normal_thresh_ = 10;
std::string pcd_path_ = "/media/lam_data/标定数据/2pb65/新方案数据/20240201/line/lidar_baselink/pre_data/pcd/0_visual.pcd";

void cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
              pcl::PointCloud<pcl::PointXYZI>& pl_feat,
              float voxel_size, float eigen_ratio, int layer_limit, double what);
void extractFloorPlane(const std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map, 
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& floor_plane_cloud);   
void mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list);                       

void addFloorConstriant(const pcl::PointCloud<pcl::PointXYZI>& pcd_floor);

// 对比两种方法: ransac is bad
int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud =
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(pcd_path_, *lidar_cloud);

  // cut floor according to init extrinsic
  Eigen::Matrix4d init_Tx_Dr_L = Eigen::Matrix4d::Identity();
  Eigen::Quaterniond init_q = Eigen::Quaterniond(0.71637160463028582,
                                                 0.0059138554274193577,
                                                 0.0089709431225763418,
                                                 0.69763620359962986);
  Eigen::Vector3d init_t = Eigen::Vector3d(1.595217885590408,
                                           0.018723255689350848,
                                           1.91);
  init_Tx_Dr_L.block<3, 3>(0, 0) = init_q.toRotationMatrix();
  init_Tx_Dr_L.block<3, 1>(0, 3) = init_t;

  pcl::PointCloud<pcl::PointXYZI> transformed_pcd;
  pcl::transformPointCloud(*lidar_cloud, transformed_pcd, init_Tx_Dr_L);

  pcl::PointCloud<pcl::PointXYZI>::Ptr floor_pcd =
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  for(const auto& p : transformed_pcd) {
    if(std::abs(p.x) < 20 && std::abs(p.y) < 20 && std::abs(p.z) < 0.25)
      floor_pcd->push_back(p);
  }
  // if(floor_pcd->size() > 0) {
  //   std::cout << "floor_pcd size: " << floor_pcd->size() << std::endl;
  //   std::string path = "/home/wd/datasets/2/floor.pcd";
  //   floor_pcd->height = 1;
  //   floor_pcd->width = floor_pcd->size();
  //   pcl::io::savePCDFile(path, *floor_pcd);
  // }

  // addFloorConstriant(*floor_pcd);

  std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
  cutVoxel(surf_map, *floor_pcd, voxel_size_, eigen_ratio_, layer_limit_, what_);

  for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();     

  pcl::PointCloud<pcl::PointXYZI>::Ptr floor_plane_cloud =
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  extractFloorPlane(surf_map, floor_plane_cloud);
  if(floor_plane_cloud->size() > 1000) {
    std::string path = "/home/wd/datasets/65/floor_plane.pcd";
    floor_plane_cloud->height = 1;
    floor_plane_cloud->width = floor_plane_cloud->size();
    pcl::io::savePCDFile(path, *floor_plane_cloud);
  }

  return 0;
}

void cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
              pcl::PointCloud<pcl::PointXYZI>& pl_feat,
              float voxel_size, float eigen_ratio, int layer_limit, double what)
{
  float loc_xyz[3];
  printf("extract_floor_plane/total point size %ld\n", pl_feat.points.size());
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

void extractFloorPlane(const std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map, 
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& floor_plane_cloud) {
  int surf_cnt = 0;
  for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
  {
    std::cout << "#### " << surf_cnt << "####" << std::endl;
    std::vector<Plane*> plane_list;
    iter->second->get_plane_list(plane_list);

    std::vector<Plane*> merge_plane_list;
    if(plane_list.size() > 1) {
      mergePlane(plane_list, merge_plane_list);
      plane_list = merge_plane_list;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes =
         pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(auto plane: plane_list)
    {
      std::vector<unsigned int> colors;
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));
      colors.push_back(static_cast<unsigned int>(rand() % 255));

      std::cout << "plane points size: " << plane->plane_points.size() << std::endl;
      // for(auto pv: plane->plane_points)
      // {
      //   pcl::PointXYZRGB pi;
      //   pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
      //   pi.r = colors[0]; pi.g = colors[1]; pi.b = colors[2];
      //   planes->points.push_back(pi);
      // }

      std::cout << "plane normal: " << plane->normal.transpose() << std::endl;

      if(plane->plane_points.size() < 1000) continue;
      if(std::abs(plane->normal.dot(Eigen::Vector3d::UnitZ())) > 0.98) {
        for(auto pv: plane->plane_points) {
          pcl::PointXYZI pi;
          pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
          // pi.intensity = todo
          floor_plane_cloud->points.push_back(pi);
        }
      }
    }

    // if(planes->size() > 1000) {
    //   std::string path = "/home/wd/datasets/2/floor_plane/" + std::to_string(surf_cnt) + ".pcd";
    //   planes->height = 1;
    //   planes->width = planes->size();
    //   pcl::io::savePCDFile(path, *planes);
    //   ++surf_cnt;
    // }
  }                       
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

void addFloorConstriant(const pcl::PointCloud<pcl::PointXYZI>& pcd_floor) {
  pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr model_p(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(pcd_floor.makeShared()));
  pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_p);
  ransac.setDistanceThreshold(floor_thred_);
  ransac.computeModel();

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  ransac.getInliers(inliers->indices);
  double floor_pts_thresh = 100;
  if (inliers->indices.size() < floor_pts_thresh) {
    std::cerr << "too few inliers" << std::endl;
    return;
  } 

  Eigen::VectorXf coeffs;
  ransac.getModelCoefficients(coeffs);  
  Eigen::Vector3d plane_normal = Eigen::Vector3d((double)coeffs.head<3>()(0),
                                   (double)coeffs.head<3>()(1),
                                   (double)coeffs.head<3>()(2));      

  // make the normal upward
  if (plane_normal.dot(Eigen::Vector3d::UnitZ()) < 0.0f) {
    plane_normal *= -1.0f;
  }
  double c = plane_normal.dot(Eigen::Vector3d::UnitZ());
  if(c < 0.98) return;
  pcl::PointCloud<pcl::PointXYZI> floor_plane;
  for(size_t i = 0; i < inliers->indices.size(); ++i)
    floor_plane.push_back(pcd_floor[i]);

  if(floor_plane.size() > 0) {
      std::cout << "floor_plane size: " << floor_plane.size() << std::endl;
      std::string path = "/home/wd/datasets/65/floor_plane_rac.pcd";
      floor_plane.height = 1;
      floor_plane.width = floor_plane.size();
      pcl::io::savePCDFile(path, floor_plane);
  }  

  
}