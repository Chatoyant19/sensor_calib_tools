#ifndef FLOOR_PLANE_CONSTRAINT
#define FLOOR_PLANE_CONSTRAINT

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>

class VOXEL_LOC;
class OCTO_TREE_ROOT;
typedef struct Plane;

class FloorPlaneConstriant{
 public:
  FloorPlaneConstriant() {}
  bool addFloorConstriant(const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_pcd, 
                          const Eigen::Matrix4d& Tx_dr_L,
                          Eigen::Matrix4d& update_Tx_dr_L);
  bool addFloorConstriantRac(const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_pcd, 
                          const Eigen::Matrix4d& Tx_dr_L,
                          Eigen::Matrix4d& update_Tx_dr_L);                          
 private:
  void filteredPcd(const pcl::PointCloud<pcl::PointXYZI>& raw_pcd,
                   pcl::PointCloud<pcl::PointXYZI>& cloud);
  bool extractFloorPlane(const pcl::PointCloud<pcl::PointXYZI>& cloud, 
                         pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud,
                         Eigen::Vector3d& plane_normal); 
  void cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
              pcl::PointCloud<pcl::PointXYZI>& pl_feat);   
  void mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list);   
  Eigen::Vector3d computePlaneNormal(const pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud);          
  Eigen::Matrix4d computeRollPitchAndZ(const pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud,
                                       const Eigen::Matrix4d& Tx_dr_L,
                                       Eigen::Vector3d& plane_normal);     
  bool extractFloorPlaneRac(const pcl::PointCloud<pcl::PointXYZI>& cloud, 
                            pcl::PointCloud<pcl::PointXYZI>& floor_plane_cloud,
                            Eigen::Vector3d& plane_normal);                                                                           
 private:
  double voxel_size_ = 40;
  double eigen_ratio_ = 0.025;
  int layer_limit_ = 3; // origin: 3 
  double what_ = 0.98; // origin: 0.98, what???

  double plane_ransac_thred_ = 0.025;

};

#endif