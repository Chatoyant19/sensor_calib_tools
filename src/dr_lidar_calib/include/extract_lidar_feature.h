#ifndef EXTRACT_LIDAR_FEATURE
#define EXTRACT_LIDAR_FEATURE

#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

class VOXEL_LOC;
class OCTO_TREE_ROOT;
typedef struct Plane;
typedef struct Voxel;
typedef struct SinglePlane;

class ExtractLidarFeature{
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

  ExtractLidarFeature(const double& voxel_size, const double& eigen_ratio, const double& p2line_dis_thre,
                      double& theta_min, double& theta_max);
  ExtractLidarFeature(const double& voxel_size, const double& ransac_dis_threshold, const int& plane_size_threshold,
                      const double& p2line_dis_thre,
                      double& theta_min, double& theta_max);

  void getEdgeFeaturesByAdaVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_lidar_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud);
  void getEdgeFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_lidar_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud);    
 
 private:
  void cutVoxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                pcl::PointCloud<pcl::PointXYZI>& pl_feat);
  void estimateEdgeByAdaVoxel(const std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map, 
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud);
  void estimatePlaneByAdaVoxel(OCTO_TREE_ROOT* surf, std::vector<Plane*>& merge_plane_list);
  void mergePlane(const std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list);                
  void extractLineByAdaVoxel(const std::vector<Plane*>& merge_plane_list, 
                 const pcl::KdTreeFLANN<pcl::PointXYZI>& kd_tree,
                 const pcl::PointCloud<pcl::PointXYZI>& input_cloud,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr& lines);
  void projectLine(const Plane* plane1, const Plane* plane2, 
                   std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& line_point);
  void mergeLine(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& origin_line_list,  
                 std::vector<pcl::PointCloud<pcl::PointXYZI>>& merge_line_list);
  Eigen::Vector3d computeLineDirection(const pcl::PointCloud<pcl::PointXYZI>& line);
  double cosineSimilarity(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);  

  void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                 std::unordered_map<VOXEL_LOC, Voxel*> &voxel_map); 
  void estimateEdge(const std::unordered_map<VOXEL_LOC, Voxel*> &voxel_map, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_cloud);   
  void estimatePlane(Voxel* voxel, std::vector<SinglePlane>& merge_plane_list);
  void mergePlane_1(std::vector<SinglePlane>& merge_list);     
  void calcLine(const std::vector<SinglePlane> &plane_list, 
                const Eigen::Vector3d origin,
                std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list);                              
 
 private:
  double voxel_size_;
  double eigen_ratio_;
  double theta_min_;
  double theta_max_;
  double p2line_dis_thre_;

  int layer_limit_;
  // what is what?
  double what_;
  double similarityThreshold_;
  int line_points_nums_;

  double ransac_dis_threshold_;
  int plane_size_threshold_;
};

#endif