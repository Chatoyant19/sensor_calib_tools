#include "file_io.h"
#include "floor_plane_constriant.h"
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <unordered_map>

std::string pcd_path_ = "/home/wd/datasets/hesai64/pcd/970.000133000.pcd";
std::string lidar_extrinsic_file = "/media/lam_data/members/wangdan/mapping/mapfree/shanmuhuiyuandian_gaokexiludian/B2_234/14683_15908_1_other_20240726_any_common/zrecorder/normal/zros_files/data/changcheng_wey_vv6_ma112_dropngo3.0/component/lidar/roof_transform.pb.txt";
std::string floor_plane_save_path = "/home/wd/datasets/hesai64/floor.pcd";
bool use_RASANC = false;

// 对比两种方法: ransac is bad
int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(pcd_path_, *lidar_cloud);

  std::unique_ptr<FloorPlaneConstriant> floor_plane_constraint =
      std::make_unique<FloorPlaneConstriant>();

  Eigen::Matrix4d T_dr_lidar;
  file_io::readExtrinsicFromPbFile(lidar_extrinsic_file, T_dr_lidar);

  pcl::PointCloud<pcl::PointXYZI> floor_plane_cloud;
  Eigen::Matrix4d update_T_dr_lidar;
  if (use_RASANC) {
    if (floor_plane_constraint->addFloorConstraintRac(lidar_cloud, T_dr_lidar, update_T_dr_lidar)) {
      floor_plane_constraint->getFloorPlaneCloud(floor_plane_cloud);
    }
  } else {
    if (floor_plane_constraint->addFloorConstraint(lidar_cloud, T_dr_lidar, update_T_dr_lidar)) {
      floor_plane_constraint->getFloorPlaneCloud(floor_plane_cloud);
    }
  }

  if (floor_plane_cloud.size() > 0)
    pcl::io::savePCDFile(floor_plane_save_path, floor_plane_cloud);

  return 0;
}
