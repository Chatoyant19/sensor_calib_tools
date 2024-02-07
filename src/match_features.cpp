
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include "match_features.h"
#include "eigen_types.hpp"
#include "show_tools.h"

void MatchFeatures::buildVPnp(const cv::Mat& camera_matrix, 
                              const int& camera_width, const int& camera_height,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr& cam_edge_cloud_2d,
                              const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
                              const Eigen::Matrix4d& Tx_C_L,
                              const int& dis_threshold,
                              std::vector<VPnPData>& pnp_list,
                              cv::Mat& residual_img) {
  pnp_list.clear();
  cv::Mat distortion_coeff = (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0 );                                
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_lidar_line_cloud = 
    pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  filterOutViewPcd(lidar_line_cloud_3d, Tx_C_L, filtered_lidar_line_cloud);
  
  std::vector<cv::Point3d> pts_3d;
  for(auto p : *filtered_lidar_line_cloud) {
    pts_3d.emplace_back(cv::Point3d(p.x, p.y, p.z));
  }
  Eigen::Vector3d rvec_eigen = log(Tx_C_L).head<3>();
  Eigen::Vector3d tvec_eigen = log(Tx_C_L).tail<3>();
  cv::Vec3d rvec(rvec_eigen(0), rvec_eigen(1), rvec_eigen(2));
  cv::Vec3d tvec(tvec_eigen(0), tvec_eigen(1), tvec_eigen(2));
  std::vector<cv::Point2d> pts_2d;
  cv::projectPoints(pts_3d, rvec, tvec, camera_matrix, distortion_coeff, pts_2d);

  std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
  for (int y = 0; y < camera_height; ++y) {
    std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
    for (int x = 0; x < camera_width; ++x) {
      std::vector<pcl::PointXYZI> col_pts_container;
      row_pts_container.emplace_back(col_pts_container);
    }
    img_pts_container.emplace_back(row_pts_container);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_edge_cloud_2d(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < pts_2d.size(); i++) {
    pcl::PointXYZ p;
    p.x = pts_2d[i].x;
    p.y = -pts_2d[i].y;
    p.z = 0;
    pcl::PointXYZI pi_3d;
    pi_3d.x = pts_3d[i].x;
    pi_3d.y = pts_3d[i].y;
    pi_3d.z = pts_3d[i].z;
    pi_3d.intensity = 1;
    if (p.x > 0 && p.x < camera_width && pts_2d[i].y > 0 && pts_2d[i].y < camera_height) {
      if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) {
        lidar_edge_cloud_2d->points.push_back(p);
        img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
      } else {
        img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
      }
    }
  }

  if (show_residual_) {
    residual_img =
      show_tools::getConnectImg(cam_edge_cloud_2d, lidar_edge_cloud_2d, dis_threshold, camera_width, camera_height);
  }

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_lidar =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree->setInputCloud(cam_edge_cloud_2d);
  kdtree_lidar->setInputCloud(lidar_edge_cloud_2d);
  tree_cloud = cam_edge_cloud_2d;
  tree_cloud_lidar = lidar_edge_cloud_2d;
  search_cloud = lidar_edge_cloud_2d;
  // 指定近邻个数
  int K = 5;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  std::vector<int> pointIdxNKNSearchLidar(K);
  std::vector<float> pointNKNSquaredDistanceLidar(K);
  int match_count = 0;
  double mean_distance;
  int line_count = 0;
  std::vector<cv::Point2d> lidar_2d_list;
  std::vector<cv::Point2d> img_2d_list;
  std::vector<Eigen::Vector2d> camera_direction_list;
  std::vector<Eigen::Vector2d> lidar_direction_list;
  for (size_t i = 0; i < search_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = search_cloud->points[i];
    if ((kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) &&
        (kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar,
                                      pointNKNSquaredDistanceLidar) > 0)) {
      bool dis_check = true;
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
            pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
        if (distance > dis_threshold) {
          dis_check = false;
        }
      }
      if (dis_check) {
        cv::Point p_l_2d(search_cloud->points[i].x, -search_cloud->points[i].y);
        cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[0]].x,
                         -tree_cloud->points[pointIdxNKNSearch[0]].y);
        Eigen::Vector2d direction_cam(0, 0);
        std::vector<Eigen::Vector2d> points_cam;
        for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
          Eigen::Vector2d p(tree_cloud->points[pointIdxNKNSearch[i]].x,
                            -tree_cloud->points[pointIdxNKNSearch[i]].y);
          points_cam.push_back(p);
        }
        calcDirection(points_cam, direction_cam);
        Eigen::Vector2d direction_lidar(0, 0);
        std::vector<Eigen::Vector2d> points_lidar;
        for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
          Eigen::Vector2d p(
              tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x,
              -tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
          points_lidar.push_back(p);
        }
        calcDirection(points_lidar, direction_lidar);
        // direction.normalize();
        
        lidar_2d_list.push_back(p_l_2d);
        img_2d_list.push_back(p_c_2d);
        camera_direction_list.push_back(direction_cam);
        lidar_direction_list.push_back(direction_lidar);
      }
    }
  }
  for (size_t i = 0; i < lidar_2d_list.size(); i++) {
    int y = lidar_2d_list[i].y;
    int x = lidar_2d_list[i].x;
    int pixel_points_size = img_pts_container[y][x].size();
    if (pixel_points_size > 0) {
      VPnPData pnp;
      pnp.x = 0;
      pnp.y = 0;
      pnp.z = 0;
      pnp.u = img_2d_list[i].x;
      pnp.v = img_2d_list[i].y;
      for (size_t j = 0; j < pixel_points_size; j++) {
        pnp.x += img_pts_container[y][x][j].x;
        pnp.y += img_pts_container[y][x][j].y;
        pnp.z += img_pts_container[y][x][j].z;
      }
      pnp.x = pnp.x / pixel_points_size;
      pnp.y = pnp.y / pixel_points_size;
      pnp.z = pnp.z / pixel_points_size;
      pnp.direction = camera_direction_list[i];
      pnp.direction_lidar = lidar_direction_list[i];
      float theta = pnp.direction.dot(pnp.direction_lidar);
      if (theta > direction_theta_min_ || theta < direction_theta_max_) {
        pnp_list.push_back(pnp);
      }
    }
  }

}

void MatchFeatures::filterOutViewPcd(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
                                     const Eigen::Matrix4d& Tx_C_L,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr& filtered_cloud_3d) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr camera_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*lidar_line_cloud_3d, *camera_pcd, Tx_C_L);

  pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_pcd = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>); 
  for(auto p: *camera_pcd) {
    if(p.z < 0) continue;
      tmp_pcd->push_back(p);
  }

  Eigen::Matrix4d Tx_L_C = Tx_C_L.inverse();;
  pcl::transformPointCloud(*tmp_pcd, *filtered_cloud_3d, Tx_L_C);

}

void MatchFeatures::calcDirection(const std::vector<Eigen::Vector2d>& points,
                                  Eigen::Vector2d& direction) {
  Eigen::Vector2d mean_point(0, 0);
  for (size_t i = 0; i < points.size(); i++) {
    mean_point(0) += points[i](0);
    mean_point(1) += points[i](1);
  }
  mean_point(0) = mean_point(0) / points.size();
  mean_point(1) = mean_point(1) / points.size();
  Eigen::Matrix2d S;
  S << 0, 0, 0, 0;
  for (size_t i = 0; i < points.size(); i++) {
    Eigen::Matrix2d s =
        (points[i] - mean_point) * (points[i] - mean_point).transpose();
    S += s;
  }
  Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
  Eigen::MatrixXcd evecs = es.eigenvectors();
  Eigen::MatrixXcd evals = es.eigenvalues();
  Eigen::MatrixXd evalsReal;
  evalsReal = evals.real();
  Eigen::MatrixXf::Index evalsMax;
  evalsReal.rowwise().sum().maxCoeff(&evalsMax); //得到最大特征值的位置
  direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
}