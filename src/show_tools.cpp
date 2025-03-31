#include "show_tools.h"
#include "linear_interpolater.hpp"

#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>

enum ProjectionType { DEPTH, INTENSITY, BOTH };
enum Direction { UP, DOWN, LEFT, RIGHT };

using PoseLinearInterpolater = LinearInterpolater<Eigen::Matrix4d>;

namespace show_tools {
cv::Mat getConnectImg(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud,
    const int& dis_threshold, const int& camera_width,
    const int& camera_height) {
  cv::Mat connect_img =
      cv::Mat(camera_height, camera_width, CV_8UC3, cv::Scalar::all(255));

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  kdtree->setInputCloud(rgb_edge_cloud);
  for (auto p : *depth_edge_cloud) {
    cv::Point2d p2(p.x, -p.y);
  }

  int line_count = 0;
  // 指定近邻个数
  int K = 1;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = depth_edge_cloud->points[i];
    if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - rgb_edge_cloud->points[pointIdxNKNSearch[j]].x,
                2) +
            pow(searchPoint.y - rgb_edge_cloud->points[pointIdxNKNSearch[j]].y,
                2));
        if (distance < dis_threshold) {
          cv::Scalar color = cv::Scalar(0, 255, 0);
          line_count++;
          if ((line_count % 3) == 0) {
            cv::line(connect_img,
                     cv::Point(depth_edge_cloud->points[i].x,
                               -depth_edge_cloud->points[i].y),
                     cv::Point(rgb_edge_cloud->points[pointIdxNKNSearch[j]].x,
                               -rgb_edge_cloud->points[pointIdxNKNSearch[j]].y),
                     color, 1);
          }
        }
      }
    }
  }

  for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[0] = 255;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[2] = 0;
  }
  for (size_t i = 0; i < depth_edge_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-depth_edge_cloud->points[i].y,
                              depth_edge_cloud->points[i].x)[0] = 0;
    connect_img.at<cv::Vec3b>(-depth_edge_cloud->points[i].y,
                              depth_edge_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-depth_edge_cloud->points[i].y,
                              depth_edge_cloud->points[i].x)[2] = 255;
  }
  int expand_size = 2;
  cv::Mat expand_edge_img;
  expand_edge_img = connect_img.clone();
  for (int x = expand_size; x < connect_img.cols - expand_size; x++) {
    for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
      if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
          }
        }
      } else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
          }
        }
      }
    }
  }
  return connect_img;
}

cv::Mat fillImg(const cv::Mat& input_img, const Direction first_direct,
                const Direction second_direct) {
  cv::Mat fill_img = input_img.clone();
  for (int y = 2; y < input_img.rows - 2; y++) {
    for (int x = 2; x < input_img.cols - 2; x++) {
      if (input_img.at<uchar>(y, x) == 0) {
        if (input_img.at<uchar>(y - 1, x) != 0) {
          fill_img.at<uchar>(y, x) = input_img.at<uchar>(y - 1, x);
        } else {
          if ((input_img.at<uchar>(y, x - 1)) != 0) {
            fill_img.at<uchar>(y, x) = input_img.at<uchar>(y, x - 1);
          }
        }
      } else {
        int left_depth = input_img.at<uchar>(y, x - 1);
        int right_depth = input_img.at<uchar>(y, x + 1);
        int up_depth = input_img.at<uchar>(y + 1, x);
        int down_depth = input_img.at<uchar>(y - 1, x);
        int current_depth = input_img.at<uchar>(y, x);
        if ((current_depth - left_depth) > 5 &&
            (current_depth - right_depth) > 5 && left_depth != 0 &&
            right_depth != 0) {
          fill_img.at<uchar>(y, x) = (left_depth + right_depth) / 2;
        } else if ((current_depth - up_depth) > 5 &&
                   (current_depth - down_depth) > 5 && up_depth != 0 &&
                   down_depth != 0) {
          fill_img.at<uchar>(y, x) = (up_depth + right_depth) / 2;
        }
      }
    }
  }
  return fill_img;
}

void projection(const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_point,
                const Eigen::Matrix4d& Tx_C_L, const cv::Mat& camera_matrix,
                cv::Mat& distortion_coeff, const int& img_height,
                const int& img_width, const float& min_depth,
                const float& max_depth, cv::Mat& image_project) {
  // filter out-view pcd;
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pcd =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*raw_point, *transformed_pcd, Tx_C_L);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filted_pcd =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  for (auto p : *transformed_pcd) {
    if (p.z <= 0) continue;
    filted_pcd->push_back(p);
  }

  std::vector<cv::Point3f> pts_3d;
  std::vector<float> intensity_list;
  float max_intensity = 0.0;
  for (auto p : *filted_pcd) {
    float depth = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
    if (depth > min_depth && depth < max_depth) {
      pts_3d.emplace_back(cv::Point3f(p.x, p.y, p.z));
      intensity_list.emplace_back(p.intensity);
    }

    if (p.intensity > max_intensity) max_intensity = p.intensity;
  }

  std::string projection_type = "depth";
  if (max_intensity > 0) {
    projection_type = "intensity";
  }

  cv::Vec3d rvec(0, 0, 0);
  cv::Vec3d tvec(0, 0, 0);
  std::vector<cv::Point2f> pts_2d;
  cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, camera_matrix,
                             distortion_coeff);

  image_project = cv::Mat::zeros(img_height, img_width, CV_16UC1);
  for (size_t i = 0; i < pts_2d.size(); ++i) {
    cv::Point2f point_2d = pts_2d[i];
    if (point_2d.x <= 0 || point_2d.x >= img_width || point_2d.y <= 0 ||
        point_2d.y >= img_height)
      continue;
    else {
      if (projection_type == "depth") {
        float depth = sqrt(pow(pts_3d[i].x, 2) + pow(pts_3d[i].y, 2) +
                           pow(pts_3d[i].z, 2));
        float intensity = intensity_list[i];
        float depth_weight = 1;
        float grey = depth_weight * depth / max_depth * 65535 +
                     (1 - depth_weight) * intensity / 150 * 65535;
        if (image_project.at<ushort>(point_2d.y, point_2d.x) == 0) {
          image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
        } else if (depth < image_project.at<ushort>(point_2d.y, point_2d.x)) {
          image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
        }
      } else {
        float intensity = intensity_list[i];
        if (intensity > 100) {
          intensity = 65535;
        } else {
          intensity = (intensity / 150.0) * 65535;
        }
        image_project.at<ushort>(point_2d.y, point_2d.x) = intensity;
      }
    }
  }
  image_project.convertTo(image_project, CV_8UC1, 1 / 256.0);
}

void mapJet(double v, double vmin, double vmax, uint8_t& r, uint8_t& g,
            uint8_t& b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

cv::Mat getProjectionImg(const cv::Mat& raw_img,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_point,
                         const Eigen::Matrix4d& Tx_C_L,
                         const cv::Mat& camera_matrix,
                         cv::Mat& distortion_coeff) {
  float min_depth = 1;
  float max_depth = 80;

  int img_height = raw_img.rows;
  int img_width = raw_img.cols;
  cv::Mat depth_projection_img;
  projection(raw_point, Tx_C_L, camera_matrix, distortion_coeff, img_height,
             img_width, min_depth, max_depth, depth_projection_img);

  cv::Mat merge_img;
  raw_img.copyTo(merge_img);
  for (int x = 0; x < raw_img.cols; ++x) {
    for (int y = 0; y < raw_img.rows; ++y) {
      uint8_t r, g, b;
      float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
      mapJet(norm, 0, 1, r, g, b);

      if (norm != 0.0) {
        cv::circle(merge_img, cv::Point2i(x, y), 2, cv::Scalar(b, g, r), -1);
      }
    }
  }

  return merge_img;
}

void getColorCloud(const std::vector<cv::Mat>& rgb_img_vec,
                   const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& pcd_vec,
                   const std::vector<double>& cam_stamp_vec,
                   const StampedPoseVectorPtr& stamp_pose_vec,
                   const Eigen::Matrix4d& Tx_C_L,
                   const cv::Mat& camera_matrix,
                   const cv::Mat& distortion_coeff,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color_cloud) {
  const int density = 1;
  const float min_depth = 1;
  const float max_depth = 50;

  // todo: check pose interpolater
  std::string save_path = " ";
  StampedPoseVectorPtr save_pose_vec = std::make_shared<StampedPoseVector>();

  color_cloud =  pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
    new pcl::PointCloud<pcl::PointXYZRGB>);

  std::unique_ptr<PoseLinearInterpolater> pose_interpolater = 
    std::make_unique<PoseLinearInterpolater>(1000, 0.1);
  for(size_t i = 0; i < stamp_pose_vec->size(); ++i) {
    pose_interpolater->insert(stamp_pose_vec->at(i).first, stamp_pose_vec->at(i).second);
  }
  std::cout << "pose_interpolater size: " << pose_interpolater->size() << std::endl;
 
  for(size_t a = 0; a < rgb_img_vec.size(); ++a) { // camera poses
    std::vector<cv::Point3f> pts_3d;

    double curr_cam_stamp = cam_stamp_vec[a];
    Eigen::Matrix4d curr_pose = Eigen::Matrix4d::Identity();
    pose_interpolater->evaluate(curr_cam_stamp, &curr_pose);
    // todo
    save_pose_vec->emplace_back(StampedPose(curr_cam_stamp, curr_pose));

    for(size_t b = 0; b < stamp_pose_vec->size(); ++b) {
      for(size_t i = 0; i < pcd_vec[b]->size(); i += density) {
        pcl::PointXYZI point = pcd_vec[b]->points[i];
        Eigen::Vector4d pt1(point.x, point.y, point.z, 1.0);
        pt1 = stamp_pose_vec->at(b).second * pt1;
        pt1 = curr_pose.inverse() * pt1;
        pt1 = Tx_C_L * pt1;
        if(pt1[2] <= 0.0) continue;
        float depth = sqrt(pow(pt1[0], 2) + pow(pt1[1], 2) + pow(pt1[2], 2));
        if(depth > min_depth && depth < max_depth)
          pts_3d.emplace_back(cv::Point3f(pt1[0], pt1[1], pt1[2]));
      }
    }
    
    std::vector<cv::Point2f> pts_2d;
    cv::Vec3d rvec(0, 0, 0);
    cv::Vec3d tvec(0, 0, 0);
    // cv::projectPoints(pts_3d, rvec, tvec, camera_matrix, distortion_coeff, pts_2d);
    cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, camera_matrix,
                               distortion_coeff);                 
    
    int image_rows = rgb_img_vec[a].rows;
    int image_cols = rgb_img_vec[a].cols;
                
    for(size_t i = 0; i < pts_2d.size(); ++i) {
      cv::Point2f point_2d = pts_2d[i];
      if (point_2d.x <= 0 || point_2d.x >= image_cols || point_2d.y <= 0 ||
          point_2d.y >= image_rows)
        continue;
      else {
        cv::Scalar color = rgb_img_vec[a].at<cv::Vec3b>(point_2d);
        if(color[0] == 0 && color[1] == 0 && color[2] == 0) continue;
                          
        Eigen::Vector4d pt(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z, 1.0);
        pt = curr_pose * Tx_C_L.inverse() * pt;
        pcl::PointXYZRGB p;
        p.x = pt(0); 
        p.y = pt(1); 
        p.z = pt(2);
        p.b = color[0]; 
        p.g = color[1]; 
        p.r = color[2];

        color_cloud->push_back(p);
      }
    }
  }
  
  // todo
  std::cout << "save_pose_vec size: " << save_pose_vec->size() << std::endl;
  file_io::writeStampPoseToFile(save_pose_vec, save_path);
}

}  // namespace show_tools