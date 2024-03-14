#ifndef COMMON
#define COMMON

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/common/io.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

#define MIN_PS 15
#define LAYER_LIMIT 3

void assign_qt(Eigen::Quaterniond& q, Eigen::Vector3d& t,
               const Eigen::Quaterniond& q_, const Eigen::Vector3d& t_);
Eigen::Matrix3d wedge(const Eigen::Vector3d& v);               


#endif