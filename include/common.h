
#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/common/io.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

struct PnPData {
  double x, y, z, u, v;
};

struct VPnPData {
  double x, y, z, u, v;
  Eigen::Vector2d direction;
  Eigen::Vector2d direction_lidar;
  int number;
};

using namespace std;

template <class T> void input(T matrix[4][5]) {
  cout << "please input matrix element's data" << endl;
  for (int i = 1; i < 4; i++) {
    for (int j = 1; j < 5; j++) {
      cin >> matrix[i][j];
    }
  }
  cout << "input ok";
}

// void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
//             uint8_t &b) {
//   r = 255;
//   g = 255;
//   b = 255;

//   if (v < vmin) {
//     v = vmin;
//   }

//   if (v > vmax) {
//     v = vmax;
//   }

//   double dr, dg, db;

//   if (v < 0.1242) {
//     db = 0.504 + ((1. - 0.504) / 0.1242) * v;
//     dg = dr = 0.;
//   } else if (v < 0.3747) {
//     db = 1.;
//     dr = 0.;
//     dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
//   } else if (v < 0.6253) {
//     db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
//     dg = 1.;
//     dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
//   } else if (v < 0.8758) {
//     db = 0.;
//     dr = 1.;
//     dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
//   } else {
//     db = 0.;
//     dg = 0.;
//     dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
//   }

//   r = (uint8_t)(255 * dr);
//   g = (uint8_t)(255 * dg);
//   b = (uint8_t)(255 * db);
// }
typedef struct VoxelGrid {
  float size = 0.5;
  int index;
  Eigen::Vector3d origin;
  pcl::PointCloud<pcl::PointXYZI> cloud;
} VoxelGrid;