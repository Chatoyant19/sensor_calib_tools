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

typedef struct VoxelGrid {
  float size = 0.5;
  int index;
  Eigen::Vector3d origin;
  pcl::PointCloud<pcl::PointXYZI> cloud;
} VoxelGrid;