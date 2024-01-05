#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>

namespace Eigen {
template <size_t _row> using VectorNd = Eigen::Matrix<double, _row, 1>;
template <size_t _dem> using MatrixNd = Eigen::Matrix<double, _dem, _dem>;
using Vector6d = VectorNd<6>;
} // namespace Eigen

Eigen::AngleAxisd convertRotationMatrixToAngleAxis(const Eigen::Matrix3d& R) {
  const double trace = R(0, 0) + R(1, 1) + R(2, 2);
  Eigen::Quaterniond q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  if (trace > 0) {
    const double s = 2 * sqrt(trace + 1);
    const double inv_s = 1.0 / s;
    q.w() = 0.25f * s;
    q.x() = (R(2, 1) - R(1, 2)) * inv_s;
    q.y() = (R(0, 2) - R(2, 0)) * inv_s;
    q.z() = (R(1, 0) - R(0, 1)) * inv_s;
  } else {
    if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
      const double s = 2 * sqrt(1 + R(0, 0) - R(1, 1) - R(2, 2));
      const double inv_s = 1.0 / s;
      q.w() = (R(2, 1) - R(1, 2)) * inv_s;
      q.x() = 0.25f * s;
      q.y() = (R(0, 1) + R(1, 0)) * inv_s;
      q.z() = (R(0, 2) + R(2, 0)) * inv_s;
    } else if (R(1, 1) > R(2, 2)) {
      const double s = 2 * sqrt(1 + R(1, 1) - R(0, 0) - R(2, 2));
      const double inv_s = 1.0 / s;
      q.w() = (R(0, 2) - R(2, 0)) * inv_s;
      q.x() = (R(0, 1) + R(1, 0)) * inv_s;
      q.y() = 0.25f * s;
      q.z() = (R(1, 2) + R(2, 1)) * inv_s;
    } else {
      const double s = 2 * sqrt(1.0f + R(2, 2) - R(0, 0) - R(1, 1));
      const double inv_s = 1.0 / s;
      q.w() = (R(1, 0) - R(0, 1)) * inv_s;
      q.x() = (R(0, 2) + R(2, 0)) * inv_s;
      q.y() = (R(1, 2) + R(2, 1)) * inv_s;
      q.z() = 0.25f * s;
    }
  }

  if (!(std::abs(q.squaredNorm() - 1.0) < 1e-5)) {
    std::cout << "!!!!error quanterion" << std::endl;
    q.normalize();
  }

  Eigen::AngleAxisd angle_axis;
  const double sin_half_theta = q.vec().norm();
  if (sin_half_theta <= std::numeric_limits<double>::epsilon()) {
    angle_axis.angle() = 0;
    angle_axis.axis() << 1, 0, 0;
    return angle_axis;
  }

  double theta;
  const double cos_half_theta = q.w();
  if (cos_half_theta > 0.0) {
    theta = 2 * atan2(sin_half_theta, cos_half_theta);
  } else {
    theta = -2 * atan2(sin_half_theta, -cos_half_theta);
  }

  angle_axis.angle() = theta;
  angle_axis.axis() = q.vec() / sin_half_theta;
  return angle_axis;
}

Eigen::Vector3d rminus(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) {
  Eigen::Matrix3d R_2_1 = R2.inverse() * R1;
  Eigen::AngleAxisd angle_axis = convertRotationMatrixToAngleAxis(R_2_1);
  return angle_axis.angle() * angle_axis.axis();
}

Eigen::Vector6d bundleMinus(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2) {
  Eigen::Matrix3d R1 = T1.topLeftCorner(3, 3);
  Eigen::Matrix3d R2 = T2.topLeftCorner(3, 3);
  Eigen::Vector3d t1 = T1.topRightCorner(3, 1);
  Eigen::Vector3d t2 = T2.topRightCorner(3, 1);

  Eigen::Vector6d delta;
  delta.head<3>() = rminus(R1, R2);
  delta.tail<3>() = t1 - t2;
  return delta;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_dr_calib");
  Eigen::Matrix4d T1;
  T1 << 0.00157428,-0.998757,0.0498268,1.65789,
0.999996,0.00144208,-0.00268897,0.021756,
0.00261376,0.0498308,0.998754,1.89985,
          0,           0,           0,           1;
  Eigen::Matrix4d T2;
  T2 << 0.004817, -0.998671,  0.051316,   1.64046,
 0.999967,  0.004473, -0.006811,  0.010228,
 0.006572,  0.051348,  0.998659,   1.92772,
0,0,0,1;

  Eigen::Vector6d err = bundleMinus(T1, T2);
  std::cout << "rot err: " << err.head<3>().norm() << std::endl
            << "trans err: " << err.tail<3>().norm() << std::endl;

  // if (err.head<3>().norm() < 0.005 && err.tail<3>().norm() < 0.01) {
  //   return true; //ok
  // }

  return 0;
}


