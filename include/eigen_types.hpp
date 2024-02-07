#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

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


// Vector6d log_R_t(const Eigen::Matrix3d R, const Eigen::Vector3d t) {
//   Vector6d tf_vec;
//   Eigen::AngleAxisd angle_axis = convertRotationMatrixToAngleAxis(R);
//   Eigen::Vector3d rvec = angle_axis.angle() * angle_axis.axis();
//   tf_vec.head<3>() = rvec;
//   tf_vec.tail<3>() = t;
//   return tf_vec;
// }

Eigen::Matrix3d exp_R(const Eigen::Vector3d& rotation_vec) {
  const double half_angle = rotation_vec.norm() / 2.0;
  if (half_angle <= std::numeric_limits<double>::epsilon()) {
    return (Eigen::Matrix3d::Identity());
  }
  Eigen::Quaterniond q;
  q.w() = cos(half_angle);
 
  Eigen::Vector3d vec = rotation_vec.normalized();
  q.vec() = sin(half_angle) * vec;
  return (q.toRotationMatrix());
}

Eigen::Matrix4d exp_T(const Vector6d& tf) {
  Eigen::Matrix4d T_exp = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d R_exp = exp_R(tf.head<3>());
  T_exp.block<3, 3>(0, 0) = R_exp;
  T_exp.block<3, 1>(0, 3) = tf.tail<3>();
  return T_exp;
}

Eigen::Matrix3d rplus(const Eigen::Matrix3d& R, const Eigen::Vector3d& rot_vec) { 
  return (R * exp_R(rot_vec));
}

Eigen::Matrix3d lplus(const Eigen::Matrix3d& R, const Eigen::Vector3d& rot_vec) {
  return (exp_R(rot_vec) * R);
}

Eigen::Matrix4d bundlePlus(const Eigen::Matrix4d& pose, const Vector6d& delta) {
  Eigen::Matrix4d res = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d pose_R = pose.topLeftCorner(3, 3);
  
  Eigen::Matrix3d res_R = rplus(pose_R, delta.head<3>());
  // Eigen::Matrix3d res_R = lplus(pose_R, delta.head<3>());
  Eigen::Vector3d res_t = pose.topRightCorner(3, 1) + delta.tail<3>();
  res.block<3, 3>(0, 0) = res_R;
  res.block<3, 1>(0, 3) = res_t;
  return res;
}

Vector6d log(const Eigen::Matrix4d T) {
  Vector6d tf_vec;
  Eigen::Matrix3d R = T.topLeftCorner(3, 3);
  Eigen::Vector3d t = T.topRightCorner(3, 1);
  Eigen::AngleAxisd angle_axis = convertRotationMatrixToAngleAxis(R);
  Eigen::Vector3d rvec = angle_axis.angle() * angle_axis.axis();
  tf_vec.head<3>() = rvec;
  tf_vec.tail<3>() = t;
  return tf_vec;
}
