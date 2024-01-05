#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double* x, const double* delta,
                                    double* x_plus_delta) const {
  Eigen::Map<const Vector6d> pose_vec(x);
  Eigen::Map<const Vector6d> delta_pose_vec(delta);
  Eigen::Map<Vector6d> output_pose_vec(x_plus_delta);

  Eigen::Matrix4d pose = exp_T(pose_vec);
  Eigen::Matrix4d pose_out = bundlePlus(pose, delta_pose_vec);
  output_pose_vec = log(pose_out);

  return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double* x, double* jacobian) const {
  if (jacobian != nullptr) {
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacobian_mat(jacobian);
    jacobian_mat.setIdentity();
  }

  return true;
}

int PoseLocalParameterization::GlobalSize() const { return 6; }

int PoseLocalParameterization::LocalSize() const { return 6; }