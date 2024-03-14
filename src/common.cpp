#include "common.h"

void assign_qt(Eigen::Quaterniond& q, Eigen::Vector3d& t,
               const Eigen::Quaterniond& q_, const Eigen::Vector3d& t_) {
  q.w() = q_.w(); q.x() = q_.x(); q.y() = q_.y(); q.z() = q_.z();
  t(0) = t_(0); t(1) = t_(1); t(2) = t_(2);
}

Eigen::Matrix3d wedge(const Eigen::Vector3d& v) {
	Eigen::Matrix3d V;
	V << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
	return V;
}
