#pragma once

#include <Eigen/Dense>

// #include "eigen_types.hpp"

template <typename T>
struct LinearInterpolaterTrait {
  static T plus(const T& x, const T& delta) { return x + delta; }

  static T between(const T& lhs, const T& rhs) { return -lhs + rhs; }
};

template <>
struct LinearInterpolaterTrait<Eigen::Matrix4d> {
  typedef Eigen::Matrix4d T;

  static T plus(const T& x, const T& delta) { 
    Eigen::Matrix4d Tx_12 = Eigen::Matrix4d::Identity();
    Tx_12.block<3, 3>(0, 0) = x.block<3, 3>(0, 0) * delta.block<3, 3>(0, 0);
    Tx_12.block<3, 1>(0, 3) = x.block<3, 1>(0, 3) + delta.block<3, 1>(0, 3);
    return Tx_12;
   }

  static Eigen::Matrix4d between(const T& lhs, const T& rhs) { 
    Eigen::Matrix4d Tx_21 = Eigen::Matrix4d::Identity();
    Tx_21.block<3, 3>(0, 0) = lhs.block<3, 3>(0, 0).transpose() * rhs.block<3, 3>(0, 0);
    Tx_21.block<3, 1>(0, 3) = rhs.block<3, 1>(0, 3) - lhs.block<3, 1>(0, 3);
    Tx_21; 
  }
};
