#include <ceres/ceres.h>

#include "common.h"
#include "pose_local_parameterization.h"
#include "eigen_types.hpp"

// pnp calib with direction vector
// auto diff, pinhole camera intrinsics model
class vpnp_calib_orin {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  vpnp_calib_orin(VPnPData p, Eigen::Matrix3d _inner, Eigen::Vector4d _distor) :
    pd(p), inner(_inner), distor(_distor) {}
  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    } else {
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);
      Eigen::Matrix<T, 2, 2> I =
          Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;
      Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 0) = residuals[1];
      R = V * R;
      // Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<float, 2,
      // 2>::Zero().cast<T>(); R.coeffRef(0, 0) = residuals[0];
      // R.coeffRef(1, 1) = residuals[1]; R = V * R * V.transpose();
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 0);
      
    }
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p, Eigen::Matrix3d _inner, Eigen::Vector4d _distor) {
    return (new ceres::AutoDiffCostFunction<vpnp_calib_orin, 2, 4, 3>(
        new vpnp_calib_orin(p, _inner, _distor)));
  }

private:
  VPnPData pd;
  Eigen::Matrix3d inner;
  Eigen::Vector4d distor;
};

// vpnp calib with direction vector
// pinhole camera intrinsics model
// auto diff
class vpnp_calib_pin_auto {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vpnp_calib_pin_auto(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
                      Eigen::Vector4d _distor) : 
    pd(p), R_dr_C(_R_dr_C), t_dr_C(_t_dr_C), inner(_inner), distor(_distor) {}

  template<typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 3> R_dr_C_T = R_dr_C.cast<T>();
    Eigen::Matrix<T, 3, 1> t_dr_C_T = t_dr_C.cast<T>();
    Eigen::Matrix<T, 3, 1> p_c =
      R_dr_C_T.transpose() * q_incre.toRotationMatrix() * p_l + (R_dr_C_T.transpose() * t_incre -
      R_dr_C_T.transpose() * t_dr_C_T);
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    } else {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
     
      Eigen::Matrix<T, 2, 2> I =
              Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;
      Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<float, 2, 2>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 1) = residuals[1];
      R = V * R * V.transpose();
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 1);
    }
    // residuals[0] = (ud - T(pd.u)) * T(pd.direction(1)) - (vd - T(pd.v)) * T(pd.direction(0));
    return true;
  }

  static ceres::CostFunction *
  Create(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
      Eigen::Vector4d _distor) {
    return (new ceres::AutoDiffCostFunction<vpnp_calib_pin_auto, 2, 4, 3>(
            new vpnp_calib_pin_auto(p, _R_dr_C, _t_dr_C, _inner, _distor)));
  }

 private:
  VPnPData pd;
  Eigen::Matrix3d inner;
  // Distortion coefficient
  Eigen::Vector4d distor;
  Eigen::Matrix3d R_dr_C; // fix
  Eigen::Vector3d t_dr_C; // fix
};

// analy diff, fisheye camera intrinsics model, p2p
// just to study, not using it!
class vpnp_calib_fisheye_ana : public ceres::SizedCostFunction<2, 6> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vpnp_calib_fisheye_ana(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
                           Eigen::Vector4d _distor) : pd(p), R_dr_C(_R_dr_C), t_dr_C(_t_dr_C), inner(_inner), distor(_distor) {}

    ~vpnp_calib_fisheye_ana() {}

    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> pose_vec(parameters[0]);   //Tx_dr_L
        Eigen::Map<Eigen::Matrix<double, 2, 1>> residual(residuals);

        Eigen::Matrix3d R_dr_L = exp_R(Eigen::Vector3d(pose_vec[0], pose_vec[1], pose_vec[2]));
        Eigen::Vector3d t_dr_L = Eigen::Vector3d(Eigen::Vector3d(pose_vec[3], pose_vec[4], pose_vec[5]));
        // std::cout << "vpnp_calib_fisheye_ana- R_dr_L: " << R_dr_L << std::endl
        //           << "- t_dr_L: " << t_dr_L << std::endl;

        Eigen::Vector3d p_l = Eigen::Vector3d(pd.x, pd.y, pd.z);

        Eigen::Vector3d p_dr = R_dr_L * p_l + t_dr_L;
        Eigen::Vector3d p_c = R_dr_C.transpose() * R_dr_L * p_l +
                              (R_dr_C.transpose() * t_dr_L - R_dr_C.transpose() * t_dr_C);

        double a = p_c[0] / p_c[2];
        double b = p_c[1] / p_c[2];
        double r = sqrt(a * a + b * b);
        double theta = atan(r);
        double theta_d = theta *
                         (1.0 + distor[0] * pow(theta, 2) + distor[1] * pow(theta, 4) +
                          distor[2] * pow(theta, 6) + distor[3] * pow(theta, 8));

        double dx = (theta_d / r) * a;
        double dy = (theta_d / r) * b;

        double fx = inner(0, 0);
        double fy = inner(1, 1);
        double cx = inner(0, 2);
        double cy = inner(1, 2);
        double ud = fx * dx + cx;
        double vd = fy * dy + cy;
        residual[0] = ud - pd.u;
        residual[1] = vd - pd.v;

        // calculate jacobians
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            // std::cout << "cal jacobians" << std::endl;

            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian(jacobians[0]);

            double tmp_1 = a / r;
            double tmp_2 = (a * theta_d) / (r * r);
            double tmp_3 = theta_d / r;
            double tmp_4 = b / r;
            double tmp_5 = (b * theta_d) / (r * r);

            double X2_Y2_Z2 = p_c(0) * p_c(0) + p_c(1) * p_c(1) + p_c(2) * p_c(2);
            double X2_Y2 = p_c(0) * p_c(0) + p_c(1) * p_c(1);
            double dev_theta_pc_X = (p_c(0) * p_c(2)) / (X2_Y2_Z2 * sqrt(X2_Y2));
            double theta_2 = theta * theta;
            double theta_4 = theta_2 * theta_2;
            double theta_6 = theta_4 * theta_2;
            double theta_8 = theta_4 * theta_4;
            double dev_theta_d_pc_X = dev_theta_pc_X + 3.0 * distor[0] * theta_2 * dev_theta_pc_X +
                                      5.0 * distor[1] * theta_4 * dev_theta_pc_X +
                                      7.0 * distor[2] * theta_6 * dev_theta_pc_X +
                                      9.0 * distor[3] * theta_8 * dev_theta_pc_X;
            double dev_r_pc_X = p_c(0) / (p_c(2) * sqrt(X2_Y2));
            double dev_a_pc_X = 1.0 / p_c(2);
            double dev_dx_pc_X = tmp_1 * dev_theta_d_pc_X - tmp_2 * dev_r_pc_X +
                                 tmp_3 * dev_a_pc_X;

            double dev_theta_pc_Y = (p_c(1) * p_c(2)) / (X2_Y2_Z2 * sqrt(X2_Y2));
            double dev_theta_d_pc_Y = dev_theta_pc_Y + 3.0 * distor[0] * theta_2 * dev_theta_pc_Y +
                                      5.0 * distor[1] * theta_4 * dev_theta_pc_Y +
                                      7.0 * distor[2] * theta_6 * dev_theta_pc_Y +
                                      9.0 * distor[3] * theta_8 * dev_theta_pc_Y;
            double dev_r_pc_Y = p_c(1) / (p_c(2) * sqrt(X2_Y2));
            double dev_dx_pc_Y = tmp_1 * dev_theta_d_pc_Y - tmp_2 * dev_r_pc_Y;

            double dev_theta_pc_Z = -(sqrt(X2_Y2) / X2_Y2_Z2);
            double dev_theta_d_pc_Z = dev_theta_pc_Z + 3.0 * distor[0] * theta_2 * dev_theta_pc_Z +
                                      5.0 * distor[1] * theta_4 * dev_theta_pc_Z +
                                      7.0 * distor[2] * theta_6 * dev_theta_pc_Z +
                                      9.0 * distor[3] * theta_8 * dev_theta_pc_Z;
            double dev_r_pc_Z = -(sqrt(X2_Y2) / (p_c(2) * p_c(2)));
            double dev_a_pc_Z = -(p_c(0) / (p_c(2) * p_c(2)));
            double dev_dx_pc_Z = tmp_1 * dev_theta_d_pc_Z - tmp_2 * dev_r_pc_Z + tmp_3 * dev_a_pc_Z;

            double dev_dy_pc_X = tmp_4 * dev_theta_d_pc_X - tmp_5 * dev_r_pc_X;

            double dev_b_pc_Y = 1.0 / p_c(2);
            double dev_dy_pc_Y = tmp_4 * dev_theta_d_pc_Y - tmp_5 * dev_r_pc_Y + tmp_3 * dev_b_pc_Y;

            double dev_b_pc_Z = -(p_c(1) / (p_c(2) * p_c(2)));
            double dev_dy_pc_Z = tmp_4 * dev_theta_d_pc_Z - tmp_5 * dev_r_pc_Z + tmp_3 * dev_b_pc_Z;

            Eigen::Matrix3d p_l_cross;
            p_l_cross << 0, -p_l(2), p_l(1),
                    p_l(2), 0, -p_l(0),
                    -p_l(1), p_l(0), 0;

            Eigen::Matrix<double, 2, 3> _uv_jacobian;
            _uv_jacobian(0, 0) = fx * dev_dx_pc_X;
            _uv_jacobian(0, 1) = fx * dev_dx_pc_Y;
            _uv_jacobian(0, 2) = fx * dev_dx_pc_Z;

            _uv_jacobian(1, 0) = fy * dev_dy_pc_X;
            _uv_jacobian(1, 1) = fy * dev_dy_pc_Y;
            _uv_jacobian(1, 2) = fy * dev_dy_pc_Z;

            Eigen::Matrix<double, 3, 6> _pose_jacobian;
            Eigen::Matrix3d R_C_dr = R_dr_C.transpose();
            _pose_jacobian.block<3, 3>(0, 0) = -R_dr_L * p_l_cross;
            _pose_jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            // _pose_jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

            jacobian = _uv_jacobian * R_C_dr * _pose_jacobian;
        }
        return true;
    }

    static ceres::CostFunction *
    Create(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
           Eigen::Vector4d _distor) {
        return (new vpnp_calib_fisheye_ana(p, _R_dr_C, _t_dr_C, _inner, _distor));
    }

private:
    VPnPData pd;
    Eigen::Matrix3d inner;
    Eigen::Vector4d distor;
    Eigen::Matrix3d R_dr_C; // fix
    Eigen::Vector3d t_dr_C; // fix
};