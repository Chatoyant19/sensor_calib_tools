#include "ceres/ceres.h"
#include "include/common.h"
#include "include/multi_dr_lidar_calib.hpp"
#include "include/pose_local_parameterization.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>


using namespace std;

Eigen::Vector3d rminus(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2);

Vector6d bundleMinus(const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2);

void roughCalib(Calibration &calibra, const double &search_resolution, const int &max_iter);

bool use_p2line_;

template<typename T>
Eigen::Matrix<T, 3, 3> AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1> &rvec) {
    T angle = rvec.norm();
    if (angle == T(0)) {
        return Eigen::Matrix<T, 3, 3>::Identity();
    }

    Eigen::Matrix<T, 3, 1> axis;
    axis = rvec.normalized();

    Eigen::Matrix<T, 3, 3> rmat;
    rmat = Eigen::AngleAxis<T>(angle, axis);

    return rmat;
}

// pnp calib with direction vector
class vpnp_calib_undistort {
public:
    vpnp_calib_undistort(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
                         Eigen::Vector4d _distor) {
        pd = p;
        pd.direction.normalize();
        R_dr_C = _R_dr_C;
        t_dr_C = _t_dr_C;
        inner = _inner;
        distor = _distor;
    }

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
            if (use_p2line_) {
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
        }
        // residuals[0] = (ud - T(pd.u)) * T(pd.direction(1)) - (vd - T(pd.v)) * T(pd.direction(0));
        return true;
    }

    static ceres::CostFunction *
    Create(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
           Eigen::Vector4d _distor) {
        return (new ceres::AutoDiffCostFunction<vpnp_calib_undistort, 2, 4, 3>(
                new vpnp_calib_undistort(p, _R_dr_C, _t_dr_C, _inner, _distor)));
    }

private:
    VPnPData pd;
    Eigen::Matrix3d inner;
    // Distortion coefficient
    Eigen::Vector4d distor;
    Eigen::Matrix3d R_dr_C; // fix
    Eigen::Vector3d t_dr_C; // fix
};


// // AutoDiff, input param is 6X1
// class vpnp_calib {
//  public:
//   vpnp_calib(VPnPData p) { pd = p; }
//   template <typename T>
//   bool operator()(const T *_param, T *residuals) const {
//     Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
//     Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
//     Eigen::Matrix<T, 3, 1> rvec_incre{_param[0], _param[1], _param[2]};
//     Eigen::Matrix<T, 3, 1> t_incre{_param[3], _param[4], _param[5]};
//     Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
//     Eigen::Matrix<T, 3, 3> R_dr_C_T = R_dr_C.cast<T>();
//     Eigen::Matrix<T, 3, 1> t_dr_C_T = t_dr_C.cast<T>();

//     Eigen::Matrix<T,3,3> R_dr_L_T = AngleAxisToRotationMatrix<T>(rvec_incre);
//     Eigen::Matrix<T, 3, 1> p_c = R_dr_C_T.inverse() * R_dr_L_T * p_l + (R_dr_C_T.inverse() * t_incre - 
//       R_dr_C_T.inverse() * t_dr_C_T);

//     const T &fx = innerT.coeffRef(0, 0);
//     const T &cx = innerT.coeffRef(0, 2);
//     const T &fy = innerT.coeffRef(1, 1);
//     const T &cy = innerT.coeffRef(1, 2);
//     T a = p_c[0] / p_c[2];
//     T b = p_c[1] / p_c[2];
//     T r = sqrt(a * a + b * b);
//     T theta = atan(r);
//     T theta_d = theta *
//       (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
//         distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

//     T dx = (theta_d / r) * a;
//     T dy = (theta_d / r) * b;
//     T ud = fx * dx + cx;
//     T vd = fy * dy + cy;
//     residuals[0] = ud - T(pd.u);
//     residuals[1] = vd - T(pd.v);

//     return true;
//   }
//   static ceres::CostFunction *Create(VPnPData p) {
//     return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 6>(
//         new vpnp_calib(p)));
//   }

//  private:
//   VPnPData pd;
// };

// autoDiff, input param is (4+3)
// to refine Tx_dr_L, not Tx_C_L
// fisheye
class vpnp_calib {
public:
    vpnp_calib(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
               Eigen::Vector4d _distor) {
        pd = p;
        R_dr_C = _R_dr_C;
        t_dr_C = _t_dr_C;
        inner = _inner;
        distor = _distor;
    }

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
                R_dr_C_T.inverse() * q_incre.toRotationMatrix() * p_l + (R_dr_C_T.inverse() * t_incre -
                                                                         R_dr_C_T.inverse() * t_dr_C_T);
        // Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
        // Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
        const T &fx = innerT.coeffRef(0, 0);
        const T &cx = innerT.coeffRef(0, 2);
        const T &fy = innerT.coeffRef(1, 1);
        const T &cy = innerT.coeffRef(1, 2);

        T a = p_c[0] / p_c[2];
        T b = p_c[1] / p_c[2];
        // assert(!ceres::IsNaN(a));
        // assert(!ceres::IsNaN(b));
        T r = sqrt(a * a + b * b);
        T theta = atan(r);
        T theta_d = theta *
                    (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
                     distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

        T dx = (theta_d / r) * a;
        T dy = (theta_d / r) * b;
        T ud = fx * dx + cx;
        T vd = fy * dy + cy;
        residuals[0] = ud - T(pd.u);
        residuals[1] = vd - T(pd.v);

        return true;
    }

    static ceres::CostFunction *
    Create(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
           Eigen::Vector4d _distor) {
        return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(
                new vpnp_calib(p, _R_dr_C, _t_dr_C, _inner, _distor)));
    }

private:
    VPnPData pd;
    Eigen::Matrix3d inner;
    // Distortion coefficient
    Eigen::Vector4d distor;
    Eigen::Matrix3d R_dr_C; // fix
    Eigen::Vector3d t_dr_C; // fix
};

// analy diff
class vpnp_calib_new : public ceres::SizedCostFunction<2, 6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    vpnp_calib_new(VPnPData p, Eigen::Matrix3d _R_dr_C, Eigen::Vector3d _t_dr_C, Eigen::Matrix3d _inner,
                   Eigen::Vector4d _distor) {
        pd = p;
        R_dr_C = _R_dr_C;
        t_dr_C = _t_dr_C;
        inner = _inner;
        distor = _distor;
    }

    ~vpnp_calib_new() {}

    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> pose_vec(parameters[0]);   //Tx_dr_L
        Eigen::Map<Eigen::Matrix<double, 2, 1>> residual(residuals);

        Eigen::Matrix3d R_dr_L = exp_R(Eigen::Vector3d(pose_vec[0], pose_vec[1], pose_vec[2]));
        Eigen::Vector3d t_dr_L = Eigen::Vector3d(Eigen::Vector3d(pose_vec[3], pose_vec[4], pose_vec[5]));
        // std::cout << "vpnp_calib_new- R_dr_L: " << R_dr_L << std::endl
        //           << "- t_dr_L: " << t_dr_L << std::endl;

        Eigen::Vector3d p_l = Eigen::Vector3d(pd.x, pd.y, pd.z);

        Eigen::Vector3d p_dr = R_dr_L * p_l + t_dr_L;
        Eigen::Vector3d p_c = R_dr_C.inverse() * R_dr_L * p_l +
                              (R_dr_C.inverse() * t_dr_L - R_dr_C.inverse() * t_dr_C);

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
            Eigen::Matrix3d R_C_dr = R_dr_C.inverse();
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
        return (new vpnp_calib_new(p, _R_dr_C, _t_dr_C, _inner, _distor));
    }

private:
    VPnPData pd;
    Eigen::Matrix3d inner;
    // Distortion coefficient
    Eigen::Vector4d distor;
    Eigen::Matrix3d R_dr_C; // fix
    Eigen::Vector3d t_dr_C; // fix
};

int main(int argc, char **argv) {

    time_t t_start = clock();
    const std::string CalibSettingPath = std::string(argv[1]);
    const std::string ResultPath = std::string(argv[2]);
    const int debugMode = std::stoi(argv[3]);
    std::cout << "---argc num:" << argc << std::endl;
    std::cout << "CalibSettingPath: " << CalibSettingPath << std::endl
              << "ResultPath: " << ResultPath << std::endl
              << "debugMode: " << debugMode << std::endl;

    Calibration calib(CalibSettingPath, ResultPath, debugMode);

    if (calib.match_type_ == 0) {
        std::cout << "======P2P" << std::endl;
        use_p2line_ = false;
    } else if (calib.match_type_ == 1) {
        std::cout << "======P2Line" << std::endl;
        use_p2line_ = true;
    }

    // bool use_rough_calib = false;
    // if (use_rough_calib) {
    //   roughCalib(calib, DEG2RAD(0.1), 30);
    // }

    // std::cout << "after rough calib, " << std::endl;
    // for(int cam_index; cam_index < calib.cams.size(); ++cam_index) {
    //   std::cout << "******[" << calib.cams[cam_index].cam_name_ << "]***** Tx_C_L: " << std::endl
    //             << calib.cams[cam_index].Tx_C_L_ << std::endl;
    // }

    // refine Tx_dr_L
    Eigen::Matrix3d R_dr_L /*= calib.lidar.R_dr_L_*/;
    Eigen::Vector3d t_dr_L /*= calib.lidar.t_dr_L_*/;
    // std::cout << "if not roughCalib, Tx_dr_L should be equal input init value" << std::endl
    //           << "wd check R_dr_L: " << R_dr_L << std::endl
    //           << "wd check t_dr_L: " << t_dr_L << std::endl;

    int iter = 0;

    // Maximum match distance threshold: 15 pixels
    // If initial extrinsic lead to error over 15 pixels, the algorithm will not work/ camNum
    // Iteratively reducve the matching distance threshold
    // bool test_flag = false;
    int low_dis_threshold = 6;
    for (int dis_threshold = 20; dis_threshold > low_dis_threshold; dis_threshold -= 1) {
        // if(test_flag) break;
        // For each distance, do twice optimization
        for (int cnt = 0; cnt < 6; cnt++) {
            ceres::Problem problem;
            R_dr_L = calib.lidar.R_dr_L_;
            t_dr_L = calib.lidar.t_dr_L_;
            // std::cout << "refining R_dr_L: " << R_dr_L << std::endl;
            // std::cout << "refining t_dr_L: " << t_dr_L << std::endl;

            /*****自动求导******/

            Eigen::Quaterniond q(R_dr_L);
            double ext[7];
            ext[0] = q.x();
            ext[1] = q.y();
            ext[2] = q.z();
            ext[3] = q.w();
            ext[4] = t_dr_L[0];
            ext[5] = t_dr_L[1];
            ext[6] = t_dr_L[2];
            Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
            Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);
            ceres::LocalParameterization *q_parameterization =
                    new ceres::EigenQuaternionParameterization();
            // ceres::Problem problem;
            problem.AddParameterBlock(ext, 4, q_parameterization);
            problem.AddParameterBlock(ext + 4, 3);

            Eigen::Matrix3d rot;


            /****** 解析求导 *****/
            /*
            double parameter[6];
            Vector6d param_eigen = log_R_t(R_dr_L, t_dr_L);

            for (size_t i = 0; i < 6; i++) {
              parameter[i] = param_eigen[i];
            }

            problem.AddParameterBlock(parameter, 6, new PoseLocalParameterization());
            */

            // if(test_flag) break;
            int total_vpnp_size = 0;
            for (int camNum = 0; camNum < calib.cams.size(); ++camNum) {
                // if(camNum > 3) break; // just refine avm
                std::cout << "******[" << calib.cams[camNum].cam_name_ << "]*****" << std::endl;
                std::vector<std::vector<VPnPData>> vpnp_list_vect;
                int vpnp_size = 0;

                calib.buildVPnp(calib.cams[camNum], dis_threshold,
                                cnt,
                        // true,
                                false,
                                calib.cams[camNum].rgb_edge_clouds_,
                                calib.lidar.plane_line_cloud_vec_,
                        /*calib.lidar.plane_line_number_vec_,*/
                                vpnp_list_vect);

                for (auto vpnp_list: vpnp_list_vect) {
                    vpnp_size += vpnp_list.size();
                }
                total_vpnp_size += vpnp_size;

                if (vpnp_size == 0) {
                    std::cout << "not enough measurement!" << std::endl;
                    continue;
                }

                std::cout << "Iteration:" << iter++ << " Dis:" << dis_threshold
                          << " pnp size: " << vpnp_size << std::endl;

                // // show optimizing image, just show first scene
                // if(debugMode) {
                //   pcl::PointCloud<pcl::PointXYZI>::Ptr exa_pcd = calib.lidar.pcd_vec_[0];
                //   cv::Mat exa_img = calib.cams[camNum].rgb_imgs[0];
                //   cv::Mat projection_img = calib.getProjectionImg(calib.cams[camNum], exa_pcd, exa_img);
                //   if(projection_img.cols > 2000) {
                //     cv::resize(projection_img, projection_img, cv::Size(projection_img.cols/2, projection_img.rows/2));
                //   }
                //   cv::imshow("Optimization", projection_img);
                //   cv::waitKey(100);
                // }

                // instrins matrix
                Eigen::Matrix3d inner;
                // Distortion coefficient
                Eigen::Vector4d distor;
                Eigen::Matrix3d R_dr_C; // fix
                Eigen::Vector3d t_dr_C; // fix
                inner << calib.cams[camNum].camera_matrix_.at<double>(0,
                                                                      0), 0.0, calib.cams[camNum].camera_matrix_.at<double>(
                        0, 2),
                        0.0, calib.cams[camNum].camera_matrix_.at<double>(1,
                                                                          1), calib.cams[camNum].camera_matrix_.at<double>(
                        1, 2),
                        0.0, 0.0, 1.0;
                // distor << calib.cams[camNum].k1_, calib.cams[camNum].k2_, calib.cams[camNum].k3_, calib.cams[camNum].k4_;
                distor << 0.0, 0.0, 0.0, 0.0;

                R_dr_C = calib.cams[camNum].R_dr_C_;
                t_dr_C = calib.cams[camNum].t_dr_C_;

                if (debugMode) {
                    std::cout << "check inner: " << inner << std::endl
                              << "check distor: " << distor << std::endl
                              << "check R_dr_C: " << R_dr_C << std::endl
                              << "check t_dr_C: " << t_dr_C << std::endl;
                }

                /*****自动求导******/

                for (int scene_index = 0; scene_index < calib.scene_num_; ++scene_index) {
                    for (auto val: vpnp_list_vect[scene_index]) {
                        ceres::CostFunction *cost_function;
                        cost_function = vpnp_calib_undistort::Create(val, R_dr_C, t_dr_C, inner, distor);
                        // problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
                        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.3), ext, ext + 4);
                    }
                }


                /*****自动求导: refine value: 6X1******/
                /*
                 double parameter[6];
                 Vector6d param_eigen = log_R_t(R_dr_L, t_dr_L);

                 for (size_t i = 0; i < 6; i++) {
                   parameter[i] = param_eigen[i];
                 }

                 problem.AddParameterBlock(parameter, 6, new PoseLocalParameterization());

                 for (int scene_index = 0; scene_index < calib.scene_num_; ++scene_index) {
                   for (auto val : vpnp_list_vect[scene_index]) {
                     ceres::CostFunction *cost_function;
                     cost_function = vpnp_calib::Create(val);
                     problem.AddResidualBlock(cost_function, NULL, parameter);

                     //   // output jacobian
                     // const double *const parameters[] = {parameter};
                     // double* residuals = new double[2];
                     // double** jacobians = new double *[2];
                     // jacobians[0] = new double[6];
                     // cost_function->Evaluate(( const double *const *)parameters, (double*)residuals, (double**)jacobians);
                     // for(int i = 0; i < 6; ++i) {
                     //   std::cout << "jacobians: " << jacobians[0][i] << std::endl;
                     // }
                   }
                 }
                 */

                /****** 解析求导 *****/
                /*
                for (int scene_index = 0; scene_index < calib.scene_num_; ++scene_index) {
                  for (auto val : vpnp_list_vect[scene_index]) {
                    ceres::CostFunction *cost_function;
                    cost_function = vpnp_calib_new::Create(val, R_dr_C, t_dr_C, inner, distor);
                    // problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.3), parameter);
                    problem.AddResidualBlock(cost_function, NULL, parameter);
                  }
                }
                */

                std::vector<std::vector<VPnPData>>().swap(vpnp_list_vect);

                // if (err.head<3>().norm() < 1e-3 && err.tail<3>().norm() < 0.01) {
                //   test_flag = true;
                //   break;
                // }
            } // camNum

            ceres::Solver::Options options;
            // options.preconditioner_type = ceres::JACOBI;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            options.trust_region_strategy_type = ceres::DOGLEG;
            // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.max_num_iterations = 500;
            options.use_nonmonotonic_steps = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            if (debugMode) {
                // std::cout << summary.BriefReport() << std::endl;
                std::cout << summary.FullReport() << std::endl;
            }

            /*****自动求导******/
            rot = m_q.toRotationMatrix();
            /****** 解析求导 *****/
            /*
            Eigen::Matrix3d rot = exp_R(Eigen::Vector3d(parameter[0], parameter[1], parameter[2]));
            Eigen::Vector3d m_t = Eigen::Vector3d(parameter[3], parameter[4], parameter[5]);
            */

            Eigen::Matrix4d T1;
            T1 << rot(0, 0), rot(0, 1), rot(0, 2), m_t(0),
                    rot(1, 0), rot(1, 1), rot(1, 2), m_t(1),
                    rot(2, 0), rot(2, 1), rot(2, 2), m_t(2),
                    0.0, 0.0, 0.0, 1.0;
            if (debugMode) {
                std::cout << "current Tx_dr_L: " << T1 << std::endl;
            }
            Eigen::Matrix4d T2;
            T2 = calib.lidar.Tx_dr_L_;
            std::cout << "last Tx_dr_L: " << T2 << std::endl;

            Vector6d err = bundleMinus(T1, T2);
            calib.lidar.update_Rt(rot, m_t);

            for (int camNum = 0; camNum < calib.cams.size(); ++camNum) {
                Eigen::Matrix4d Tx_C_L = calib.cams[camNum].Tx_dr_C_.inverse() * calib.lidar.Tx_dr_L_;
                calib.cams[camNum].update_TxCL(Tx_C_L);
            }

            // if (err.head<3>().norm() < 0.0005 && err.tail<3>().norm() < 0.005) {
            //     break;
            // }
        }
    }


    Eigen::Matrix4d T_dr_L_new = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_dr_pr = Eigen::Matrix4d::Identity();
    for (int scene_index = 0; scene_index < calib.scene_num_; ++scene_index) {
        if (calib.addFloorConstriant(calib.lidar.floor_plane_vec_[scene_index], calib.lidar.Tx_dr_L_, T_dr_L_new)) {
            T_dr_pr = T_dr_L_new * (calib.lidar.Tx_dr_L_.inverse());
            calib.lidar.update_T(T_dr_L_new);
        }
    }

    /* output calibrated extrinsic results */
    std::string result_save_path = calib.result_path_ + "/" + calib.save_lidar_extrinsic_name_;
    calib.writeCamExToPbFile(calib.lidar.Tx_dr_L_, result_save_path);

    for (int camNum = 0; camNum < calib.cams.size(); ++camNum) {
        std::cout << "******[" << calib.cams[camNum].cam_name_ << "]*****" << std::endl;
        if (calib.update_camera_extrinsic_) {
            Eigen::Matrix4d T_dr_C_new = T_dr_pr * calib.cams[camNum].Tx_dr_C_;
            calib.cams[camNum].update_T(T_dr_C_new);

            // std::cout << calib.cams[camNum].cam_name_ << " T_C_L: " << calib.cams[camNum].Tx_C_L_ << std::endl;

            std::string cam_extrinsic_save_path =
                    calib.result_path_ + "/" + calib.cams[camNum].cam_name_ + "_transform.pb.txt";
            calib.writeCamExToPbFile(T_dr_C_new, cam_extrinsic_save_path);
        }

            for (int scene_index = 0; scene_index < calib.scene_num_; ++scene_index) {
                // pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcd = calib.lidar.pcd_vec_[scene_index];
                std::string visual_pcd_path = calib.pcd_path_ + "/" + std::to_string(scene_index) + "_visual.pcd";
                pcl::PointCloud<pcl::PointXYZI>::Ptr raw_point(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::io::loadPCDFile(visual_pcd_path, *raw_point);
                // cv::Mat rgb_image = calib.cams[camNum].rgb_imgs[scene_index];
                cv::Mat rgb_image = calib.cams[camNum].raw_imgs[scene_index];
                cv::Mat opt_img = calib.getProjectionImg(calib.cams[camNum], raw_point, rgb_image);
                // cv::Mat opt_img = calib.showPcdOnImg(calib.cams[camNum], raw_pcd, rgb_image);

                cv::imwrite(calib.result_path_ + "/" + calib.cams[camNum].cam_name_ + "_sceneID_" +
                            std::to_string(scene_index) + "_result.png", opt_img);
            }
        
    }// camNum


    time_t t_end = clock();
    std::cout << "calib process use time: " << (double) (t_end - t_start) / (CLOCKS_PER_SEC) << "s" << std::endl;

    return 0;
}

Eigen::Vector3d rminus(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2) {
    Eigen::Matrix3d R_2_1 = R2.inverse() * R1;
    Eigen::AngleAxisd angle_axis = convertRotationMatrixToAngleAxis(R_2_1);
    return angle_axis.angle() * angle_axis.axis();
}

Vector6d bundleMinus(const Eigen::Matrix4d &T1, const Eigen::Matrix4d &T2) {
    Eigen::Matrix3d R1 = T1.topLeftCorner(3, 3);
    Eigen::Matrix3d R2 = T2.topLeftCorner(3, 3);
    Eigen::Vector3d t1 = T1.topRightCorner(3, 1);
    Eigen::Vector3d t2 = T2.topRightCorner(3, 1);

    Vector6d delta;
    delta.head<3>() = rminus(R1, R2);
    delta.tail<3>() = t1 - t2;
    return delta;
}

void roughCalib(Calibration &calibra, const double &search_resolution, const int &max_iter) {
    float match_dis = 20;
    Eigen::Vector3d fix_adjust_euler(0, 0, 0);

    ROS_INFO_STREAM("roughCalib");

    for (int n = 0; n < 2; n++) {
        for (int round = 0; round < 3; ++round) {
            for (int cam_index = 0; cam_index < calibra.cams.size(); ++cam_index) {
                Eigen::Matrix3d rot = calibra.cams[cam_index].R_C_L_;
                Eigen::Vector3d translation = calibra.cams[cam_index].t_C_L_;
                std::cout << "rot(R_C_L): " << rot << std::endl
                          << "trans(t_C_L): " << translation << std::endl;

                float min_cost = 1000;
                for (int iter = 0; iter < max_iter; ++iter) {
                    Eigen::Vector3d adjust_euler = fix_adjust_euler;
                    adjust_euler[round] = fix_adjust_euler[round] + pow(-1, iter) * int(iter / 2) * search_resolution;
                    Eigen::Matrix3d adjust_rotation_matrix;
                    adjust_rotation_matrix =
                            Eigen::AngleAxisd(adjust_euler[0], Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(adjust_euler[1], Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(adjust_euler[2], Eigen::Vector3d::UnitX());
                    Eigen::Matrix3d test_rot = rot * adjust_rotation_matrix;
                    Eigen::Matrix4d test_params;
                    test_params << test_rot(0, 0), test_rot(0, 1), test_rot(0, 2), translation(0),
                            test_rot(1, 0), test_rot(1, 1), test_rot(1, 2), translation(1),
                            test_rot(2, 0), test_rot(2, 1), test_rot(2, 2), translation(2),
                            0.0, 0.0, 0.0, 1.0;
                    std::vector<std::vector<VPnPData>> pnp_list_vec;

                    calibra.buildVPnp(calibra.cams[cam_index], match_dis,
                                      false,
                                      calibra.cams[cam_index].rgb_edge_clouds_,
                                      calibra.lidar.plane_line_cloud_vec_,
                            /*calibra.lidar.plane_line_number_vec_,*/
                                      pnp_list_vec);

                    int total_edge_size = 0;
                    int average_edge_size = 0;
                    int total_pnp_size = 0;
                    int average_pnp_size = 0;
                    for (int scene_index = 0; scene_index < calibra.scene_num_; ++scene_index) {
                        total_edge_size += calibra.lidar.plane_line_cloud_vec_[scene_index]->size();
                        total_pnp_size += pnp_list_vec[scene_index].size();
                    }
                    average_edge_size = total_edge_size / calibra.scene_num_;
                    average_pnp_size = total_pnp_size / calibra.scene_num_;
                    float cost = ((float) (average_edge_size - average_pnp_size) / (float) average_edge_size);
#ifdef debug_mode
                    std::cout << "n " << n << " round " << round << " cam_index " << cam_index << " iter "
                              << iter << " cost:" << cost << std::endl;
#endif
                    if (cost < min_cost) {
                        ROS_INFO_STREAM("cost " << cost << " edge size "
                                                << average_edge_size
                                                << " pnp_list size " << average_pnp_size);
                        min_cost = cost;
                        calibra.cams[cam_index].update_TxCL(test_params);
                        calibra.buildVPnp(calibra.cams[cam_index], match_dis,
                                          true,
                                          calibra.cams[cam_index].rgb_edge_clouds_,
                                          calibra.lidar.plane_line_cloud_vec_,
                                /*calibra.lidar.plane_line_number_vec_,*/
                                          pnp_list_vec);

                        pcl::PointCloud<pcl::PointXYZI>::Ptr exa_pcd = calibra.lidar.pcd_vec_[0];
                        cv::Mat exa_img = calibra.cams[cam_index].rgb_imgs[0];
                        cv::Mat projection_img = calibra.getProjectionImg(calibra.cams[cam_index], exa_pcd, exa_img);

                        std::string img_name = calibra.cams[cam_index].cam_name_ + "_" + "scene_0" + "_projection";
                        cv::imshow(img_name, projection_img);
                        cv::waitKey(10);
                    }
                }
            }
        }
    }
}