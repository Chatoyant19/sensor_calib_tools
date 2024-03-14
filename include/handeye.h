#ifndef HANDEYE
#define HANDEYE

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

typedef std::pair<double, Eigen::Matrix4d> StampedPose;
typedef std::vector<StampedPose, Eigen::aligned_allocator<StampedPose>> StampedPoseVector;
typedef std::shared_ptr<StampedPoseVector> StampedPoseVectorPtr;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vector_vec3d;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> vector_quad;

template<typename T>
Eigen::Matrix<T, 3, 1> RotationToAngleAxis(const Eigen::Matrix<T, 3, 3> & rmat)
{
    Eigen::AngleAxis<T> angleaxis;
    angleaxis.fromRotationMatrix(rmat);
    return angleaxis.angle() * angleaxis.axis();

}

template<typename T>
Eigen::Matrix<T, 3, 3> AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1>& rvec)
{
    T angle = rvec.norm();
    if (angle == T(0))
    {
        return Eigen::Matrix<T, 3, 3>::Identity();
    }

    Eigen::Matrix<T, 3, 1> axis;
    axis = rvec.normalized();

    Eigen::Matrix<T, 3, 3> rmat;
    rmat = Eigen::AngleAxis<T>(angle, axis);

    return rmat;
}

template<typename T>
Eigen::Quaternion<T> AngleAxisToQuaternion(const Eigen::Matrix<T, 3, 1>& rvec)
{
    Eigen::Matrix<T, 3, 3> rmat = AngleAxisToRotationMatrix<T>(rvec);

    return Eigen::Quaternion<T>(rmat);
}


template<typename T>
Eigen::Matrix<T, 3, 3> QuaternionToRotation(const T* const q)
{
    T R[9];
    ceres::QuaternionToRotation(q, R);

    Eigen::Matrix<T, 3, 3> rmat;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            rmat(i,j) = R[i * 3 + j];
        }
    }

    return rmat;
}

// JPL
template<typename T>
Eigen::Matrix<T,4,4> QuaternionMultMatLeft(const Eigen::Quaternion<T>& q)
{
    return (Eigen::Matrix<T,4,4>() << q.w(), -q.z(), q.y(), q.x(),
                                      q.z(), q.w(), -q.x(), q.y(),
                                      -q.y(), q.x(), q.w(), q.z(),
                                      -q.x(), -q.y(), -q.z(), q.w()).finished();
}

template<typename T>
Eigen::Matrix<T,4,4> QuaternionMultMatRight(const Eigen::Quaternion<T>& q)
{
    return (Eigen::Matrix<T,4,4>() << q.w(), q.z(), -q.y(), q.x(),
                                      -q.z(), q.w(), q.x(), q.y(),
                                      q.y(), -q.x(), q.w(), q.z(),
                                      -q.x(), -q.y(), -q.z(), q.w()).finished();
}

template<typename T>
void mat2RPY(const Eigen::Matrix<T, 3, 3>& m, T& roll, T& pitch, T& yaw)
{
    roll = atan2(m(2,1), m(2,2));
    pitch = atan2(-m(2,0), sqrt(m(2,1) * m(2,1) + m(2,2) * m(2,2)));
    yaw = atan2(m(1,0), m(0,0));
}

class CameraOdometerError2
{
public:
    CameraOdometerError2(Eigen::Vector3d r1, Eigen::Vector3d t1,
                        Eigen::Vector3d r2, Eigen::Vector3d t2)
        : m_rvec1(r1), m_tvec1(t1), m_rvec2(r2), m_tvec2(t2)
    {}

    template<typename T>
    bool operator() (const T* const q4x1, const T* const t3x1, T* residuals) const
    {
        Eigen::Quaternion<T> q(q4x1[0], q4x1[1], q4x1[2], q4x1[3]);
        Eigen::Matrix<T,3,1> t;
        t << t3x1[0], t3x1[1], T(0);

        Eigen::Matrix<T,3,1> r1 = m_rvec1.cast<T>();
        Eigen::Matrix<T,3,1> t1 = m_tvec1.cast<T>();
        Eigen::Matrix<T,3,1> r2 = m_rvec2.cast<T>();
        Eigen::Matrix<T,3,1> t2 = m_tvec2.cast<T>();

        Eigen::Quaternion<T> q1 = AngleAxisToQuaternion<T>(r1);
        Eigen::Quaternion<T> q2 = AngleAxisToQuaternion<T>(r2);

        Eigen::Matrix<T,3,3> R1 = AngleAxisToRotationMatrix<T>(r1);

        T q_coeffs[4] = {q.w(), q.x(), q.y(), q.z()};
        Eigen::Matrix<T,3,3> R = QuaternionToRotation<T>(q_coeffs);

        Eigen::Matrix<T,3,1> t_err = (R1 - Eigen::Matrix<T,3,3>::Identity()) * t
                                     - (R * t2) + t1;

        Eigen::Quaternion<T> q_err = q.conjugate() * q1 * q * q2.conjugate();

        T q_err_coeffs[4] = {q_err.w(), q_err.x(), q_err.y(), q_err.z()};
        Eigen::Matrix<T,3,3> R_err = QuaternionToRotation<T>(q_err_coeffs);

        T roll, pitch, yaw;
        mat2RPY(R_err, roll, pitch, yaw);

        residuals[0] = t_err(0);
        residuals[1] = t_err(1);
        residuals[2] = t_err(2);
        residuals[3] = roll;
        residuals[4] = pitch;
        residuals[5] = yaw;

        return true;
    }

private:
    Eigen::Vector3d m_rvec1, m_rvec2, m_tvec1, m_tvec2;
};

class HandEyeCalib {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  HandEyeCalib(const StampedPoseVectorPtr& poses_1, 
               const StampedPoseVectorPtr& poses_2);

  void processingPoses();
  bool estimate(Eigen::Matrix4d& H_odo_cam) const;

private:
  double calc_time_offset();
  double calcTimeDiffMean(const std::vector<double>& timestamp_seq);
  void resample_quaternions(const StampedPoseVectorPtr& poses,
                            const double& dt,
                            vector_quad& quatenoins_interp_seq,
                            std::vector<double>& sample_seq);
  void interpolate_poses_from_samples(
                    const StampedPoseVectorPtr& timestamp_pose_seq, 
                    const std::vector<double>& samples,
                    StampedPoseVectorPtr& new_timestamp_pose_seq);

  std::vector<double> linear_interp(const std::vector<double>& x, std::vector<double>& xp, const std::vector<double>& fp);
  void compute_angular_velocity_norms(const vector_quad& quatenoins_interp_seq,
          const std::vector<double>& sample_seq,
          const int smoothing_kernel_size,
          const double clipping_percentile,
          std::vector<double>& angular_velocity_norms,
          vector_vec3d& angular_vel_seq);
  Eigen::Vector3d angular_velocity_between_quaternions(const Eigen::Quaterniond& q0, 
                                                     const Eigen::Quaterniond& q1,
                                                     const double& dt);
  double calculate_time_offset_from_signals(const std::vector<double>& sample_seq1,
                                          const std::vector<double>& angular_velocity_norms1,
                                          const std::vector<double>& sample_seq2,
                                          const std::vector<double>& angular_velocity_norms2);                                                     
  std::vector<double> calculateCorrelation(const std::vector<double>& vec1, const std::vector<double>& vec2);
  vector_vec3d filter_and_smooth_angular_velocity(const vector_vec3d& angular_vel_seq,
                                                  const int low_pass_kernel_size,
                                                  const double clip_percentile);
  double calculatePercentile(const vector_vec3d& angular_vel_seq, double p);
  vector_vec3d calculateEigenCorrelation(const vector_vec3d & vec1, 
    const std::vector<double>& vec2);

  void compute_aligned_poses(const double& time_offset,
                        StampedPoseVectorPtr& aligned_pose_seq1,
                        StampedPoseVectorPtr& aligned_pose_seq2); 

  bool estimateRyx(Eigen::Matrix3d& R_yx_1, Eigen::Matrix3d& R_yx_2) const;

  void refineEstimate(Eigen::Matrix4d& H_odo_cam) const;  

  bool solveQuadraticEquation(double a, double b, double c, double& x1, double& x2) const;                    

  StampedPoseVectorPtr poses_1_;
  StampedPoseVectorPtr poses_2_;

  int smoothing_kernel_size1_ = 25;
  double clipping_percentile1_ = 99.5;
  int smoothing_kernel_size2_ = 25;
  double clipping_percentile2_ = 99.0;

  StampedPoseVectorPtr aligned_pose_seq1_;
  StampedPoseVectorPtr aligned_pose_seq2_;

  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> rvecs1_;
  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> tvecs1_;
  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> rvecs2_;
  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> tvecs2_;
  
};

#endif