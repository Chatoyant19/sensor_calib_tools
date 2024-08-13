#include "handeye.h"

HandEyeCalib::HandEyeCalib(const StampedPoseVectorPtr& poses_1,
                           const StampedPoseVectorPtr& poses_2) {
  poses_1_ = poses_1;
  poses_2_ = poses_2;

  aligned_pose_seq1_ = std::make_shared<StampedPoseVector>();
  aligned_pose_seq2_ = std::make_shared<StampedPoseVector>();
}

void HandEyeCalib::processingPoses() {
  double time_offset = calc_time_offset();
  compute_aligned_poses(time_offset, aligned_pose_seq1_, aligned_pose_seq2_);

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >
      T_seq1, T_seq2;
  for (int i = 0; i < aligned_pose_seq1_->size(); ++i) {
    Eigen::Matrix4d aligned_pose1 = Eigen::Matrix4d::Identity();
    aligned_pose1 = aligned_pose_seq1_->at(i).second;
    Eigen::Matrix4d aligned_pose2 = Eigen::Matrix4d::Identity();
    aligned_pose2 = aligned_pose_seq2_->at(i).second;
    T_seq1.emplace_back(aligned_pose1);
    T_seq2.emplace_back(aligned_pose2);
  }

  // step2: compute relative pose
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >
      relative_seq1, relative_seq2;
  for (int i = 1; i < T_seq1.size(); ++i) {
    Eigen::Matrix4d relativeOdometryPose = T_seq1[i].inverse() * T_seq1[i - 1];
    relative_seq1.emplace_back(relativeOdometryPose);
    Eigen::Matrix4d relativeCameraPose = T_seq2[i].inverse() * T_seq2[i - 1];
    relative_seq2.emplace_back(relativeCameraPose);
  }

  // std::vector<std::vector<Eigen::Vector3d,
  // Eigen::aligned_allocator<Eigen::Vector3d>>> rvecs1, tvecs1, rvecs2, tvecs2;
  // Eigen::Matrix4d H_odo_cam = Eigen::Matrix4d::Identity();
  // std::vector<double> scales;

  int motion_cnt = 0;
  motion_cnt = relative_seq1.size();
  rvecs1_.resize(1);
  tvecs1_.resize(1);
  rvecs2_.resize(1);
  tvecs2_.resize(1);
  // std::cout << "motionCount: " << motion_cnt << std::endl;
  // std::cout << "segmentCount: " << segment_cnt << std::endl;
  // std::cout << "length: " << length << std::endl;
  Eigen::Vector3d rvec1, tvec1, rvec2, tvec2;

  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < motion_cnt; ++j) {
      Eigen::Matrix3d R1 = relative_seq1[j].block(0, 0, 3, 3);
      Eigen::Matrix3d R2 = relative_seq2[j].block(0, 0, 3, 3);
      rvec1 = RotationToAngleAxis(R1);
      rvec2 = RotationToAngleAxis(R2);
      tvec1 = relative_seq1[j].block(0, 3, 3, 1);
      tvec2 = relative_seq2[j].block(0, 3, 3, 1);
      rvecs1_.at(i).push_back(rvec1);
      tvecs1_.at(i).push_back(tvec1);
      rvecs2_.at(i).push_back(rvec2);
      tvecs2_.at(i).push_back(tvec2);
    }
  }
}

double HandEyeCalib::calc_time_offset() {
  double time_offset;

  double mean_diff_1, mean_diff_2, diff;
  std::vector<double> timestamp_seq1, timestamp_seq2;
  for (size_t i = 0; i < poses_1_->size(); ++i) {
    timestamp_seq1.emplace_back(poses_1_->at(i).first);
  }
  for (size_t i = 0; i < poses_2_->size(); ++i) {
    timestamp_seq2.emplace_back(poses_2_->at(i).first);
  }

  mean_diff_1 = calcTimeDiffMean(timestamp_seq1);
  mean_diff_2 = calcTimeDiffMean(timestamp_seq2);
  diff = (mean_diff_1 >= mean_diff_2) ? mean_diff_1 : mean_diff_2;

  vector_quad quatenoins_interp_seq1, quatenoins_interp_seq2;
  std::vector<double> sample_seq1, sample_seq2;
  resample_quaternions(poses_1_, diff, quatenoins_interp_seq1, sample_seq1);
  resample_quaternions(poses_2_, diff, quatenoins_interp_seq2, sample_seq2);

  std::vector<double> angular_velocity_norms1, angular_velocity_norms2;
  vector_vec3d angular_vel_seq1, angular_vel_seq2;
  compute_angular_velocity_norms(quatenoins_interp_seq1, sample_seq1,
                                 smoothing_kernel_size1_, clipping_percentile1_,
                                 angular_velocity_norms1, angular_vel_seq1);
  compute_angular_velocity_norms(quatenoins_interp_seq2, sample_seq2,
                                 smoothing_kernel_size2_, clipping_percentile2_,
                                 angular_velocity_norms2, angular_vel_seq2);

  int length_before = sample_seq1.size();
  // sample_seq1_ is ok!!
  sample_seq1.pop_back();
  sample_seq2.pop_back();

  assert(sample_seq1.size() == length_before - 1);

  // angular_velocity_norms1,angular_velocity_norms2 is ok!!
  time_offset =
      calculate_time_offset_from_signals(sample_seq1, angular_velocity_norms1,
                                         sample_seq2, angular_velocity_norms2);
  return time_offset;
}

double HandEyeCalib::calcTimeDiffMean(
    const std::vector<double>& timestamp_seq) {
  double time_diff = 0.;
  for (int i = 1; i < timestamp_seq.size(); i++) {
    time_diff += timestamp_seq.at(i) - timestamp_seq.at(i - 1);
  }
  time_diff /= (timestamp_seq.size() - 1);
  return time_diff;
}

void HandEyeCalib::resample_quaternions(const StampedPoseVectorPtr& poses,
                                        const double& dt,
                                        vector_quad& quatenoins_interp_seq,
                                        std::vector<double>& sample_seq) {
  // const TimeVectorPtr& timestamp_seq,
  // const QuaterniondVectorPtr& quatenoins_seq,
  int timestamp_pose_size = poses->size();
  double t_start = poses->at(0).first;
  double t_end = poses->at(timestamp_pose_size - 1).first;
  double interval = t_end - t_start;
  int new_time_seq_size = interval / dt + 1;
  Eigen::VectorXd new_sample_seq =
      Eigen::VectorXd::LinSpaced(new_time_seq_size, t_start, t_end);
  for (int i = 0; i < new_time_seq_size; ++i) {
    sample_seq.emplace_back(new_sample_seq[i]);
  }

  StampedPoseVectorPtr new_timestamp_pose_seq =
      std::make_shared<StampedPoseVector>();
  interpolate_poses_from_samples(poses, sample_seq, new_timestamp_pose_seq);
  // get Quaternions frome new timestamp_pose_seq
  for (int i = 0; i < new_timestamp_pose_seq->size(); ++i) {
    Eigen::Quaterniond q = Eigen::Quaterniond(
        new_timestamp_pose_seq->at(i).second.block<3, 3>(0, 0));
    quatenoins_interp_seq.emplace_back(q);
  }
}

void HandEyeCalib::interpolate_poses_from_samples(
    const StampedPoseVectorPtr& timestamp_pose_seq,
    const std::vector<double>& samples,
    StampedPoseVectorPtr& new_timestamp_pose_seq) {
  // get timestamp_seq
  std::vector<double> timestamp_seq;
  std::vector<double> tx_seq, ty_seq, tz_seq;
  for (int i = 0; i < timestamp_pose_seq->size(); ++i) {
    timestamp_seq.emplace_back(timestamp_pose_seq->at(i).first);
    tx_seq.emplace_back(timestamp_pose_seq->at(i).second.block<3, 1>(0, 3).x());
    ty_seq.emplace_back(timestamp_pose_seq->at(i).second.block<3, 1>(0, 3).y());
    tz_seq.emplace_back(timestamp_pose_seq->at(i).second.block<3, 1>(0, 3).z());
  }
  // get new tx,ty,tz
  std::vector<double> new_tx_seq =
      linear_interp(samples, timestamp_seq, tx_seq);
  std::vector<double> new_ty_seq =
      linear_interp(samples, timestamp_seq, ty_seq);
  std::vector<double> new_tz_seq =
      linear_interp(samples, timestamp_seq, tz_seq);

  std::vector<double>::iterator t_upper, t_lower;
  for (int i = 0; i < samples.size(); i++) {
    double t = samples[i];
    assert(t <= samples[samples.size() - 1]);
    assert(t >= samples[0]);

    t_upper = std::lower_bound(timestamp_seq.begin(), timestamp_seq.end(), t);
    int right_idx = std::distance(timestamp_seq.begin(), t_upper);
    t_lower = t_upper - 1;
    int left_idx = right_idx - 1;
    // if (t_lower < timestamp_seq.begin()) t_lower = timestamp_seq.begin();
    // if (t_upper > timestamp_seq.end()) t_upper = timestamp_seq.end();

    double atol = 1e-16;
    double rtol = 1e-5;
    double tmp1 = abs(samples[i] - timestamp_seq[right_idx]);
    double tmp2 = atol + rtol * abs(timestamp_seq[right_idx]);
    Eigen::Vector3d p =
        Eigen::Vector3d(new_tx_seq[i], new_ty_seq[i], new_tz_seq[i]);
    if (tmp1 <= tmp2) {
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.block<3, 3>(0, 0) =
          timestamp_pose_seq->at(right_idx).second.block<3, 3>(0, 0);
      T.block<3, 1>(0, 3) = p;
      new_timestamp_pose_seq->emplace_back(StampedPose(t, T));
    } else {
      assert(right_idx < timestamp_seq.size());
      assert(left_idx >= 0);
      double times_scaled =
          (samples[i] - timestamp_seq[left_idx]) /
          (timestamp_seq[right_idx] - timestamp_seq[left_idx]);
      assert(times_scaled >= 0.);
      assert(times_scaled <= 1.0);
      double dot_product = 0.;
      double theta_prime = 0., theta = 0.;
      Eigen::Quaterniond q_1 = Eigen::Quaterniond(
          timestamp_pose_seq->at(left_idx).second.block<3, 3>(0, 0));
      Eigen::Quaterniond q_2 = Eigen::Quaterniond(
          timestamp_pose_seq->at(right_idx).second.block<3, 3>(0, 0));
      Eigen::Quaterniond q;

      if (times_scaled == 0.)
        q = q_1;
      else if (times_scaled = 1.0)
        q = q_2;
      else {
        dot_product = q_1.dot(q_2);
        if (dot_product < 0.) {
          dot_product = dot_product * (-1);
          // q_1 = q_1*(-1);
          q_1 = Eigen::Quaterniond(q_1.w() * (-1), q_1.x() * (-1),
                                   q_1.y() * (-1), q_1.z() * (-1));
          // wd to check
          // std::cout << "q_1: " << q_1.coeffs() << std::endl;
        }
        if (dot_product < -1.0)
          dot_product = 1.0;
        else if (dot_product > 1.0)
          dot_product = 1.0;
        theta_prime = acos(dot_product);
        theta = theta_prime * times_scaled;
        Eigen::Quaterniond q_2_tmp =
            Eigen::Quaterniond(q_2.w() * dot_product, q_2.x() * dot_product,
                               q_2.y() * dot_product, q_2.z() * dot_product);
        Eigen::Quaterniond q_3 =
            Eigen::Quaterniond(q_1.w() - q_2_tmp.w(), q_1.x() - q_2_tmp.x(),
                               q_1.y() - q_2_tmp.y(), q_1.z() - q_2_tmp.z());

        q_3.normalize();
        // wd to check
        // q = q_1*cos(theta) + q_3 * sin(theta);
        Eigen::Quaterniond q_1_tmp =
            Eigen::Quaterniond(q_1.w() * cos(theta), q_1.x() * cos(theta),
                               q_1.y() * cos(theta), q_1.z() * cos(theta));
        Eigen::Quaterniond q_3_tmp =
            Eigen::Quaterniond(q_3.w() * sin(theta), q_3.x() * sin(theta),
                               q_3.y() * sin(theta), q_3.z() * sin(theta));
        q = Eigen::Quaterniond(
            q_1_tmp.w() + q_3_tmp.w(), q_1_tmp.x() + q_3_tmp.x(),
            q_1_tmp.y() + q_3_tmp.y(), q_1_tmp.z() + q_3_tmp.z());
      }
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.block<3, 3>(0, 0) = q.toRotationMatrix();
      T.block<3, 1>(0, 3) = p;
      new_timestamp_pose_seq->emplace_back(StampedPose(t, T));
    }
  }
}

std::vector<double> HandEyeCalib::linear_interp(const std::vector<double>& x,
                                                std::vector<double>& xp,
                                                const std::vector<double>& fp) {
  std::vector<double> res(x.size(), 0);
  std::vector<double>::iterator right_it;
  int left_idx = 0;
  int right_idx = 0;
  double x1 = 0, x2 = 0;
  double y1 = 0, y2 = 0;
  double a = 0, b = 0;
  for (int i = 0; i < x.size(); ++i) {
    if (x[i] <= xp[0])
      res[i] = fp[0];
    else if (x[i] >= xp[xp.size() - 1])
      res[i] = fp[fp.size() - 1];
    else {
      right_it = std::lower_bound(xp.begin(), xp.end(), x[i]);
      right_idx = std::distance(xp.begin(), right_it);
      if (xp[right_idx] == x[i])
        res[i] = fp[right_idx];
      else {
        left_idx = right_idx - 1;
        x1 = xp[left_idx];
        x2 = xp[right_idx];
        y1 = fp[left_idx];
        y2 = fp[right_idx];
        a = (y1 - y2) / (x1 - x2);
        b = y1 - x1 * a;
        res[i] = a * x[i] + b;
      }
    }
  }
  return res;
}

void HandEyeCalib::compute_angular_velocity_norms(
    const vector_quad& quatenoins_interp_seq,
    const std::vector<double>& sample_seq, const int smoothing_kernel_size,
    const double clipping_percentile,
    std::vector<double>& angular_velocity_norms,
    vector_vec3d& angular_vel_seq) {
  int angular_velocity_size = quatenoins_interp_seq.size();
  for (int i = 1; i < angular_velocity_size; ++i) {
    double dt = sample_seq.at(i) - sample_seq.at(i - 1);
    assert(dt > 0);
    Eigen::Vector3d angular_vel = angular_velocity_between_quaternions(
        quatenoins_interp_seq[i - 1], quatenoins_interp_seq[i], dt);
    angular_vel_seq.emplace_back(angular_vel);
  }
  vector_vec3d angular_velocity_filtered_seq =
      filter_and_smooth_angular_velocity(angular_vel_seq, smoothing_kernel_size,
                                         clipping_percentile);
  for (int i = 0; i < angular_velocity_size - 1; ++i) {
    angular_velocity_norms.emplace_back(
        angular_velocity_filtered_seq[i].norm());
  }
  assert(angular_velocity_norms.size() == quatenoins_interp_seq.size() - 1);
}

Eigen::Vector3d HandEyeCalib::angular_velocity_between_quaternions(
    const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1,
    const double& dt) {
  Eigen::Quaterniond q0_q1_inv = q0 * q1.inverse();
  // std::cout << q0_q1_inv.x() << " " << q0_q1_inv.y() << " " << q0_q1_inv.z()
  // << " " << q0_q1_inv.w() << std::endl;
  if (q0_q1_inv.w() < 0.) {
    // q0_q1_inv = q0_q1_inv * (-1.0);
    q0_q1_inv =
        Eigen::Quaterniond(q0_q1_inv.w() * (-1.0), q0_q1_inv.x() * (-1.0),
                           q0_q1_inv.y() * (-1.0), q0_q1_inv.z() * (-1.0));
  }
  Eigen::AngleAxisd angle_axis(q0_q1_inv);
  return 0.5 / dt * angle_axis.angle() * angle_axis.axis();
}

vector_vec3d HandEyeCalib::filter_and_smooth_angular_velocity(
    const vector_vec3d& angular_vel_seq, const int low_pass_kernel_size,
    const double clip_percentile) {
  double max_value = calculatePercentile(angular_vel_seq, clip_percentile);
  // std::cout << "Clipping angular velocity norms to : " << max_value << "rad/s
  // ..." << std::endl;
  vector_vec3d angular_velocity_clipped(angular_vel_seq.size(),
                                        Eigen::Vector3d(0., 0., 0.));
  for (int i = 0; i < angular_vel_seq.size(); ++i) {
    angular_velocity_clipped.at(i) = angular_vel_seq[i];
  }
  for (int i = 0; i < angular_velocity_clipped.size(); ++i) {
    if (angular_velocity_clipped[i].x() < (max_value * (-1.0)))
      angular_velocity_clipped[i].x() = max_value * (-1.0);
    if (angular_velocity_clipped[i].x() > max_value)
      angular_velocity_clipped[i].x() = max_value;
    if (angular_velocity_clipped[i].y() < (max_value * (-1.0)))
      angular_velocity_clipped[i].y() = max_value * (-1.0);
    if (angular_velocity_clipped[i].y() > max_value)
      angular_velocity_clipped[i].y() = max_value;
    if (angular_velocity_clipped[i].z() < (max_value * (-1.0)))
      angular_velocity_clipped[i].z() = max_value * (-1.0);
    if (angular_velocity_clipped[i].z() > max_value)
      angular_velocity_clipped[i].z() = max_value;
  }
  // std::cout << "Done clipping angular velocity norms..." << std::endl;
  double low_pass_kernel_value = 1.0 / (double)low_pass_kernel_size;
  std::vector<double> low_pass_kernel(low_pass_kernel_size,
                                      low_pass_kernel_value);
  // std::cout << "low_pass_kernel: " << low_pass_kernel.size() << std::endl;
  // std::cout << "Smoothing with kernel size " << low_pass_kernel_size <<
  // "samples..." << std::endl;
  vector_vec3d angular_velocity_smoothed =
      calculateEigenCorrelation(angular_velocity_clipped, low_pass_kernel);
  // std::cout << "Done smoothing angular velocity norms..." << std::endl;
  return angular_velocity_smoothed;
}

double HandEyeCalib::calculatePercentile(const vector_vec3d& angular_vel_seq,
                                         double p) {
  std::vector<double> vec;
  for (int i = 0; i < angular_vel_seq.size(); ++i) {
    vec.emplace_back(angular_vel_seq[i].x());
    vec.emplace_back(angular_vel_seq[i].y());
    vec.emplace_back(angular_vel_seq[i].z());
  }
  std::sort(vec.begin(), vec.end());
  int n = vec.size();
  p = p * (0.01);
  double tmp = (n - 1) * p;
  int tmp1 = floor(tmp);
  double tmp2 = tmp - tmp1;
  double res = (1 - tmp2) * vec[tmp1] + tmp2 * vec[tmp1 + 1];
  return res;
}

vector_vec3d HandEyeCalib::calculateEigenCorrelation(
    const vector_vec3d& vec1, const std::vector<double>& vec2) {
  int M = vec1.size();
  int N = vec2.size();
  vector_vec3d res_full(M + N - 1, Eigen::Vector3d(0, 0, 0));
  vector_vec3d res_same(M, Eigen::Vector3d(0, 0, 0));
  int start_1 = 0, end_1 = 0;
  int start_2 = N, end_2 = N;
  for (int i = 0; i < M + N - 1; ++i) {
    end_1 = (end_1 == M ? M : ++end_1);
    start_2 = (start_2 == 0 ? 0 : --start_2);
    if (i > N - 1) start_1++;
    if (i > M - 1) end_2--;
    Eigen::Vector3d result(0, 0, 0);
    for (int j = start_1, k = start_2; j < end_1, k < end_2; ++j, ++k) {
      result.x() += (vec1[j].x() * vec2[k]);
      result.y() += (vec1[j].y() * vec2[k]);
      result.z() += (vec1[j].z() * vec2[k]);
    }
    res_full[i] = result;
  }
  int begin = N / 2 - 1;
  for (int i = begin, j = 0; i < begin + M, j < M; ++i, ++j) {
    res_same[j] = res_full[i];
  }
  return res_same;
}

double HandEyeCalib::calculate_time_offset_from_signals(
    const std::vector<double>& sample_seq1,
    const std::vector<double>& angular_velocity_norms1,
    const std::vector<double>& sample_seq2,
    const std::vector<double>& angular_velocity_norms2) {
  std::vector<double> convoluted_signals =
      calculateCorrelation(angular_velocity_norms2, angular_velocity_norms1);
  double sample_diff1 = 0.;
  for (int i = 1; i < sample_seq1.size(); ++i) {
    sample_diff1 += sample_seq1.at(i) - sample_seq1.at(i - 1);
  }
  // sample_diff1 is ok!!
  sample_diff1 /= ((double)sample_seq1.size() - 1.);
  std::vector<int> offset_indices;
  int ang_vel_size = angular_velocity_norms1.size();
  int index_start = ang_vel_size * (-1) + 1;
  for (int i = index_start; i < ang_vel_size; ++i) {
    offset_indices.push_back(i);
  }
  std::vector<double>::iterator iter =
      std::max_element(convoluted_signals.begin(), convoluted_signals.end());
  int max_index = std::distance(convoluted_signals.begin(), iter);
  int offset_index = offset_indices[max_index];
  double time_offset = sample_diff1 * (double)offset_index + sample_seq2.at(0) -
                       sample_seq1.at(0);
  return time_offset;
}

std::vector<double> HandEyeCalib::calculateCorrelation(
    const std::vector<double>& vec1, const std::vector<double>& vec2) {
  int M = vec1.size();
  int N = vec2.size();
  std::vector<double> res_full(M + N - 1, 0.);
  int start_1 = 0, end_1 = 0;
  int start_2 = N, end_2 = N;
  for (int i = 0; i < M + N - 1; ++i) {
    end_1 = (end_1 == M ? M : ++end_1);
    start_2 = (start_2 == 0 ? 0 : --start_2);
    if (i > N - 1) start_1++;
    if (i > M - 1) end_2--;
    double result = 0.;
    for (int j = start_1, k = start_2; j < end_1, k < end_2; ++j, ++k) {
      result += (vec1[j] * vec2[k]);
    }
    res_full[i] = result;
  }
  return res_full;
}

void HandEyeCalib::compute_aligned_poses(
    const double& time_offset, StampedPoseVectorPtr& aligned_pose_seq1,
    StampedPoseVectorPtr& aligned_pose_seq2) {
  StampedPoseVectorPtr timestamp_pose_shifted_seq1 = poses_1_;
  double time_tmp = 0.;
  for (int i = 0; i < timestamp_pose_shifted_seq1->size(); ++i) {
    timestamp_pose_shifted_seq1->at(i).first += time_offset;
  }
  double start_time = std::max(timestamp_pose_shifted_seq1->begin()->first,
                               poses_2_->begin()->first);
  double end_time = std::min(timestamp_pose_shifted_seq1->end()->first,
                             poses_2_->end()->first);
  double interval = end_time - start_time;
  double mean_diff_1 = 0.;
  double mean_diff_2 = 0.;
  double diff = 0.;
  // get timestamp_shifted_seq1
  std::vector<double> timestamp_shifted_seq1, timestamp_seq2;
  for (int i = 0; i < timestamp_pose_shifted_seq1->size(); ++i) {
    timestamp_shifted_seq1.emplace_back(
        timestamp_pose_shifted_seq1->at(i).first);
  }
  // get timestamp_seq2
  for (int i = 0; i < poses_2_->size(); ++i) {
    timestamp_seq2.emplace_back(poses_2_->at(i).first);
  }

  mean_diff_1 = calcTimeDiffMean(timestamp_shifted_seq1);
  mean_diff_2 = calcTimeDiffMean(timestamp_seq2);

  std::vector<double> timestamps_low;
  std::vector<double> timestamps_high;
  if (mean_diff_1 >= mean_diff_2) {
    diff = mean_diff_1;
    for (int i = 0; i < timestamp_shifted_seq1.size(); ++i) {
      timestamps_low.emplace_back(timestamp_shifted_seq1.at(i));
    }
    for (int i = 0; i < timestamp_seq2.size(); ++i) {
      timestamps_high.emplace_back(timestamp_seq2.at(i));
    }
  } else {
    diff = mean_diff_2;
    for (int i = 0; i < timestamp_seq2.size(); ++i) {
      timestamps_low.emplace_back(timestamp_seq2.at(i));
    }
    for (int i = 0; i < timestamp_shifted_seq1.size(); ++i) {
      timestamps_high.emplace_back(timestamp_shifted_seq1.at(i));
    }
  }

  std::vector<double> samples;
  double max_time_stamp_difference = 0.1;
  int idx = 0;
  int left_idx = 0;
  int right_idx = 0;
  std::vector<double>::iterator right_it;
  for (int i = 0; i < timestamps_low.size(); ++i) {
    if (timestamps_low.at(i) < start_time) continue;
    right_it = std::lower_bound(timestamps_high.begin(), timestamps_high.end(),
                                timestamps_low.at(i));
    idx = std::distance(timestamps_high.begin(), right_it);
    if (idx >= timestamps_high.size() - 1) {
      // std::cout << "Omitting timestamps at the end of the high frequency
      // poses." << std::endl;
      break;
    }
    if (timestamps_low.at(i) == timestamps_high.at(idx)) {
      samples.emplace_back(timestamps_low.at(i));
      continue;
    }
    left_idx = idx - 1;
    if (timestamps_low.at(i) < timestamps_high.at(left_idx)) continue;
    right_idx = idx;
    assert(right_idx < timestamps_high.size());
    if ((timestamps_low.at(i) - timestamps_high.at(left_idx)) <
            max_time_stamp_difference &&
        (timestamps_high.at(right_idx) - timestamps_low.at(i)) <=
            max_time_stamp_difference) {
      samples.emplace_back(timestamps_low.at(i));
    }
  }

  interpolate_poses_from_samples(timestamp_pose_shifted_seq1, samples,
                                 aligned_pose_seq1);
  interpolate_poses_from_samples(poses_2_, samples, aligned_pose_seq2);
}

bool HandEyeCalib::estimate(Eigen::Matrix4d& H_odo_cam) const {
  // Estimate R_yx first
  Eigen::Matrix3d R_yx_1, R_yx_2;
  if (!estimateRyx(R_yx_1, R_yx_2)) {
    std::cerr << "cannot estimate Ryx" << std::endl;
    return false;
  }

  Eigen::Matrix3d R_yxs[2];
  R_yxs[0] = R_yx_1;
  R_yxs[1] = R_yx_2;

  Eigen::Matrix4d extrinsics[2];
  double total_errs[2];

  for (size_t i = 0; i < 2; ++i) {
    Eigen::Matrix3d R_yx;
    R_yx = R_yxs[i];
    int segmentCount = rvecs1_.size();
    int motionCount = 0;
    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
      motionCount += rvecs1_.at(segmentId).size();
    }

    Eigen::MatrixXd G =
        Eigen::MatrixXd::Zero(motionCount * 2, 2 + segmentCount * 2);
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(motionCount * 2, 1);
    int mark = 0;
    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
      for (size_t motionId = 0; motionId < rvecs1_.at(segmentId).size();
           ++motionId) {
        const Eigen::Vector3d& rvec1 = rvecs1_.at(segmentId).at(motionId);
        const Eigen::Vector3d& tvec1 = tvecs1_.at(segmentId).at(motionId);
        const Eigen::Vector3d& rvec2 = rvecs2_.at(segmentId).at(motionId);
        const Eigen::Vector3d& tvec2 = tvecs2_.at(segmentId).at(motionId);

        // Remove zero rotation.
        if (rvec1.norm() < 1e-10 || rvec2.norm() < 1e-10) {
          ++mark;
          continue;
        }

        Eigen::Quaterniond q1;
        q1 = Eigen::AngleAxisd(rvec1.norm(), rvec1.normalized());

        Eigen::Matrix2d J;
        J = q1.toRotationMatrix().block<2, 2>(0, 0) -
            Eigen::Matrix2d::Identity();

        // project tvec2 to plane with normal defined by 3rd row of R_yx
        Eigen::Vector3d n;
        n = R_yx.row(2);

        Eigen::Vector3d pi = R_yx * (tvec2 - tvec2.dot(n) * n);

        Eigen::Matrix2d K;
        K << pi(0), -pi(1), pi(1), pi(0);

        G.block<2, 2>(mark * 2, 0) = J;
        G.block<2, 2>(mark * 2, 2 + segmentId * 2) = K;

        w.block<2, 1>(mark * 2, 0) = tvec1.block<2, 1>(0, 0);

        ++mark;
      }
    }

    Eigen::MatrixXd m(2 + segmentCount * 2, 1);
    m = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(w);

    Eigen::Vector2d t(-m(0), -m(1));

    std::vector<double> alpha_hypos;
    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
      double alpha = atan2(m(2 + segmentId * 2 + 1), m(2 + segmentId * 2));

      alpha_hypos.push_back(alpha);
    }

    double errorMin = std::numeric_limits<double>::max();
    double alpha_best = 0.0;

    for (size_t i = 0; i < alpha_hypos.size(); ++i) {
      double error = 0.0;
      double alpha = alpha_hypos.at(i);

      for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
        for (size_t motionId = 0; motionId < rvecs1_.at(segmentId).size();
             ++motionId) {
          const Eigen::Vector3d& rvec1 = rvecs1_.at(segmentId).at(motionId);
          const Eigen::Vector3d& tvec1 = tvecs1_.at(segmentId).at(motionId);
          const Eigen::Vector3d& rvec2 = rvecs2_.at(segmentId).at(motionId);
          const Eigen::Vector3d& tvec2 = tvecs2_.at(segmentId).at(motionId);

          Eigen::Quaterniond q1;
          q1 = Eigen::AngleAxisd(rvec1.norm(), rvec1.normalized());

          Eigen::Matrix3d N;
          N = q1.toRotationMatrix() - Eigen::Matrix3d::Identity();

          Eigen::Matrix3d R =
              Eigen::AngleAxisd(alpha, Eigen::Vector3d::UnitZ()) * R_yx;

          // project tvec2 to plane with normal defined by 3rd row of R
          Eigen::Vector3d n;
          n = R.row(2);

          Eigen::Vector3d pc = tvec2 - tvec2.dot(n) * n;
          //        Eigen::Vector3d pc = tvec2;

          Eigen::Vector3d A = R * pc;
          // std::cout << "judge R_yx ok? " << A.z() << std::endl;
          Eigen::Vector3d b =
              N * (Eigen::Vector3d() << t, 0.0).finished() + tvec1;

          error += (A - b).norm();
        }
      }

      if (error < errorMin) {
        errorMin = error;
        alpha_best = alpha;
      }
    }

    H_odo_cam.setIdentity();
    H_odo_cam.block<3, 3>(0, 0) =
        Eigen::AngleAxisd(alpha_best, Eigen::Vector3d::UnitZ()) * R_yx;
    H_odo_cam.block<2, 1>(0, 3) = t;

    Eigen::Quaterniond qua = Eigen::Quaterniond(H_odo_cam.block<3, 3>(0, 0));
    // std::cout << "w: " << qua.w() << std::endl
    //           << "x: " << qua.x() << std::endl
    //           << "y: " << qua.y() << std::endl
    //           << "z: " << qua.z() << std::endl;

    refineEstimate(H_odo_cam);

    // compute error
    double total_err = 0.0;
    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
      for (size_t motionId = 0; motionId < rvecs1_.at(segmentId).size();
           ++motionId) {
        const Eigen::Vector3d& rvec1 = rvecs1_.at(segmentId).at(motionId);
        Eigen::Vector3d tvec1 = tvecs1_.at(segmentId).at(motionId);
        const Eigen::Vector3d& rvec2 = rvecs2_.at(segmentId).at(motionId);
        Eigen::Vector3d tvec2 = tvecs2_.at(segmentId).at(motionId);

        // Remove zero rotation.
        if (rvec1.norm() < 1e-6 || rvec2.norm() < 1e-6) {
          ++mark;
          continue;
        }

        Eigen::Quaterniond q1;
        q1 = Eigen::AngleAxisd(rvec1.norm(), rvec1.normalized());

        Eigen::Quaterniond q2;
        q2 = Eigen::AngleAxisd(rvec2.norm(), rvec2.normalized());

        Eigen::Matrix3d rot = H_odo_cam.block<3, 3>(0, 0);
        Eigen::Quaterniond q = Eigen::Quaterniond(rot);
        total_err += (q.conjugate() * q1 * q * q2.conjugate()).norm();

        Eigen::Matrix3d R1 = q1.toRotationMatrix();
        Eigen::Matrix3d R2 = q2.toRotationMatrix();
        Eigen::Vector3d t = H_odo_cam.block<3, 1>(0, 3);
        Eigen::Vector3d t1 = tvec1;
        Eigen::Vector3d t2 = tvec2;
        total_err +=
            ((R1 - Eigen::Matrix3d::Identity()) * t - (rot * t2) + t1).norm();
      }

      total_err /= rvecs1_.at(segmentId).size();
      // std::cout << "total_err: " << total_err << std::endl;
    }

    extrinsics[i] = H_odo_cam;
    total_errs[i] = total_err;
  }

  H_odo_cam = total_errs[0] <= total_errs[1] ? extrinsics[0] : extrinsics[1];

  return true;
}

bool HandEyeCalib::estimateRyx(Eigen::Matrix3d& R_yx_1,
                               Eigen::Matrix3d& R_yx_2) const {
  size_t motionCount = 0;
  for (size_t i = 0; i < 1; ++i) {
    motionCount += rvecs1_.at(i).size();
  }

  Eigen::MatrixXd M(motionCount * 4, 4);
  M.setZero();

  size_t mark = 0;
  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < rvecs1_.at(i).size(); ++j) {
      const Eigen::Vector3d& rvec1 = rvecs1_.at(i).at(j);
      // const Eigen::Vector3d& tvec1 = tvecs1.at(i).at(j);
      const Eigen::Vector3d& rvec2 = rvecs2_.at(i).at(j);
      // const Eigen::Vector3d& tvec2 = tvecs2.at(i).at(j);

      // Remove zero rotation.
      if (rvec1.norm() < 1e-6 || rvec2.norm() < 1e-6) {
        continue;
      }

      Eigen::Quaterniond q1;
      q1 = Eigen::AngleAxisd(rvec1.norm(), rvec1.normalized());
      // q1 = AngleAxisToQuaternion(rvec1);

      Eigen::Quaterniond q2;
      q2 = Eigen::AngleAxisd(rvec2.norm(), rvec2.normalized());
      // q2 = AngleAxisToQuaternion(rvec2);

      M.block<4, 4>(mark * 4, 0) =
          QuaternionMultMatLeft(q1) - QuaternionMultMatRight(q2);
      ++mark;
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Vector4d t1 = svd.matrixV().block<4, 1>(0, 2);
  Eigen::Vector4d t2 = svd.matrixV().block<4, 1>(0, 3);

  // solve constraint for q_yz: xy = -zw
  double s[2];

  if (!solveQuadraticEquation(
          t1(0) * t1(1) + t1(2) * t1(3),
          t1(0) * t2(1) + t1(1) * t2(0) + t1(2) * t2(3) + t1(3) * t2(2),
          t2(0) * t2(1) + t2(2) * t2(3), s[0], s[1])) {
    std::cout << "# ERROR: Quadratic equation cannot be solved due to negative "
                 "determinant."
              << std::endl;
    return false;
  }

  Eigen::Matrix3d R_yxs[2];
  double yaw[2];

  for (int i = 0; i < 2; ++i) {
    double t = s[i] * s[i] * t1.dot(t1) + 2 * s[i] * t1.dot(t2) + t2.dot(t2);

    // solve constraint ||q_yx|| = 1
    double b = sqrt(1.0 / t);
    double a = s[i] * b;

    Eigen::Quaterniond q_yx;
    q_yx.coeffs() = a * t1 + b * t2;
    R_yxs[i] = q_yx.toRotationMatrix();

    double r, p;
    mat2RPY(R_yxs[i], r, p, yaw[i]);
  }

  R_yx_1 = R_yxs[0];
  R_yx_2 = R_yxs[1];
  // if (fabs(yaw[0]) < fabs(yaw[1]))
  // {
  //     R_yx = R_yxs[0];
  // }
  // else
  // {
  //     R_yx = R_yxs[1];
  // }
  return true;
}

void HandEyeCalib::refineEstimate(Eigen::Matrix4d& H_odo_cam) const {
  // todo
  // H_odo_cam = Eigen::Matrix4d::Identity();

  Eigen::Quaterniond q(H_odo_cam.block<3, 3>(0, 0));
  double q_coeffs[4] = {q.w(), q.x(), q.y(), q.z()};
  double t_coeffs[3] = {H_odo_cam(0, 3), H_odo_cam(1, 3), H_odo_cam(2, 3)};

  ceres::Problem problem;

  for (size_t i = 0; i < rvecs1_.size(); ++i) {
    for (size_t j = 0; j < rvecs1_.at(i).size(); ++j) {
      // ceres::CostFunction* costFunction =
      //     // t is only flexible on x and y.
      //     new ceres::AutoDiffCostFunction<CameraOdometerError1, 6, 4, 2, 1>(
      //         new CameraOdometerError1(rvecs1.at(i).at(j),
      //         tvecs1.at(i).at(j), rvecs2.at(i).at(j), tvecs2.at(i).at(j)));

      ceres::CostFunction* costFunction =
          // t is only flexible on x and y.
          new ceres::AutoDiffCostFunction<CameraOdometerError2, 6, 4, 3>(
              new CameraOdometerError2(rvecs1_.at(i).at(j), tvecs1_.at(i).at(j),
                                       rvecs2_.at(i).at(j),
                                       tvecs2_.at(i).at(j)));
      problem.AddResidualBlock(costFunction, NULL, q_coeffs, t_coeffs);
    }
  }

  ceres::LocalParameterization* quaternionParameterization =
      new ceres::QuaternionParameterization;

  problem.SetParameterization(q_coeffs, quaternionParameterization);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  //    options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-20;
  options.function_tolerance = 1e-20;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  q = Eigen::Quaterniond(q_coeffs[0], q_coeffs[1], q_coeffs[2], q_coeffs[3]);
  H_odo_cam.block<3, 1>(0, 3) << t_coeffs[0], t_coeffs[1], t_coeffs[2];
  H_odo_cam.block<3, 3>(0, 0) = q.toRotationMatrix();
}

bool HandEyeCalib::solveQuadraticEquation(double a, double b, double c,
                                          double& x1, double& x2) const {
  if (fabs(a) < 1e-12) {
    x1 = x2 = -c / b;
    return true;
  }
  double delta2 = b * b - 4.0 * a * c;

  if (delta2 < 0.0) {
    return false;
  }

  double delta = sqrt(delta2);

  x1 = (-b + delta) / (2.0 * a);
  x2 = (-b - delta) / (2.0 * a);

  return true;
}