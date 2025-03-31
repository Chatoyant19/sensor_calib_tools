#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstdlib>  // `int64_t`
#include <map>

#include "linear_interpolater_trait.hpp"

using Time = double;

template <typename T>
class LinearInterpolater {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // todo: 将第一个时间参数也定义为模板参数   
  using TimeValueMap = std::map<Time, T, std::less<Time>, 
    Eigen::aligned_allocator<std::pair<const Time, T>>>;

  LinearInterpolater(Time max_age, Time max_predict)
    : max_age_(max_age), max_predict_(max_predict) {}
  size_t insert(Time t, const T& val);
  bool evaluate(Time t, T* val, double* ratio = nullptr, double* delta = nullptr) const;
  void clear();
  bool isEmpty() const { return time_value_.empty(); }
  size_t size() const { return time_value_.size(); }

 private:
  Time getStartTime() const;
  Time getEndTime() const;
  void removeValuesBefore(Time t);
  Time max_age_;
  Time max_predict_;
  TimeValueMap time_value_;
};

template <typename T>
Time LinearInterpolater<T>::getStartTime() const {
  return time_value_.cbegin()->first;
}

template <typename T>
Time LinearInterpolater<T>::getEndTime() const {
  return time_value_.crbegin()->first;
}

template <typename T>
void LinearInterpolater<T>::removeValuesBefore(Time t) {
  auto it = time_value_.begin();
  while (it != time_value_.end() && it->first < t) {
    it = time_value_.erase(it);
  }
}

template <typename T>
size_t LinearInterpolater<T>::insert(Time t, const T& val) {
  // std::cout << std::fixed;
  // std::cout << std::setprecision(18) << t << std::endl;
  time_value_[t] = val;
  removeValuesBefore(getEndTime() - max_age_);
 
  return time_value_.size();
}

template <typename T>
bool LinearInterpolater<T>::evaluate(Time t, T* val, double* ratio /*=nullptr*/,
                                     double* delta /*=nullptr*/) const {
  if (time_value_.empty()) {
    // AR_LOG(WARN) << "time_value_ empty";
    std::cout << "time_value_ empty" << std::endl;
    return false;
  }
  if (t < getStartTime() || t > getEndTime() + max_predict_) {
    // AR_LOG(WARN) << "time not in range: " << t << ", " << getStartTime() << ", "
                //  << getEndTime() + max_predict_;
    std::cout << "time not in range: " << t << ", " << getStartTime() << ", " << getEndTime() << std::endl;
    return false;
  }
  auto it = time_value_.find(t);
  if (it != time_value_.end()) {
    *val = it->second;
    return true;
  }
  if (time_value_.size() <= 1) {
    // AR_LOG(INFO) << "small time_value_.size: " << time_value_.size();
    std::cout << "small time_value_.size: " << time_value_.size() << std::endl;
    return false;
  }
  auto upper = time_value_.upper_bound(t);
  auto lower = std::prev(upper);
  if (upper == time_value_.end()) {
    upper = lower;
    if (lower != time_value_.begin()) {
      lower--;
    }
  }
  if (upper == lower) {
    *val = lower->second;
    if (ratio != nullptr) {
      *ratio = 0;
    }
    if (delta != nullptr) {
      *delta = 0;
    }
  } else {
    //  wd add it: last and next time interval disthold is 2s
    if(double(t - lower->first) >  1.0
        || double(upper->first - t) >  1.0) {
      std::cout << "time " << t << " in range " << lower->first << " and " << upper->first
               << "but last and next time interval is too large" << std::endl;
      return false;
    }
    const double ratio_inner = double(t - lower->first) / double(upper->first - lower->first);
    const T& lower_val = lower->second;
    const T& upper_val = upper->second;
    *val = LinearInterpolaterTrait<T>::plus(
        lower_val, LinearInterpolaterTrait<T>::between(lower_val, upper_val) * ratio_inner);
    if (ratio != nullptr) {
      *ratio = ratio_inner;
    }
    if (delta != nullptr) {
      *delta = double(upper->first - lower->first);
    }
  }
  return true;
}   