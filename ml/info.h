// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_INFO_H_
#define ML_INFO_H_
#include "base/base_head.h"
namespace ml {
inline void Normal(const VReal &src, VReal* des) {
  double sum = std::accumulate(src.begin(), src.end(), 0.0);
  for (size_t i = 0; i < src.size(); i++) {
    des->at(i) = src[i] / sum;
  }
}

inline double CrossEntropy(const VReal &lhs, const VReal &rhs) {
  VReal lhs1(lhs.size());
  VReal rhs1(rhs.size());
  Normal(lhs, &lhs1);
  Normal(rhs, &rhs1);
  double sum = 0;
  for (size_t i = 0; i < lhs.size(); i++) {
    sum += lhs1[i] * Log2(rhs1[i]);
  }
  return -sum;
}
} // namespace ml
#endif // ML_INFO_H_
