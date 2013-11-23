// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_BASE_H_
#define BASE_BASE_H_ 
#include <cmath>
#include "base/type.h"

inline void Init(int len, double value, VReal* des) {
  for (int i = 0; i < len; i++) {
    des->push_back(value);
  }
}

inline void Init(int row, int col, double value, VVReal* des) {
  for (int i = 0; i < row; i++) {
    VReal tmp;
    Init(col, value, &tmp);
    des->push_back(tmp);
  }
}

inline void Cumulate(VReal* des) {
  for (VReal::size_type i = 1; i < des->size(); i++) {
    des->at(i) += des->at(i - 1);
  }
}

inline int Random(VRealC &data) {
  VReal tmp(data);
  Cumulate(&tmp);
  double u = ((double) random() / RAND_MAX) * tmp[tmp.size() - 1];
  for (VReal::size_type i = 0; i < tmp.size(); i++) {
    if (tmp[i] > u) {
      return i;
    }
  }
  return static_cast<int>(tmp.size());
}

inline int Random(int k) {
  double u = ((double) random() / RAND_MAX) * k;
  return std::floor(u);
}
#endif // BASE_H_
