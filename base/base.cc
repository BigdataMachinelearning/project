// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base.h"

void Init(int len, double value, VReal* des) {
  VReal tmp(len, value);
  des->swap(tmp);
}

void Init(int row, int col, double value, VVReal* des) {
  VReal tmp;
  Init(col, value, &tmp);
  VVReal tmp2(row, tmp);
  des->swap(tmp2);
}

void Init(int len1, int len2, int len3, double value, VVVReal* des) {
  VVReal tmp;
  Init(len2, len3, value, &tmp);
  VVVReal tmp2(len1, tmp);
  des->swap(tmp2);
}

void Cumulate(VReal* des) {
  for (VReal::size_type i = 1; i < des->size(); i++) {
    des->at(i) += des->at(i - 1);
  }
}

int Random(const VReal &data) {
  VReal tmp(data);
  Cumulate(&tmp);
  double u = (static_cast<double>(random()) / RAND_MAX) * tmp[tmp.size() - 1];
  for (VReal::size_type i = 0; i < tmp.size(); i++) {
    if (tmp[i] > u) {
      return i;
    }
  }
  return static_cast<int>(tmp.size());
}

int Random(int k) {
  double u = (static_cast<double>(random()) / RAND_MAX) * k;
  return std::floor(u);
}
