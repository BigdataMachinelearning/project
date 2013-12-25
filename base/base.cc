// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base.h"

#include <algorithm>

#include "base/random.h"

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

void Sum(const VVReal &src, VReal* des) {
  des->resize(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    des->at(i) = std::accumulate(src[i].begin(), src[i].end(), 0.0);
  }
}
