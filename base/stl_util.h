// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_STL_UTIL_H_
#define BASE_STL_UTIL_H_
#include "base/stl_util.h"

inline void Add(const VInt &lhs, const VInt &rhs, VInt* des) {
  des->resize(lhs.size());
  for (size_t i = 0; i < lhs.size(); i++) {
    des->at(i) = lhs[i] + rhs[i];
  }
}
#endif // BASE_STL_UTIL
