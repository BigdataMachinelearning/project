// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_MATH_UTIL_H_
#define BASE_MATH_UTIL_H_
inline void Subtract(double m, VReal* v) {
  for (size_t i = 0; i < v->size(); i++) {
    v->at(i) -= m;
  }
}

inline void Exp(VReal* v) {
  for (size_t i = 0; i < v->size(); i++) {
    v->at(i) = exp(v->at(i));
  }
}
#endif  // BASE_MATH_UTIL_H_
