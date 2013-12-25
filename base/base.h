// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_BASE_H_
#define BASE_BASE_H_
#include <cmath>
#include "base/type.h"
void Init(int len, double value, VReal* des);
void Init(int row, int col, double value, VVReal* des);
void Init(int len1, int len2, int len3, double value, VVVReal* des);
void Cumulate(VReal* des);

inline double Square(double a) {
  return a * a;
}

void Append(const VReal &src, VReal* des);
void Append(const VVReal &src, VReal* des);
void Append(const VVVReal &src, VReal* des);
void Sum(const VVReal &src, VReal* des);

class Time {
 public:
  void Start() { beg = clock(); }
  double GetTime() {
    return static_cast<double>(clock() - beg) / CLOCKS_PER_SEC;
  }
 private:
  int beg;
};
#endif  // BASE_BASE_H_
