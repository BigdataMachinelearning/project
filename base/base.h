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
int Random(const VReal &data);
int Random(int k);

inline double Sigmoid(double a) {
  return 1.0 / (1 + exp(-a));
}

inline double Random1() {
  return static_cast<double>(random()) / RAND_MAX;
}

inline int RandSample(double a) {
  return Random1() < a ? 1 : 0;
}

inline int SigRand(double a) {
  return Random1() < Sigmoid(a) ? 1 : 0;
}

inline double Square(double a) {
  return a * a;
}

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
