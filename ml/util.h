// Copyright 2013 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#ifndef ML_UTIL_H_
#define ML_UTIL_H_
#include "base/base_head.h"
namespace ml {
void Softmax(const VReal &a, VReal *b);
int Sample(const VReal &a);
int SoftmaxSample(const VReal &a);
double NormalSample();

void RandomInit(int len, VReal* des);
void RandomInit(int row, int col, VVReal* des);
void RandomInit(int len1, int len2, int len3, VVVReal* des);

void RandomOrder(int len, int random_num, VInt* des);

double Sum(const VReal &v);
double Sum(const VVReal &v);
double Sum(const VVVReal &v);

double Var(const VReal &v);
double Var(const VVReal &v);
double Var(const VVVReal &v);

double Mean(const VReal &v);
double Mean(const VVReal &v);
double Mean(const VVVReal &v);

double Quadratic(const VReal &lhs, const VReal &rhs, const VVReal &w);
double InnerProd(const VReal &lhs, const VReal &rhs);

template <typename V, typename M>
double Quadratic(const V &x, const V &y, const M &w) {
  double sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < y.size(); ++j) {
      sum += x[i] * w[i][j] * y[j];
    }
  }
  return sum;
}

template <typename V1, typename V2>
double InnerProd(const V1 &x, const V2 &y) {
  double sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}
} // namespace ml
#endif // ML_UTIL_H_
