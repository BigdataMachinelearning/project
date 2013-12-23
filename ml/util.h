// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_UTIL_H_
#define ML_UTIL_H_
#include "base/base_head.h"
namespace ml {
void Softmax(const VReal &a, VReal *b);
int Sample(const VReal &a);
int SoftmaxSample(const VReal &a, VReal *b);
double NormalSample();

void RandomInit(int len, VReal* des);
void RandomInit(int row, int col, VVReal* des);
void RandomInit(int len1, int len2, int len3, VVVReal* des);

double Sum(const VReal &v);
double Mean(const VReal &v);
double Var(const VReal &v);
double Var(const VVReal &v);
double Var(const VVVReal &v);
double Mean(const VReal &v);
double Mean(const VVReal &v);
double Mean(const VVVReal &v);
} // namespace ml
#endif // ML_UTIL_H_
