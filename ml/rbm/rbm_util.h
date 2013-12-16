// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef RBM_RBM_UTIL_H_
#define RBM_RBM_UTIL_H_
#include "base/base_head.h"
#include "ml/rbm/rbm.h"
namespace ml {
void LoadBaidu(const Str &name, User* user);
void LoadBaidu(const Str &name, double pro, User* train, User* test);
void LoadMovieLen(const Str &name, User* user);
void SplitData(const User &user, double value, User* train, User* test);
size_t Size(const VVInt &item);
double SoftMax(const VReal &data, double value);
void SoftMax(const VReal &data, VReal* des);
double SquareError(const VReal &lhs, const VReal &rhs);
double RBMTest(const User &train, const User &test, const RBM &rbm);
}  // namespace ml
#endif // RBM_RBM_UTIL_H_
