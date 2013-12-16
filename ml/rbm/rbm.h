// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef RBM_RBM_H_
#define RBM_RBM_H_
#include "base/base_head.h"
namespace ml {
struct User {
  VVInt item;
  VVReal rating;
};

struct RBM {
  VVVReal w1;
  VVReal b1;
  VReal c1;
  double momentum;
  double eta;
  void Init(int m, int f, int k, double momentum_, double eta_);
};

void Update(const VInt &item, const VReal &h1, const VReal &v1, const VReal &h2,
                              const VReal &v2, RBM* rbm);
void RBMTrain(const User &train, const User &test, int iter_num, RBM* rbm);
void ExpectV(const VInt &item, const VReal &h, const RBM &rbm, VVReal* v);
void ExpectH(const VInt &item, const VReal &rating, const RBM &rbm, VReal* h);
void SampleV(const VInt &item, const VReal &h, const RBM &rbm, VReal* v);
void SampleH(const VInt &item, const VReal &rating, const RBM &rbm, VReal* h);
}  // namespace ml
#endif // RBM_RBM_H_
