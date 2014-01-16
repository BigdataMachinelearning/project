// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_RBM_RBM2_H_
#define ML_RBM_RBM2_H_
#include "base/base_head.h"
namespace ml2 {
struct User {
  VVInt item;
  VVReal rating;
};

struct RBM {
  VVVReal w1;
  VVReal b1;
  VReal c1;
  VVVReal dw;
  VVReal db;
  VReal dc;
  double momentum;
  double eta;
  int bach_size;
  void Init(int f, int m, int k, int bach, double momentum_, double eta_);
  // void Init(int m, int f, int k, int bach_size, double momentum_, double eta_);
  void InitZero();
};

void Update(const VInt &item, const VReal &h1, const VReal &v1, const VReal &h2,
                              const VReal &v2, RBM* rbm);
void RBMLearning(const User &train, const User &test, int iter_num, RBM* rbm);
void ExpectV(const VInt &item, const VReal &h, const RBM &rbm, VVReal* v);
void ExpectH(const VInt &item, const VReal &v, const RBM &rbm, VReal* h);
void SampleV(const VInt &item, const VReal &h, const RBM &rbm, VReal* v);
void SampleH(const VInt &item, const VReal &v, const RBM &rbm, VReal* h);
}  // namespace ml
#endif // ML_RBM_RBM2_H_
