// Copyright 2013 yuanwujun, lijiankou. All Rights Reserved.
// Author: real.yuanwj@gmail.com lijk_start@163.com
#ifndef ML_RBM_REPSOFTMAX_H_
#define ML_RBM_REPSOFTMAX_H_
#include "base/base_head.h"
#include "ml/document.h"
namespace ml {
struct RepSoftMax {
  VVReal w;
  VReal b;
  VReal c;
  VVReal dw;
  VReal db;
  VReal dc;
  
  double momentum;
  double eta;
  int bach_size;
  void Init(int f, int k, int bach_size_, double momentum_, double eta_);
  void InitZero();
};
void RBMLearning(const Corpus &corpus, int itern, RepSoftMax* rbm);
}  // namespace ml
#endif // ML_RBM_REPSOFTMAX_H_
