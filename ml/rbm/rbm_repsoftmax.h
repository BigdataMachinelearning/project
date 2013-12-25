// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef RBM_DOCUMENT_1_H_
#define RBM_DOCUMENT_1_H_
#include "base/base_head.h"
#include "ml/document.h"
namespace ml {
struct RBM_RepSoftMax {
  VVReal w;
  VReal b;
  VReal c;
  VVReal dw;
  VReal db;
  VReal dc;
  
  double momentum;
  double eta;
  int bach_size;
  void Init(int f, int k, double momentum_, double eta_);
};
void RBMLearning(const Corpus &corpus, int itern, RBM_RepSoftMax* rbm);
}  // namespace ml
#endif // RBM_DOCUMENT_1_H_
