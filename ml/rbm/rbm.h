// Copyright 2013 zhangwei, lijiankou. All Rights Reserved.
// Author: zhangw@ios.ac.cn  lijk_start@163.com
#ifndef ML_RBM_RBM_
#define ML_RBM_RBM_
#include "ml/eigen.h"
#include <vector>
namespace ml {
class RBM {
 public:
  RBM(const SpMat &train, int nv, int nh, int nsoftmax);
  void Train(const SpMat &train, const SpMat &test, int niter,
                                 double alpha, int batch_size);
  double Predict(const SpMat &train, const SpMat &test);
 public:
  SpVec v0, vk;
  EVec h0, hk;
 private:
  std::vector<EMat> W, dW;
  EMat bv, dv;
  EVec bh, dh;
  void InitGradient();
  void UpdateGradient(double alpha, int batch_size);
  void ExpectH(const SpVec &v, EVec *h);
  void SampleH(const SpVec &v, EVec *h);
  void ExpectRating(const EVec &h, const SpVec &t, SpVec *v);
  void ExpectV(const EVec &h, const SpVec &t, VVReal* des);
  void SampleV(const EVec &h, const SpVec &t, SpVec *v);
  void PartGrad(const SpVec &v, const EVec &h, double coeff);
  void Gradient(const SpVec &x, int step);
};
} // namespace ml
#endif // ML_RBM_RBM_
