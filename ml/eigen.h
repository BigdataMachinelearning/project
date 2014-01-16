// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#ifndef ML_EIGEN_H_
#define ML_EIGEN_H_
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "base/base_head.h"
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;

typedef Eigen::MatrixXd EMat;
typedef Eigen::VectorXd EVec;

inline void Sample(EVec *h) {
  for (int i = 0; i < h->size(); ++i) {
    (*h)[i] = Sample1((*h)[i]);
  }
}
#endif // ML_EIGEN_H_
