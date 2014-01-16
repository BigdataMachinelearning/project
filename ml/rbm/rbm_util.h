// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef RBM_RBM_UTIL_H_
#define RBM_RBM_UTIL_H_
#include "base/base_head.h"
#include "ml/rbm/rbm2.h"
#include "ml/util.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
namespace ml {
using ml2::User;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::Triplet<double> T;

void LoadBaidu(const Str &name, User* user);
void LoadBaidu(const Str &name, double pro, User* train, User* test);
void SaveBaidu(const Str &name, const User &user);
void LoadMovieLen(const Str &name, User* user);

void SplitData(const User &user, double value, User* train, User* test);
double SquareError(const VReal &lhs, const VReal &rhs);
double RBMTest(const User &train, const User &test, const ml2::RBM &rbm);
size_t Size(const VVInt &item);
void ReadData(const Str &path, int rows, int cols, SpMat *m);
void ReadData(const Str &path, SpMat *mat);

inline double ExpectRating(const VReal &a){
  double s = 0;
  for(size_t i = 0; i < a.size(); ++i){
    s += (i + 1) * a[i];
  }
  return s;
}

using Eigen::MatrixXd;
using Eigen::VectorXd;

int MaxItemId(const User &user);
void RandomInit(MatrixXd* des);
void Convert(const std::vector<MatrixXd> &src, VVVReal* des);

inline double Var(const std::vector<MatrixXd> &src) {
  VVVReal tmp;
  Convert(src, &tmp);
  return Var(tmp);
}

inline double Mean(const std::vector<MatrixXd> &src) {
  VVVReal tmp;
  Convert(src, &tmp);
  return Mean(tmp);
}
}  // namespace ml
#endif // RBM_RBM_UTIL_H_
