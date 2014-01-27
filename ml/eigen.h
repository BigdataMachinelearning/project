// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#ifndef ML_EIGEN_H_
#define ML_EIGEN_H_
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "base/base_head.h"
typedef Eigen::SparseVector<double> SpVec;
typedef SpVec::InnerIterator SpVecInIt;

typedef Eigen::SparseMatrix<double> SpMat;
typedef SpMat::InnerIterator SpMatInIt;

typedef Eigen::MatrixXd EMat;
typedef Eigen::VectorXd EVec;

typedef Eigen::Triplet<double> Triple;
typedef std::vector<Eigen::Triplet<double> > TripleVec;

void ReadData(const Str &path, TripleVec* vec);
std::pair<int, int> Max(const TripleVec &vec);
std::pair<int, int> ReadData(const Str &path, SpMat* mat);
std::pair<int, int> ReadData(const Str &path, SpMat* train, SpMat* test);

void Sample(EVec *h);
void NormalRandom(EMat *mat);
void NormalRandom(EVec *vec);
void SplitData(const TripleVec &vec, double p, TripleVec* train,
                                               TripleVec* test);
#endif // ML_EIGEN_H_
