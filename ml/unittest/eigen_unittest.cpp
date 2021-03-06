// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/matrix_factorization.h"
#include "ml/util.h"
#include "gtest/gtest.h"

namespace ml {
TEST(Eigen, DemoTest) {
  TripleVec vec;
  vec.push_back(Triple(5, 0, 1));
  vec.push_back(Triple(4, 0, 2));
  vec.push_back(Triple(2, 1, 3));
  vec.push_back(Triple(3, 2, 4));
  vec.push_back(Triple(2, 1, 4));
  EMat m3(2, 3);
  SpMat m(6, 3);
  m.setFromTriplets(vec.begin(), vec.end());
  EVec v(2);
  v[1] = 1;
  v[0] = 1;
  m3.col(1) = v; 
  m3.col(0) = v; 
  m3.col(2) = v; 
  LOG(INFO) << m;
  LOG(INFO) << m3;
  LOG(INFO) << m.size();
  LOG(INFO) << m.innerSize();
  LOG(INFO) << m.rows();
}

TEST(Eigen, ReadDataTest) {
  Str path("rbm/tmp/train_g20.txt");
  TripleVec vec;
  ReadData(path, &vec);
  EXPECT_EQ(79951, vec.size());
  std::pair<int, int> p = Max(vec);
  EXPECT_EQ(1682, p.first);
  EXPECT_EQ(943, p.second);
  SpMat mat;
  ReadData(path, &mat);
  EXPECT_EQ(943, mat.cols());
}

TEST(Eigen, NormalRandomTest) {
  EMat t(2, 3);
  NormalRandom(&t);
  LOG(INFO) << t.transpose();
}

TEST(MatrixFactorization, GradientDescent) {
  Str path1("../data/movielen_train.txt");
  Str path2("../data/movielen_test.txt");
  SpMat train;
  SpMat test;
  std::pair<int, int> p = ReadData(path1, &train);
  ReadData(path2, &test);
  double eta = 0.005;
  int it_num = 100000;
  double lambda = 0.12;
  int k = 5;
  EMat v(k, p.first);
  EMat u(k, p.second);
  v.setRandom();
  u.setRandom();
  // BGD(it_num, eta, lambda, train, test, &u, &v);
  SGD(it_num, eta, lambda, train, test, &u, &v);
}

TEST(MatrixFactorization, FengXing) {
  // Str path1("rbm/tmp/r_1");
  Str path1("rbm/tmp/tr_nozero");
  // Str path1("rbm/tmp/rating");
  SpMat train;
  SpMat test;
  std::pair<int, int> p = ReadData(path1, &train, &test);
  double eta = 0.01;
  int it_num = 100000;
  double lambda = 0.8;
  int k = 10;
  EMat v(k, p.first);
  EMat u(k, p.second);
  v.setRandom();
  u.setRandom();
  // BGD(it_num, eta, lambda, train, test, &u, &v);
  SGD(it_num, eta, lambda, train, test, &u, &v);
}

TEST(MatrixFactorization, GradientDescent2) {
  Str path1("../data/movielen_train.txt");
  Str path2("../data/movielen_test.txt");
  SpMat train;
  SpMat test;
  std::pair<int, int> p = ReadData(path1, &train);
  ReadData(path2, &test);
  double eta = 0.005;
  int it_num = 100000;
  double lambda = 0.1;
  int k = 10;
  MF mf;
  RandomInit(p.second, p.first, k, &mf);
  LOG(INFO) << k;
  SGD(it_num, eta, lambda, train, test, &mf);
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
