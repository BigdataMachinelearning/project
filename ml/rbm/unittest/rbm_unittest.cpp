// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/ais.h"
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"
#include "ml/rbm/rbm2.h"
#include "ml/rbm/repsoftmax.h"
#include "gtest/gtest.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
namespace ml {
/*
TEST(RBMTest, BaiduLoadTest) {
  Str name = "data/baidu_format.txt";
  User user;
  LoadBaidu(name, &user);
  User train;
  User test;
  SplitData(user, 0.8, &train, &test);
  SaveBaidu("data/baidu_train.txt", train);
  SaveBaidu("data/baidu_test.txt", test);
  EXPECT_LT(std::abs(5.0 - user.rating[10][0]), 0.00001);
  EXPECT_EQ(568, user.item[10][0]);
}

void Init(ml2::RBM* rbm) {
  int m = 7890;
  int f = 100;
  int k = 6;
  double momentum = 0.0;
  double eta = 0.00001;
  int bach_size = 10;
  rbm->Init(f, m, k, bach_size, momentum, eta);
}

TEST(RBMTest, BaiduTest) {
  ml2::RBM rbm;
  Init(&rbm);
  Str name = "data/baidu_format.txt";
  User train;
  User test;
  LoadBaidu(name, 0.8, &train, &test);
  ml2::RBMLearning(train, test, 200, &rbm);
  RBMTest(train, test, rbm);
}

void InitMovieLen(ml2::RBM* rbm) {
  int m = 2000;
  int f = 100;
  int k = 6;
  double momentum = 0.0;
  double eta = 0.002;
  int bach_size = 100;
  rbm->Init(f, m, k, bach_size, momentum, eta);
}

void LoadMovieLen(User* train, User* test) {
  Str train_file = "data/u1.base";
  // Str train_file = "data/train_g20.txt";
  Str test_file = "data/u1.test";
  // Str test_file = "data/test_g20.txt";
  LoadMovieLen(train_file, train);
  LoadMovieLen(test_file, test);
}

TEST(RBMTest, LoadMovieLenTest) {
  User train;
  User test;
  LoadMovieLen(&train, &test);
  EXPECT_EQ(3, train.rating[1][1]);
  EXPECT_EQ(4, train.rating[1][2]);
  EXPECT_EQ(3, test.rating[460][0]);
  EXPECT_EQ(5, test.rating[462][0]);
}

TEST(RBMTest, MovieLenTest) {
  User train;
  User test;
  LoadMovieLen(&train, &test);
  ml2::RBM rbm;
  InitMovieLen(&rbm);
  ml2::RBMLearning(train, test, 2000, rbm.bach_size, &rbm);
  // RBMTest(train, test, rbm);
}

TEST(EigenRBMTest, MovieLenTest) {
  #define TRAIN_PATH "data/train_g20.txt"
  #define TEST_PATH "data/test_g20.txt"
  // #define TRAIN_PATH "data/baidu_train.txt"
  // #define TEST_PATH "data/baidu_test.txt"
  #define N_HIDDEN 100
  #define N_SOFTMAX 5
  #define N_samples 200
  #define N_skip 50
  #define N_steps 30
  SpMat u_v, u_t, t_s, test_u_v;
  ReadData(TRAIN_PATH, 0, 0, &u_v);
  SpMat v_u = u_v.transpose();
  ReadData(TEST_PATH, u_v.rows(), u_v.cols(), &test_u_v);
  SpMat test_v_u = test_u_v.transpose();
  int M = u_v.cols();
  int N = u_v.rows();

  RBM rbm(u_v, N, N_HIDDEN, N_SOFTMAX);
  rbm.Train(u_v, test_u_v, 2000, 0.1, 100);
}
*/

TEST(Ais, UniformSampleTest) {
  Corpus c;
  Str dat = "../data/document_demo";
  c.LoadData(dat);
  VInt v(c.TermNum());
  UniformSample(c.docs[0], &v);
  LOG(INFO) << Join(v, " ");
}

TEST(Ais, AisTest) {
  Corpus corpus;
  Str dat = "../data/document_demo";
  corpus.LoadData(dat);
  int bach_size = 2;
  double eta = 0.0001;
  int k = 2;
  int it_num = 2000;
  RepSoftMax rep;
  rep.Init(k, corpus.num_terms, bach_size, 1, eta);
  RBMLearning(corpus, it_num, &rep);
  int run = 10;
  VReal beta(10);
  for (size_t i = 0; i < beta.size(); i++) {
    beta[i] = 0.1 * i;
  }
  LOG(INFO) << Probability(corpus.docs[0], run, beta, rep);
}

} // namespace ml

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
