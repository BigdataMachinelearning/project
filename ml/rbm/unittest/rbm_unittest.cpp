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
TEST(RBMTest, BaiduLoadTest) {
  Str name = "tmp/baidu_format.txt";
  User user;
  LoadBaidu(name, &user);
  User train;
  User test;
  SplitData(user, 0.8, &train, &test);
  SaveBaidu("tmp/baidu_train.txt", train);
  SaveBaidu("tmp/baidu_test.txt", test);
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
  Str name = "tmp/baidu_format.txt";
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
  double eta = 0.1;
  int bach_size = 100;
  rbm->Init(f, m, k, bach_size, momentum, eta);
}

void LoadMovieLen(User* train, User* test) {
  // Str train_file = "tmp/u1.base";
  Str train_file = "tmp/train1";
  // Str train_file = "tmp/train_g20.txt";
  // Str test_file = "tmp/u1.test";
  Str test_file = "tmp/test1";
  // Str test_file = "tmp/test_g20.txt";
  LoadMovieLen(train_file, train);
  LoadMovieLen(test_file, test);
}

TEST(RBMTest, LoadMovieLenTest) {
  User train;
  User test;
  LoadMovieLen(&train, &test);
  /*
  EXPECT_EQ(944, train.item.size());
  EXPECT_EQ(5, train.rating[1][0]);
  EXPECT_EQ(3, train.rating[1][1]);
  EXPECT_EQ(4, train.rating[1][2]);
  EXPECT_EQ(3, test.rating[459][0]);
  EXPECT_EQ(3, test.rating[460][0]);
  EXPECT_EQ(5, test.rating[462][0]);
  */
}

TEST(RBMTest, MovieLenTest) {
  User train;
  User test;
  LoadMovieLen(&train, &test);
  ml2::RBM rbm;
  InitMovieLen(&rbm);
  ml2::RBMLearning(train, test, 2000, &rbm);
  // RBMTest(train, test, rbm);
}

TEST(EigenRBMTest, MovieLenTest) {
  #define TRAIN_PATH "tmp/train_g20.txt"
  #define TEST_PATH "tmp/test_g20.txt"
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
  LOG(INFO) << Likelihood(corpus.docs[0], run, beta, rep);
}

// test Partition = 2^F
TEST(Ais, LogPartitionTest) {
  RepSoftMax rep;
  int size_f = 1;
  int size_v = 2;
  ZeroRep(size_f, size_v, &rep);
  int doc_len = 2;
  int word_num = 2;
  double beta_a = 1;
  EXPECT_DOUBLE_EQ(2, exp(LogMultiPartition(doc_len, word_num, beta_a, rep)));
  size_f = 2;
  ZeroRep(size_f, size_v, &rep);
  EXPECT_DOUBLE_EQ(4, exp(LogMultiPartition(doc_len, word_num, beta_a, rep)));
}

TEST(Ais, WAisTest) {
  Corpus corpus;
  corpus.LoadData("test");
  RepSoftMax rep;
  int f_size = 1;
  int v_size = 2;
  double value = 0;
  InitRep(f_size, v_size, value, &rep);
  VReal beta;
  Range(0, 1, 0.01, &beta);
  int ais_run = 10;
  double wais = WAis(corpus.docs[0], ais_run, beta, rep);
  double z = wais * pow(2, rep.c.size()) * rep.b.size(); 
  double p = LogPartition(corpus.TLen(0), corpus.ULen(0), rep);
  EXPECT_DOUBLE_EQ(z, exp(p));
}
} // namespace ml

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
