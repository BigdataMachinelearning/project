// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
#include "gtest/gtest.h"
namespace ml {
TEST(RBMTest, BaiduLoadTest) {
  Str name = "data/baidu_format.txt";
  User user;
  LoadBaidu(name, &user);
  EXPECT_LT(std::abs(5.0 - user.rating[10][0]), 0.00001);
  EXPECT_EQ(568, user.item[10][0]);
}

void Init(RBM* rbm) {
  int m = 7890;
  int f = 100;
  int k = 6;
  double momentum = 0.0;
  double eta = 0.00001;
  rbm->Init(f, m, k, momentum, eta);
}

TEST(RBMTest, BaiduTest) {
  RBM rbm;
  Init(&rbm);
  Str name = "data/baidu_format.txt";
  User train;
  User test;
  LoadBaidu(name, 0.8, &train, &test);
  RBMTrain(train, test, 200, &rbm);
  RBMTest(train, test, rbm);
}

TEST(RBMTest, SoftMaxTest) {
  VReal tmp;
  tmp.push_back(1);
  tmp.push_back(2);
  tmp.push_back(0.5);
  tmp.push_back(0.7);
  VReal tmp2;
  SoftMax(tmp, &tmp2);
  LOG(INFO) << Join(tmp2, " ");
}

void InitMovieLen(RBM* rbm) {
  int m = 2000;
  int f = 100;
  int k = 6;
  double momentum = 0.0;
  double eta = 0.000002;
  rbm->Init(f, m, k, momentum, eta);
}

void LoadMovieLen(User* train, User* test) {
  Str train_file = "data/u1.base";
  LoadMovieLen(train_file, train);
  Str test_file = "data/u1.test";
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
  RBM rbm;
  InitMovieLen(&rbm);
  RBMTrain(train, test, 2000, &rbm);
  // RBMTest(train, test, rbm);
}
} // namespace ml

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
