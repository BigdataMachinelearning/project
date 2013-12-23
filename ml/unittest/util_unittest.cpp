// copyright 2013 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/util.h"
#include "gtest/gtest.h"

namespace ml {
TEST(Util, SampleTest) {
  VReal v;
  v.push_back(0.25);
  v.push_back(0.25);
  v.push_back(0.25);
  v.push_back(0.25);
  MIntInt dic;
  for (int i = 0; i < 10000; i++) {
    dic[Sample(v)]++;
  }
  for (int i = 1; i < 5; i++) {
    EXPECT_LT(std::abs(dic[i] - 2500), 100);
  }
}

TEST(Util, SoftMaxTest) {
  VReal tmp;
  tmp.push_back(1);
  tmp.push_back(2);
  tmp.push_back(0.5);
  tmp.push_back(0.7);
  VReal tmp2;
  tmp2.resize(tmp.size());
  Softmax(tmp, &tmp2);
  LOG(INFO) << Join(tmp2, " ");
}

TEST(Util, MeanTest) {
  VReal tmp;
  tmp.push_back(1);
  tmp.push_back(2);
  tmp.push_back(3);
  EXPECT_DOUBLE_EQ(2, Mean(tmp));
}

TEST(Util, VarTest) {
  VReal tmp;
  tmp.push_back(1);
  tmp.push_back(2);
  tmp.push_back(3);
  EXPECT_DOUBLE_EQ(2.0 / 3, Var(tmp));
  VVReal tmp2;
  tmp2.push_back(tmp);
  tmp2.push_back(tmp);
  EXPECT_DOUBLE_EQ(2.0 / 3, Var(tmp2));
  VVVReal tmp3;
  tmp3.push_back(tmp2);
  tmp3.push_back(tmp2);
  EXPECT_DOUBLE_EQ(2.0 / 3, Var(tmp3));
}

TEST(Util, NormalSampleTest) {
  VReal tmp;
  for (int i = 0; i < 1000000; i++) {
    tmp.push_back(NormalSample() / 100);
  }
  EXPECT_LT(std::abs(0.0 - Mean(tmp)), 0.00001);
  EXPECT_LT(std::abs(0.0001 - Var(tmp)), 0.00001);
  VVVReal tmp2;
  RandomInit(100, 100, 100, &tmp2);
  EXPECT_LT(std::abs(0.0 - Mean(tmp2)), 0.00001);
  EXPECT_LT(std::abs(0.0001 - Var(tmp2)), 0.00001);
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
