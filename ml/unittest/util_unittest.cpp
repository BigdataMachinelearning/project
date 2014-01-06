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
  LOG(INFO) << std::accumulate(tmp2.begin(), tmp2.end(), 0.0);
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

TEST(Util, SumTest) {
  VVReal tmp;
  Init(2, 2, 1.0, &tmp);
  EXPECT_DOUBLE_EQ(4, Sum(tmp));
  VVVReal tmp2;
  Init(2, 2, 2, 2.0, &tmp2);
  EXPECT_DOUBLE_EQ(16, Sum(tmp2));
}

TEST(Util, RandomOrderTest) {
  VInt tmp;
  RandomOrder(100, 1000, &tmp);
  SInt dic;
  ToSet(tmp, &dic);
  EXPECT_EQ(100, dic.size());
  int c = 0;
  for (size_t i = 0; i < tmp.size(); i++) {
    if (tmp[i] == i) {
      c++;
    }
  }
  EXPECT_LT(c, 3);
}

TEST(Util, QuadraticTest) {
  VReal lhs(2);
  lhs[0] = 1;
  lhs[1] = 2;
  VReal rhs(3);
  rhs[0] = 3;
  rhs[1] = 4;
  rhs[2] = 5;
  VVReal w;
  Init(lhs.size(), rhs.size(), 0, &w);
  w[0][0] = 1;
  w[0][1] = 2;
  w[0][2] = 3;
  w[1][0] = 4;
  w[1][1] = 5;
  w[1][2] = 6;
  EXPECT_DOUBLE_EQ(150, Quadratic(lhs, rhs, w));
}

TEST(Util, InnerProdTest) {
  VReal lhs(2);
  lhs[0] = 1;
  lhs[1] = 2;
  VReal rhs(2);
  rhs[0] = 3;
  rhs[1] = 4;
  EXPECT_DOUBLE_EQ(11, InnerProd(lhs, rhs));
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
