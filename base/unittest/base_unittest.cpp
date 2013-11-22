// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "gtest/gtest.h"

TEST(StringUtil, SplitStrTEST) {
  Str str("this,is,a,test");
  VStr vec;
  SplitStr(str, ',', &vec);
  EXPECT_EQ(4, vec.size());
  EXPECT_EQ("this", vec[0]);
  EXPECT_EQ("is", vec[1]);
  EXPECT_EQ("a", vec[2]);
  EXPECT_EQ("test", vec[3]);
  vec.clear();
  SplitStr(str, ",", &vec);
  EXPECT_EQ(4, vec.size());
  EXPECT_EQ("this", vec[0]);
  EXPECT_EQ("is", vec[1]);
  EXPECT_EQ("a", vec[2]);
  EXPECT_EQ("test", vec[3]);
}

TEST(StringUtil, TrimTEST) {
  Str str("  this\t");
  Str str2;
  TrimStr(str, " \t", &str2);
  EXPECT_EQ("this", str2);
  EXPECT_EQ("this", TrimStr(str));
}

TEST(StringUtil, StartWithTEST) {
  Str str("this");
  EXPECT_TRUE(StartWith(str, "th"));
  EXPECT_FALSE(StartWith(str, "ax"));
}

TEST(StringUtil, EndWithTEST) {
  Str str("this");
  EXPECT_TRUE(EndWith(str, "is"));
  EXPECT_FALSE(EndWith(str, "  "));
}

TEST(StringUtil, LowerTEST) {
  Str str("tHiS");
  EXPECT_EQ("this", Lower(str));
}

TEST(Join, JoinTEST) {
  VStr v;
  v.push_back("hello");
  v.push_back("world");
  EXPECT_EQ("hello world ", Join(v, " "));
  VVStr v2;
  v2.push_back(v);
  v2.push_back(v);
  EXPECT_EQ("hello world \nhello world \n", Join(v2, " ", "\n"));
  int len1 = 2;
  int len2 = 3;
  double**a = NewArray(len1, len2);
  Init(len1, len2, 1, a);
  EXPECT_EQ("1 1 1 \n1 1 1 \n", Join(a, len1, len2));
  DelArray(a, len1);
}

TEST(StatTest, LogSumTest) {
  EXPECT_LT(std::abs(1.69315 - LogSum(1, 1)), 0.0001);
}

TEST(StatTest, TriGammaTest) {
  EXPECT_LT(std::abs(0.105166 - TriGamma(10)), 0.0001);
}

TEST(StatTest, DigGmmaTest) {
  EXPECT_LT(std::abs(2.25175 - DiGamma(10)), 0.0001);
}

TEST(StatTest, LoggammaTest) {
  EXPECT_LT(std::abs(-94.0718 - LogGamma(0.1)), 0.0001);
}

TEST(StatTest, MaxTest) {
  double a[] = {1, 2, 40.0, 10, -1};
  EXPECT_LT(std::abs(40 - Max(a, 5)), 0.0001);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
