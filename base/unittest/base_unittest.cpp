// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include <fstream>

#include "base/base_head.h"
#include "gtest/gtest.h"

TEST(StrUtil, SplitStrTEST) {
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

TEST(StrUtil, TrimTEST) {
  Str str("  this\t");
  Str str2;
  TrimStr(str, " \t", &str2);
  EXPECT_EQ("this", str2);
  EXPECT_EQ("this", TrimStr(str));
}

TEST(StrUtil, StartWithTEST) {
  Str str("this");
  EXPECT_TRUE(StartWith(str, "th"));
  EXPECT_FALSE(StartWith(str, "ax"));
}

TEST(StrUtil, EndWithTEST) {
  Str str("this");
  EXPECT_TRUE(EndWith(str, "is"));
  EXPECT_FALSE(EndWith(str, "  "));
}

TEST(StrUtil, LowerTEST) {
  Str str("tHiS");
  EXPECT_EQ("this", Lower(str));
}

TEST(Join, MapToStr) {
  MIntInt m;
  m[1] = 1;
  m[2] = 2;
  EXPECT_EQ("1 1 \n2 2 \n", MapToStr(m));
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

TEST(Join, IntTEST) {
  VInt data;
  data.push_back(1);
  data.push_back(2);
  EXPECT_EQ("1 2 ", Join(data, " "));
  VVInt data2;
  data2.push_back(data);
  data2.push_back(data);
  EXPECT_EQ("1 2 \n1 2 \n", Join(data2, " ", "\n"));
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

TEST(BaseTest, RandomTest) {
  VReal tmp;
  tmp.push_back(20);
  tmp.push_back(10);
  tmp.push_back(10);
  int num = 100000;
  MIntInt m;
  for (int i = 0; i < num; i++) {
    m[Random(tmp)]++;
  }
  int num2 = 25000;
  int num3 = 50000;
  EXPECT_LT(std::abs(num3 - m[0]), 100);
  EXPECT_LT(std::abs(num2 - m[1]), 100);
  EXPECT_LT(std::abs(num2 - m[2]), 100);
}

TEST(BaseTest, RandomTest2) {
  int k = 4;
  int num = 100000;
  MIntInt m;
  for (int i = 0; i < num; i++) {
    m[Random(k)]++;
  }
  VReal r(k);
  for (int i = 0; i < k; i++) {
    r[i] = static_cast<double>(m[i]) / num;
  }
  double precision = 0.01;
  double p = 0.25;
  EXPECT_LT(std::abs(p - r[0]), precision);
  EXPECT_LT(std::abs(p - r[1]), precision);
  EXPECT_LT(std::abs(p - r[2]), precision);
  EXPECT_LT(std::abs(p - r[3]), precision);
}

TEST(Join, InitTEST) {
  VReal tmp;
  Init(2, 0.0, &tmp);
  EXPECT_EQ("0 0 ", Join(tmp, " "));
  VVReal tmp2;
  Init(2, 2, 0.0, &tmp2);
  EXPECT_EQ("0 0 \n0 0 \n", Join(tmp2, " ", "\n"));

  VReal tmp3;
  Init(2, 1, &tmp3);
  EXPECT_EQ("1 1 ", Join(tmp3, " "));
  VVInt tmp4;
  Init(2, 2, 1, &tmp4);
  EXPECT_EQ("1 1 \n1 1 \n", Join(tmp4, " ", "\n"));
}

TEST(Random, SigmoidTest) {
  double a = 0;
  MIntInt m;
  for (int i = 0; i < 1000; i++) {
    m[SigmoidSample(a)]++;
  }
  EXPECT_LT(500 - m[0], 50);
}

TEST(Random, Sample1Test) {
  double a = 0;
  MIntInt m;
  for (int i = 0; i < 1000; i++) {
    m[Sample1(a)]++;
  }
  EXPECT_LT(500 - m[0], 50);
}


TEST(BaseTest, SquareTEST) {
  double a = 2.0;
  EXPECT_LT(std::abs(Square(a) - 4), 0.001);
}

TEST(StatTest, ProbabilityTEST) {
  VReal v(4, 1.0);
  VReal v2;
  Probability(v, &v2);
  EXPECT_LT(std::abs(v2.at(0) - 0.25), 0.001);
}

TEST(StatTest, ExpectTEST) {
  VReal v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  VReal v2;
  Probability(v, &v2);
  EXPECT_LT(std::abs(Expect(v2) - 3), 0.000001);
}

TEST(BaseTest, SumTEST) {
  VReal v;
  v.push_back(1);
  v.push_back(2);
  VVReal v2;
  v2.push_back(v);
  v2.push_back(v);
  VReal v3;
  Sum(v2, &v3);
  EXPECT_EQ(3, v3[0]);
  EXPECT_EQ(3, v3[1]);
}

TEST(StlUtil, ToSetTEST) {
  VInt v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(2);
  v.push_back(3);
  SInt s;
  ToSet(v, &s);
  EXPECT_EQ(3, s.size());
  EXPECT_EQ("1 2 3 ", Join(s, " "));
}

TEST(StlUtil, DiffNumTEST) {
  VInt v;
  v.push_back(1);
  v.push_back(2);
  VInt v2;
  v2.push_back(1);
  v2.push_back(2);
  EXPECT_EQ(0, DiffNum(v, v2));
  v.push_back(2);
  v2.push_back(1);
  EXPECT_EQ(1, DiffNum(v, v2));
}

TEST(Random, UniformSampleTEST) {
  VInt v(3);
  int len = 3000;
  UniformSample(len, &v);
  EXPECT_LT(std::abs(v[0] - 1000), 100);
  EXPECT_LT(std::abs(v[1] - 1000), 100);
  EXPECT_LT(std::abs(v[2] - 1000), 100);
}

TEST(StlUtil, MultiplyTEST) {
  VReal v(2);
  v[0] = 1;
  v[1] = 2;
  VVReal v2(2, v);
  VReal v3(v.size());
  double m = 2;
  Multiply(v, m, &v);
  EXPECT_DOUBLE_EQ(v[0], 2);
  EXPECT_DOUBLE_EQ(v[1], 4);
  Multiply(v2, m, &v2);
}

TEST(Probability, NextMutiSeqTEST) {
  int num = 10;
  VInt v(4, 0);
  v[0] = 10;
  while (NextMultiSeq(num, &v)) {
    EXPECT_EQ(Sum(v), 10);
  }
}

struct F{
  int* a;
  int size;
};

/*
TEST(IOUtilTest, FIOTEST) {
  int a[2];
  a[0] = 1;
  a[1] = 2;
  std::ofstream out("test", std::ios::binary);
  out.write((char*)(&a[0]), sizeof(a));
  out.close();

  VInt v2(2);
  std::ifstream in("test", std::ios::binary);
  in.read((char*)(&v2[0]), sizeof(v2));
  in.close( );
  LOG(INFO) << Join(v2, " ");
  LOG(INFO) << sizeof(a);
  LOG(INFO) << sizeof(v2);
}
*/

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
