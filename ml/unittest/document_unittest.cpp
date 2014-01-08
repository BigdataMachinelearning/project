// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/document.h"
#include "gtest/gtest.h"

namespace ml {
TEST(Document, ReadDataTest) {
  Str data = "../data/ap.dat";
  Corpus c;
  c.LoadData(data);
  EXPECT_EQ(10473,  c.num_terms);
  EXPECT_EQ(2246,  c.docs.size());
  EXPECT_EQ(186,  c.docs[0].ULen());
  EXPECT_EQ(263,  c.TLen(0));
  EXPECT_EQ(6144,  c.docs[0].words[1]);
  EXPECT_EQ(1,  c.docs[0].counts[1]);
}

TEST(Document, LoadDataTest) {
  Str data = "data/document_demo";
  Corpus c;
  c.LoadData(data);
  EXPECT_EQ(6,  c.Len());
  EXPECT_EQ(4,  c.ULen(0));
  EXPECT_EQ(10,  c.Count(0, 0));
  EXPECT_EQ(4,  c.TermNum());
}

TEST(Document, SplitDataTest) {
  Str data = "data/document_demo";
  Corpus c;
  c.LoadData(data);
  Corpus train;
  Corpus test;
  double value = 0.5;
  SplitData(c, 0.8, &train, &test);
  LOG(INFO) << train.Len();
  LOG(INFO) << test.Len();
}

TEST(Document, RandomOrderTest) {
  Str data = "../data/ap.dat";
  Corpus c;
  c.LoadData(data);
  VInt len;
  c.ULen(&len);
  c.RandomOrder();
  VInt len2;
  c.ULen(&len2);
  VInt len3;
  c.ULen(&len3);
  EXPECT_GT(DiffNum(len, len2), c.Len() - 100);
  EXPECT_EQ(DiffNum(len3, len2), 0);
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
