// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/document.h"
#include "gtest/gtest.h"

namespace ml {
TEST(DocumentTest, ReadDataTest) {
  Str data = "../data/ap.dat";
  Corpus c;
  c.LoadData(data);
  EXPECT_EQ(10473,  c.num_terms);
  EXPECT_EQ(2246,  c.docs.size());
  EXPECT_EQ(186,  c.docs[0].Len());
  EXPECT_EQ(263,  c.docs[0].total);
  EXPECT_EQ(6144,  c.docs[0].words[1]);
  EXPECT_EQ(1,  c.docs[0].counts[1]);
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
