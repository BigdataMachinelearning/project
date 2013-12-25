// copyright 2013 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/info.h"
#include "gtest/gtest.h"

namespace ml {
TEST(Info, NormalTest) {
  VReal v;
  v.push_back(2);
  v.push_back(3);
  v.push_back(5);
  VReal v2(v.size());
  Normal(v, &v2);
  EXPECT_DOUBLE_EQ(0.2, v2[0]);
  EXPECT_DOUBLE_EQ(0.3, v2[1]);
  EXPECT_DOUBLE_EQ(0.5, v2[2]);
}

TEST(Info, CrossEntropyTest) {
  VReal v;
  v.push_back(1);
  v.push_back(1);
  VReal v2;
  v2.push_back(2);
  v2.push_back(2);
  EXPECT_DOUBLE_EQ(1, CrossEntropy(v, v2));
  VReal v3;
  v3.push_back(1);
  v3.push_back(9);
  double r = Log2(0.9) + Log2(0.1);
  EXPECT_DOUBLE_EQ(-r / 2, CrossEntropy(v, v3));
 
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
