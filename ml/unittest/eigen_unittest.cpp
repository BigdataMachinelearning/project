// copyright 2013 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/util.h"
#include "ml/eigen.h"
#include "gtest/gtest.h"

namespace ml {
TEST(Eigen, DemoTest) {
  Eigen::Vector4d v1;  
  v1<< 1,2,3,4;  
  LOG(INFO) <<v1;
}

TEST(Eigen, ReadDataTest) {
  Str path("rbm/tmp/train_g20.txt");
  TripleVec vec;
  ReadData(path, &vec);
  EXPECT_EQ(79951, vec.size());
  std::pair<int, int> p = Max(vec);
  EXPECT_EQ(1682, p.first);
  EXPECT_EQ(943, p.second);
  SpMat mat;
  ReadData(path, &mat);
  EXPECT_EQ(943, mat.cols());
  // LOG(INFO) << mat.cwiseMax();
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
