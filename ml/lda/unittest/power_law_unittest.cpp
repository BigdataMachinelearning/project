// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/document.h"
#include "ml/lda/power_law.h"
#include "gtest/gtest.h"

namespace ml {
TEST(PowerTest, PowerTest) {
  Str path = "../../data/ap.dat";
  Corpus c;
  c.LoadData(path);
  VVReal theta;
  VReal alpha;
  int k = 10;
  EM(c, k, &theta, &alpha);
  WriteStrToFile(Join(theta, " ", "\n"), "theta");
  WriteStrToFile(Join(alpha, " "), "alpha");
}
} // namespace topic

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
