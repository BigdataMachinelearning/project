// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/matrix_factorization.h"
#include "ml/util.h"

DEFINE_double(lambda, 0.1, "regularization");
DEFINE_double(eta, 0.1, "learning rate");
DEFINE_double(k, 10, "topic num");
DEFINE_double(it_num, 10000, "topic num");
DEFINE_string(train_path, "tmp/train1", "");
DEFINE_string(test_path, "tmp/test1", "");

namespace ml {
void App() {
  SpMat train;
  SpMat test;
  LOG(INFO) << FLAGS_train_path;
  LOG(INFO) << FLAGS_test_path;
  LOG(INFO) << FLAGS_k;
  LOG(INFO) << FLAGS_eta;
  LOG(INFO) << FLAGS_lambda;
  std::pair<int, int> p = ReadData(FLAGS_train_path, &train);
  ReadData(FLAGS_test_path, &test);
  MF mf;
  RandomInit(p.second, p.first, FLAGS_k, &mf);
  SGD(FLAGS_it_num, FLAGS_eta, FLAGS_lambda, train, test, &mf);
}
} // namespace ml

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  ml::App();
  return  0;
}
