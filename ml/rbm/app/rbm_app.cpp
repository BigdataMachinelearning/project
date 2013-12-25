// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
#include "ml/rbm/rbm2.h"
#include "ml/rbm/rbm_repsoftmax.h"
#include "ml/util.h"
#include "ml/document.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>

DEFINE_double(eta, 0.1, "learning rate");

DEFINE_int32(bach_size, 100, "bach size");
DEFINE_int32(k, 5, "class size");
DEFINE_int32(m, 2000, "visual size");
DEFINE_int32(hidden, 100, "hidden feature size");
DEFINE_int32(it_num, 1000, "iter number");

DEFINE_string(type, "", "");
DEFINE_string(train_path, "", "");
DEFINE_string(test_path, "", "");

namespace ml {
void InitMovieLen(ml2::RBM* rbm) {
  double momentum = 0.0;
  double eta = FLAGS_eta;
  rbm->bach_size = FLAGS_bach_size;
  rbm->Init(FLAGS_hidden, FLAGS_m, FLAGS_k, momentum, eta);
}

void App() {
  if (FLAGS_type != "stl") {
    return;
  }
  User train;
  User test;
  LoadMovieLen(FLAGS_train_path, &train);
  LoadMovieLen(FLAGS_test_path, &test);
  LOG(INFO) << MaxItemId(train);
  ml2::RBM rbm;
  InitMovieLen(&rbm);
  RBMLearning(train, test, FLAGS_it_num, rbm.bach_size, &rbm);
  // RBMTest(train, test, rbm);
}

void App2() {
  if (FLAGS_type != "eigen") {
    return;
  }
  SpMat u_v;
  ReadData(FLAGS_train_path, 0, 0, &u_v);
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  ReadData(FLAGS_test_path, u_v.rows(), u_v.cols(), &test_u_v);
  SpMat test_v_u = test_u_v.transpose();
  int M = u_v.cols();
  int N = u_v.rows();
  RBM rbm(u_v, N, FLAGS_hidden, FLAGS_k);
  rbm.Train(u_v, test_u_v, 2000, FLAGS_eta, FLAGS_bach_size);
}

void App3() {
  Corpus corpus;
  VVInt hidden;
  RBM_RepSoftMax rbm;
  Str dat = "../../data/ap.dat";
  corpus.LoadData(dat);
  rbm.Init(FLAGS_k, corpus.num_terms, 1, 0.000000001);
  RBMLearning(corpus, 100, &rbm);
}
} // namespace ml

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  // ml::App3();
  ml::App();
  ml::App2();
  return  0;
}
