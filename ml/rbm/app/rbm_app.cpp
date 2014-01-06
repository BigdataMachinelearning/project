// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/ais.h"
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
#include "ml/rbm/rbm2.h"
#include "ml/rbm/repsoftmax.h"
#include "ml/util.h"
#include "ml/document.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>

DEFINE_double(eta, 0.1, "learning rate");
DEFINE_int32(beta, 50, "beta");
DEFINE_int32(ais_run, 2000, "ais sample time");

DEFINE_int32(bach_size, 100, "bach size");
DEFINE_int32(k, 5, "class size");
DEFINE_int32(m, 2000, "visual size");
DEFINE_int32(hidden, 100, "hidden feature size");
DEFINE_int32(it_num, 1000, "iter number");
DEFINE_int32(algorithm_type, 1, "iter number");

DEFINE_string(type, "softmax", "");
DEFINE_string(train_path, "", "");
DEFINE_string(test_path, "", "");

namespace ml {
void InitMovieLen(ml2::RBM* rbm) {
  double momentum = 0.0;
  double eta = FLAGS_eta;
  rbm->Init(FLAGS_hidden, FLAGS_m, FLAGS_k, FLAGS_bach_size, momentum, eta);
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
  if (FLAGS_type != "softmax") {
    return;
  }
  Corpus corpus;
  corpus.LoadData(FLAGS_train_path);
  // corpus.RandomOrder();
  RepSoftMax softmax;
  softmax.Init(FLAGS_k, corpus.num_terms, FLAGS_bach_size, 1, FLAGS_eta);
  if (FLAGS_algorithm_type == 1) {
    RBMLearning(corpus, FLAGS_it_num, &softmax);
  } else {
    RBMLearning2(corpus, FLAGS_it_num, &softmax);
  }
  VReal beta;
  Range(0.01, 1, 0.01, &beta);
  LOG(INFO) << Probability(corpus.docs[0], FLAGS_ais_run, beta, softmax);

  RepSoftMax rep;
  EyeRep(1, 2, &rep);
  LOG(INFO) << LogPartition(2, 2, rep);
  LOG(INFO) << log(3 * (exp(4) + exp(7)));
}
} // namespace ml

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  ml::App3();
  ml::App();
  ml::App2();
  return  0;
}
