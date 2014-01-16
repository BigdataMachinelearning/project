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
DEFINE_double(beta_beg, 0.5, "beta");
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
  RBMLearning(train, test, FLAGS_it_num, &rbm);
  // RBMTest(train, test, rbm);
}

void App2() {
  if (FLAGS_type != "eigen") {
    return;
  }
  SpMat u_v;
  ReadData(FLAGS_train_path, 0, 0, &u_v);
  // SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  ReadData(FLAGS_test_path, u_v.rows(), u_v.cols(), &test_u_v);
  // SpMat test_v_u = test_u_v.transpose();
  RBM rbm(u_v, u_v.rows(), FLAGS_hidden, FLAGS_k);
  rbm.Train(u_v, test_u_v, 2000, FLAGS_eta, FLAGS_bach_size);
}

void App3() {
  if (FLAGS_type != "softmax") {
    return;
  }
  Corpus corpus;
  corpus.LoadData(FLAGS_train_path);
  // corpus.RandomOrder();
  /*
  RepSoftMax rep;
  rep.Init(FLAGS_k, corpus.num_terms, FLAGS_bach_size, 1, FLAGS_eta);
  if (FLAGS_algorithm_type == 1) {
    RBMLearning(corpus, FLAGS_it_num, &rep);
  } else {
    RBMLearning2(corpus, FLAGS_it_num, &rep);
  }
  */
  RepSoftMax rep;
  // rep.Init(FLAGS_k, corpus.num_terms, FLAGS_bach_size, 1, FLAGS_eta);
  int size_v = 2;
  // InitRep(FLAGS_k, size_v, 0.1, &rep);
  InitRep(FLAGS_k, size_v, 0.01, &rep);
  VReal beta;
  Range(0, 1, FLAGS_beta_beg, &beta);
  double l = Likelihood(corpus.docs[0], FLAGS_ais_run, beta, rep);
  RepSoftMax tmp;
  Multiply(rep, beta[1], &tmp);
  // double p = LogPartition(corpus.TLen(0), corpus.ULen(0), tmp);
  double beta_a = 1;
  double p = LogMultiPartition(corpus.TLen(0), corpus.ULen(0), beta_a, rep);
  LOG(INFO) << p << " real:" << exp(p);
}
} // namespace ml

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  ml::App3();
  ml::App();
  ml::App2();
  return  0;
}
