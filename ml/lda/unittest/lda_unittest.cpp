// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/lda/lda.h"
#include "ml/lda/lda_var_em.h"
#include "gtest/gtest.h"

namespace ml {
TEST(LDATest, NewArrayTest) {
  double **a = NewArray(2, 3);
  Init(2, 3, 1.0, a);
  EXPECT_EQ("1 1 1 \n1 1 1 \n", Join(a, 2, 3));
}
  
TEST(LDATest, LikelihoodTest) {
  const int num_topics = 2;
  const int num_terms = 4;
  LdaModel m;
  NewLdaModel(num_topics, num_terms, &m);
  Init(num_topics, num_terms, 0.5, m.log_prob_w);
  Document doc;
  int len = 2;
  doc.total = 4;
  doc.words.resize(len);
  doc.counts.resize(len);
  for (int i = 0; i < doc.Len(); i++) {
    doc.words[i] = i; 
    doc.counts[i] = 2; 
  }
  VReal gamma;
  double value = 0.5;
  Init(num_topics, value, &gamma);
  VVReal phi;
  Init(num_topics, num_terms, value, &phi);
  LDA lda;
  lda.AddDoc(doc);
  EXPECT_LT(std::abs(-1.01415 - lda.Likelihood(0, m, gamma, phi)), 0.0001);
  VReal gamma3;
  double value3 = 0.8;
  Init(num_topics, value3, &gamma3);
  VVReal phi3;
  Init(num_topics, num_terms, value3, &phi3);
  EXPECT_LT(std::abs(-2.37435 - lda.Likelihood(0, m, gamma3, phi3)), 0.0001);
}
 
TEST(LDATest, VAREMTest) {
  long t1;
  (void) time(&t1);
  seedMT(t1);
  float em_converged = 1e-4;
  int em_max_iter = 30;
  int em_estimate_alpha = 1;
  int var_max_iter = 20;
  double var_converged = 1e-2;
  double initial_alpha = 0.1;
  int n_topic = 10;
  LDA lda;
  lda.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,
                         var_converged, initial_alpha, n_topic);
  Str data = "../../data/ap.dat";
  lda.LoadCorpus(data);
  Str result = "result/result";
  Str type = "seeded";
  LdaModel m;
  lda.RunEM(type, &m);
  LOG(INFO) << m.alpha;
  VVReal gamma;
  VVVReal phi;
  lda.Infer(m, &gamma, &phi);
  WriteStrToFile(Join(gamma, " ", "\n"), "gamma");
  WriteStrToFile(Join(phi, " ", "\n", "\n\n"), "phi");
}

TEST(LDATest, GibbsTest) {
  long t1;
  (void) time(&t1);
  seedMT(t1);
  float em_converged = 1e-4;
  int em_max_iter = 30;
  int em_estimate_alpha = 1;
  int var_max_iter = 20;
  double var_converged = 1e-2;
  double initial_alpha = 0.1;
  int n_topic = 10;
  LDA lda;
  lda.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,
                         var_converged, initial_alpha, n_topic);
  Str data = "../../data/ap.dat";
  lda.LoadCorpus(data);
  lda.Gibbs();
}
} // namespace ml

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
