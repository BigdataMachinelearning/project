// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "lda/lda_estimate.h"
#include "lda/lda.h"
#include "gtest/gtest.h"

namespace topic {
TEST(LDATest, ReadDataTest) {
  Str data = "../data/ap.dat";
  Corpus c;
  ReadFileToCorpus(data.c_str(), &c);
  EXPECT_EQ(10473,  c.num_terms);
  EXPECT_EQ(2246,  c.num_docs);
  EXPECT_EQ(186,  c.docs[0].length);
  EXPECT_EQ(263,  c.docs[0].total);
  EXPECT_EQ(6144,  c.docs[0].words[1]);
  EXPECT_EQ(1,  c.docs[0].counts[1]);
}
 
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
  doc.length = 2;
  doc.total = 4;
  doc.words = new int[doc.length];
  doc.counts = new int[doc.length];
  for (int i = 0; i < doc.length; i++) {
    doc.words[i] = i; 
    doc.counts[i] = 2; 
  }
  VReal gamma;
  double value = 0.5;
  Init(num_topics, value, &gamma);
  VVReal phi;
  Init(num_topics, num_terms, value, &phi);
  LDA lda;
  EXPECT_LT(std::abs(-1.01415 - lda.Likelihood(doc, m, phi, gamma)), 0.0001);
  VReal gamma3;
  double value3 = 0.8;
  Init(num_topics, value3, &gamma3);
  VVReal phi3;
  Init(num_topics, num_terms, value3, &phi3);
  EXPECT_LT(std::abs(-2.37435 - lda.Likelihood(doc, m, phi3, gamma3)), 0.0001);
}

TEST(LDATest, LDATest) {
  std::cout << "aa";
  long t1;
  (void) time(&t1);
  seedMT(t1);
  float em_converged = 1e-4;
  int em_max_iter = 100;
  int em_estimate_alpha = 1;
  int var_max_iter = 20;
  double var_converged = 1e-2;
  double initial_alpha = 0.1;
  int n_topic = 10;
  LDA lda;
  lda.Init(em_converged, em_max_iter, em_estimate_alpha, var_max_iter,
                         var_converged, initial_alpha, n_topic);
  Str data = "../data/ap.dat";
  Corpus c;
  ReadFileToCorpus(data.c_str(), &c);
  Str result = "result/result";
  Str mode = "seeded";
  double** gamma = NewArray(c.num_docs, n_topic);
  double** phi = NewArray(MaxCorpusLen(c), n_topic);
  lda.RunEM(mode, c, gamma, phi);
  WriteStrToFile(Join(gamma, c.num_docs, n_topic), "gamma");
}
 
} // namespace topic

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
