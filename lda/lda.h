// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_LDA_H
#define LDA_LDA_H
#include "base/base_head.h"
namespace topic {
struct Document {
  int* words;
  int* counts;
  // VInt words;
  // VInt counts;
  int length;
  int total;
  Document () : total(0) {}
};

struct Corpus {
  std::vector<Document> docs;
  int num_terms;
  Corpus () : num_terms(0) {}
};

// alpha, beta : hpyerparameter; 
// log_prob_w : topic-word distribution
// theta : document-topic distribution
struct LdaModel {
  double alpha;
  double beta;
  double** log_prob_w;
  VVReal theta;
  VVReal phi;
  int num_topics;
  int num_terms;
  LdaModel () : alpha(0), log_prob_w(NULL), num_topics(0), num_terms(0) { }
};

struct LdaSuffStats {
  VVInt ss_phi;
  VVInt ss_theta; 
  VInt sum_ss_phi;
  VInt sum_ss_theta; 
  double** class_word;
  double* class_total;
  double alpha_suffstats;
  int num_docs;
  LdaSuffStats () : class_word(NULL), class_total(NULL),
                    alpha_suffstats(0), num_docs(0) { }
};

void ReadFileToCorpus(const char* filename, Corpus* corpus);
int MaxCorpusLen(const Corpus &c);
} // namespace topic
#endif // LDA_LDA_H
