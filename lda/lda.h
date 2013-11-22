// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_LDA_H
#define LDA_LDA_H
#define OFFSET 0;                  // offset for reading data
#include "base/base_head.h"
typedef struct Document {
  int* words;
  int* counts;
  int length;
  int total;
  Document () : words(NULL), counts(NULL), length(0), total(0) {}
} document;

typedef struct Corpus {
  document* docs;
  int num_terms;
  int num_docs;
  Corpus () : docs(NULL), num_terms(0), num_docs(0) { }
} corpus;

typedef struct {
  double alpha;
  double** log_prob_w;
  int num_topics;
  int num_terms;
} lda_model, LdaModel;

typedef struct {
  double** class_word;
  double* class_total;
  double alpha_suffstats;
  int num_docs;
} lda_suffstats, LdaSuffStats;

void ReadFileToCorpus(const char* filename, Corpus* corpus);
int MaxCorpusLen(const Corpus &c);
#endif // LDA_LDA_H
