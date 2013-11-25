// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_LDA_H
#define LDA_LDA_H
#include "base/base_head.h"
namespace topic {
struct Document {
  VInt words;
  VInt counts;
  int total;

  Document () : total(0) {}
  int Len() const { return static_cast<int>(words.size());}
};
typedef const Document DocumentC;

struct Corpus {
  std::vector<Document> docs;
  int num_terms;  //max index of words
  Corpus () : num_terms(0) {}
  int Len() const { return static_cast<int>(docs.size());}
  int DocLen(int d) const { return static_cast<int>(docs[d].Len());}
  int Word(int d, int n) const { return docs[d].words[n];}
  int Count(int d, int n) const { return docs[d].counts[n];}
  void LoadData(const Str &filename);
  int MaxCorpusLen() const;
};
typedef const Corpus CorpusC;

struct LdaModel {
  double alpha;  // hyperparameter
  double beta;   // hyperparameter
  double** log_prob_w; // topic-word distribution
  VVReal theta;  // document-topic distribution
  VVReal phi;
  int num_topics;
  int num_terms;

  LdaModel () : alpha(0), log_prob_w(NULL), num_topics(0), num_terms(0) { }
};
typedef const LdaModel LdaModelC;

struct LdaSuffStats {
  VVInt phi;
  VVInt theta; 
  VInt sum_phi;
  VInt sum_theta; 
  double** class_word;
  double* class_total;
  double alpha_suffstats;
  int num_docs;
  LdaSuffStats () : class_word(NULL), class_total(NULL),
                    alpha_suffstats(0), num_docs(0) { }
  void Init(int m, int k, int v);
};
typedef const LdaSuffStats LdaSuffStatsC;
} // namespace topic
#endif // LDA_LDA_H
