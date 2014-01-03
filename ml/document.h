// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_DOCUMENT_H_
#define ML_DOCUMENT_H_
#include "base/base_head.h"
namespace ml {
struct Document {
  VInt words;
  VInt counts;
  int total;
  Document() : total(0) {}
  inline int Len() const { return static_cast<int>(words.size());}
  inline int TotalLen() const { return total;}
};

typedef std::vector<Document> VDocument;
typedef const Document DocumentC;

struct Corpus {
  VDocument docs;
  int num_terms;  // max index of words
  Corpus() : num_terms(0) {}
  int Len() const { return static_cast<int>(docs.size());}
  int TermNum() const { return num_terms;}
  int DocLen(int d) const { return static_cast<int>(docs[d].Len());}
  void DocLen(VInt* v) const;

  int Word(int d, int n) const { return docs[d].words[n];}
  int Count(int d, int n) const { return docs[d].counts[n];}

  void LoadData(const Str &filename);
  int MaxCorpusLen() const;
  void RandomOrder();

  void NewLatent(VVInt* z) const;
  void NewLatent(VVReal* z) const;
  void NewLatent(VVVReal* z, int k) const;
};

typedef const Corpus CorpusC;

void SplitData(const Corpus &c, double value, Corpus* train, Corpus* test);
}  // namespace ml 
#endif// ML_DOCUMENT_H_
