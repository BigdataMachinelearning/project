// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_DOCUMENT_H_
#define LDA_DOCUMENT_H_
#include <vector>
#include "base/base_head.h"
namespace ml {
struct Document {
  VInt words;
  VInt counts;
  int total;
  Document() : total(0) {}
  int Len() const { return static_cast<int>(words.size());}
};
typedef std::vector<Document> VDocument;
typedef const Document DocumentC;

struct Corpus {
  VDocument docs;
  int num_terms;  // max index of words
  Corpus() : num_terms(0) {}
  int Len() const { return static_cast<int>(docs.size());}
  int DocLen(int d) const { return static_cast<int>(docs[d].Len());}
  int Word(int d, int n) const { return docs[d].words[n];}
  int Count(int d, int n) const { return docs[d].counts[n];}
  void LoadData(const Str &filename);
  int MaxCorpusLen() const;
  void NewLatent(VVInt* z) const;
  void NewLatent(VVReal* z) const;
  void NewLatent(VVVReal* z, int k) const;
};
typedef const Corpus CorpusC;
}  // namespace ml 
#endif// LDA_DOCUMENT_H_
