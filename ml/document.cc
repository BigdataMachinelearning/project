// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/document.h"

#include <cstdio>
#include <cstdlib>
#include "base/base_head.h"

namespace ml {
void Corpus::LoadData(const Str &filename) {
  FILE *fileptr = fopen(filename.c_str(), "r");
  int length = 0;
  num_terms = 0;
  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    Document doc;
    doc.total = 0;
    doc.words.resize(length);
    doc.counts.resize(length);
    for (int n = 0; n < length; n++) {
      int count;
      int word;
      if (fscanf(fileptr, "%10d:%10d", &word, &count) == 0) {
        continue;
      }
      doc.words[n] = word;
      doc.counts[n] = count;
      doc.total += count;
      if (word >= num_terms) {
        num_terms = word + 1;
      }
    }
    docs.push_back(doc);
  }
  fclose(fileptr);
  printf("number of docs    : %d\n", docs.size());
  printf("number of terms   : %d\n", num_terms);
}

int Corpus::MaxCorpusLen() const {
  int max_len = 0;
  for (size_t i = 0; i < docs.size(); i++) {
    if (docs[i].Len() > max_len) {
      max_len = static_cast<int>(docs[i].words.size());
    }
  }
  return max_len;
}

void Corpus::NewLatent(VVInt* z) const {
  z->resize(this->Len());
  for (int i = 0; i < this->Len(); i++) {
    z->at(i).resize(this->DocLen(i));
  }
}

void Corpus::NewLatent(VVReal* z) const {
  z->resize(this->Len());
  for (int i = 0; i < this->Len(); i++) {
    z->at(i).resize(this->DocLen(i));
  }
}

void Corpus::NewLatent(VVVReal* z, int k) const {
  z->resize(this->Len());
  for (int i = 0; i < this->Len(); i++) {
    z->at(i).resize(this->DocLen(i));
    for (int j = 0; j < this->DocLen(i); j++) {
      z->at(i).at(j).resize(k);
    }
  }
}

//
// void ()
}  // namespace ml 
