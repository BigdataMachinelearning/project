// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda/lda.h"

#include "base/base_head.h"
#include <cstdio>
#include <cstdlib>

namespace topic {
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

void LdaSuffStats::Init(int m, int k, int v) {
  phi.resize(k);
  sum_phi.resize(k);
  for (int i = 0; i < k; i++) {
    phi[i].resize(v);
  }
  theta.resize(m);
  sum_theta.resize(m);
  for (int i = 0; i < m; i++) {
    theta[i].resize(k);
  }
}
} // namespace topic
