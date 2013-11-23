// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda/lda.h"

#include "base/base_head.h"
#include <cstdio>
#include <cstdlib>

namespace topic {
const int  OFFSET = 0;                  // offset for reading data
void Corpus::LoadData(const Str &filename) {
  FILE *fileptr = fopen(filename.c_str(), "r");
  int nw = 0;
  int length;
  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    Document doc;
    // doc.length = length;
    doc.total = 0;
    // doc.words = new int[sizeof(int)*length];
    doc.words.resize(length);
    //doc.counts = new int[sizeof(int)*length];
    doc.counts.resize(length);
    int count;
    int word;
    for (int n = 0; n < length; n++) {
      if (fscanf(fileptr, "%10d:%10d", &word, &count) == 0) {
        continue;
      }
      word = word - OFFSET;
      doc.words[n] = word;
      doc.counts[n] = count;
      doc.total += count;
      if (word >= nw) {
        nw = word + 1;
      }
    }
    docs.push_back(doc);
  }
  fclose(fileptr);
  num_terms = nw;
  printf("number of docs    : %d\n", docs.size());
  printf("number of terms   : %d\n", nw);
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
} // namespace topic
