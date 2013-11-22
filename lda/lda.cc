// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda/lda.h"

#include "base/base_head.h"
#include <cstdio>
#include <cstdlib>

namespace topic {
const int  OFFSET = 0;                  // offset for reading data
void ReadFileToCorpus(const char* filename, Corpus* c) {
  FILE *fileptr = fopen(filename, "r");
  int nw = 0;
  int length;
  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    Document doc;
    doc.length = length;
    doc.total = 0;
    doc.words = new int[sizeof(int)*length];
    doc.counts = new int[sizeof(int)*length];
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
    c->docs.push_back(doc);
  }
  fclose(fileptr);
  c->num_terms = nw;
  printf("number of docs    : %d\n", c->docs.size());
  printf("number of terms   : %d\n", nw);
}

int MaxCorpusLen(const Corpus &c) {
  int max_len = 0;
  for (size_t i = 0; i < c.docs.size(); i++) {
    if (c.docs[i].length > max_len) {
      max_len = c.docs[i].length;
    }
  }
  return max_len;
}
} // namespace topic
