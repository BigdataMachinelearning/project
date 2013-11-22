// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda/lda.h"

#include "base/base_head.h"
#include <cstdio>
#include <cstdlib>

void ReadFileToCorpus(const char* filename, Corpus* c) {
  FILE *fileptr = fopen(filename, "r");
  int nd = 0;
  int nw = 0;
  int length;
  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    c->docs = (Document*) realloc(c->docs, sizeof(document)*(nd+1));
    c->docs[nd].length = length;
    c->docs[nd].total = 0;
    c->docs[nd].words = new int[sizeof(int)*length];
    c->docs[nd].counts = new int[sizeof(int)*length];
    int count;
    int word;
    for (int n = 0; n < length; n++) {
      fscanf(fileptr, "%10d:%10d", &word, &count);
      word = word - OFFSET;
      c->docs[nd].words[n] = word;
      c->docs[nd].counts[n] = count;
      c->docs[nd].total += count;
      if (word >= nw) {
        nw = word + 1;
      }
    }
    nd++;
  }
  fclose(fileptr);
  c->num_docs = nd;
  c->num_terms = nw;
  printf("number of docs    : %d\n", nd);
  printf("number of terms   : %d\n", nw);
}

int MaxCorpusLen(const Corpus &c) {
  int max_len = 0;
  for (int i = 0; i < c.num_docs; i++) {
    if (c.docs[i].length > max_len) {
      max_len = c.docs[i].length;
    }
  }
  return max_len;
}
