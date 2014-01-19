// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#include "ml/eigen.h"

#include "ml/util.h"

void Sample(EVec *h) {
  for (int i = 0; i < h->size(); ++i) {
    (*h)[i] = Sample1((*h)[i]);
  }
}

void ReadData(const Str &path, TripleVec* vec) {
  FILE *fin = fopen(path.c_str(), "r");
  int u;
  int v;
  float r;
  while(fscanf(fin, "%d %d %f", &u, &v, &r) > 0) {
    vec->push_back(Triple(v, u, r));
  }
}

std::pair<int, int> Max(const TripleVec &vec) {
  size_t col = 0;
  size_t row = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    col = col > vec[i].col() ? col : vec[i].col();
    row = row > vec[i].row() ? row : vec[i].row();
  }
  return std::make_pair(row + 1, col + 1);
}

std::pair<int, int> ReadData(const Str &path, SpMat *mat) {
  TripleVec vec;
  ReadData(path, &vec);
  std::pair<int, int> p;
  if (mat->cols() == 0) {
    p = Max(vec);
    mat->resize(p.first, p.second);
  }
  mat->setFromTriplets(vec.begin(), vec.end());
  return p;
}

void NormalRandom(EVec* des) {
  for (int i = 0; i < des->size(); i++) {
    (*des)[i] = ml::NormalSample() / 100;
  }
}

void NormalRandom(EMat *mat) {
  for (int i = 0; i < mat->rows(); i++) {
    for (int j = 0; j < mat->cols(); j++) {
      (*mat)(i, j) = ml::NormalSample() / 100;
    }
  }
}
