// Copyright 2013 yuanwujun, All Rights Reserved.
// Author: real.yuanwj@gmail.com
#include "ml/rbm/ais.h"

#include "ml/rbm/repsoftmax.h"
#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"

namespace ml {
double CalculateP(int len, const VInt &v, double beta, const RepSoftMax &rbm) {
  double result = 0;
  for (size_t f = 0; f < rbm.c.size(); ++f) {
    double sum = 0.0;
    for (size_t k = 0; k < rbm.b.size(); ++k) {
      sum += rbm.w[f][k] * v[k];
    }
    sum += len * rbm.c[f];
    result += sum * beta;
  }
  for(size_t k = 0; k < rbm.b.size(); ++k) {
    result += rbm.b[k] * v[k] * beta;
  }
  return result;
}

void UniformSample(const Document &doc, VInt* v) {
  VInt tmp(doc.Len());
  ::UniformSample(doc.TotalLen(), &tmp);
  for (int i = 0; i < doc.Len(); i++) {
    v->at(doc.words[i]) = tmp[i];
  }
}

void Multiply(const RepSoftMax &src, double beta, RepSoftMax* des) {
  des->Init(src.w.size(), src.w[0].size(), src.bach_size, src.momentum,
                                           src.eta);
  for (size_t i = 0; i < src.b.size(); i++) {
    des->b[i] = src.b[i] * beta;
  }
  for (size_t i = 0; i < src.c.size(); i++) {
    des->c[i] = src.c[i] * beta;
  }
  for (size_t i = 0; i < src.w.size(); i++) {
    for (size_t j = 0; j < src.w[0].size(); j++) {
      des->w[i][j] = src.w[i][j] * beta;
    }
  }
}

double Partition(const Document &doc, int runs, const VReal &beta,
                                                const RepSoftMax &rbm) {
  VReal wais;
  Init(runs, 1, &wais);
  for(int k = 0; k < runs; ++k) {
    VVInt v;
    Init(beta.size(), rbm.b.size(), 1, &v);
    UniformSample(doc, &v[0]);
    for(size_t i = 0; i < beta.size() - 1; ++i) {
      RepSoftMax tmp;
      Multiply(rbm, beta[i], &tmp);
      VReal h;
      SampleH(doc.TotalLen(), doc.words, v[i], tmp, &h);
      SampleV(doc, h, tmp, &v[i + 1]);
    }
    LOG(INFO) << Join(v, " ", "\n");
    for(size_t i = 1; i < beta.size(); ++i ) {
      wais[k] *= exp(CalculateP(doc.TotalLen(), v[i], beta[i], rbm)) 
          / exp(CalculateP(doc.TotalLen(), v[i], beta[i - 1], rbm));
    }
  }
  return std::accumulate(wais.begin(), wais.end(), 0.0) / runs;
}

double Probability(const Document &doc, int runs, const VReal &beta,
                                        const RepSoftMax &rbm) {
  double partition = Partition(doc, runs, beta, rbm); 
  LOG(INFO) << partition;
  double p = CalculateP(doc.TotalLen(), doc.counts, 1, rbm);
  LOG(INFO) << p;
  return p / partition;
}
} // namespace ml
