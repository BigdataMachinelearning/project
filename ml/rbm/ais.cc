// Copyright 2013 yuanwujun, lijiankou. All Rights Reserved.
// Author: real.yuanwj@gmail.com lijk_start@163.com
#include "ml/rbm/ais.h"

#include "ml/rbm/repsoftmax.h"
#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"

namespace ml {
double MinusFreeEnergy(int len, const VInt &v, double beta, const RepSoftMax &rbm) {
  double result = 0;
  for (size_t f = 0; f < rbm.c.size(); ++f) {
    double sum = 0.0;
    for (size_t k = 0; k < rbm.b.size(); ++k) {
      sum += rbm.w[f][k] * v[k];
    }
    sum += rbm.c[f];
    result += log(1 + exp(sum * beta));
  }
  return result;
}

void Multiply(const RepSoftMax &src, double beta, RepSoftMax* des) {
  des->Init(src.w.size(), src.w[0].size(), src.bach_size, src.momentum,
                                           src.eta);
  ::Multiply(src.b, beta, &(des->b));
  ::Multiply(src.c, beta, &(des->c));
  ::Multiply(src.w, beta, &(des->w));
}

void UniformSample(const Document &doc, VInt* v) {
  VInt tmp(doc.ULen());
  ::UniformSample(doc.TLen(), &tmp);
  for (size_t i = 0; i < doc.ULen(); i++) {
    v->at(doc.words[i]) = tmp[i];
  }
}

double WAis(const Document &doc, int runs, const VReal &beta,
                                                const RepSoftMax &rbm) {
  double sum = 0.0;
  for(int k = 0; k < runs; ++k) {
    VInt v1(rbm.b.size());
    UniformSample(doc, &v1);// uninform shoud in the document
    double wais = 1;
    for(size_t i = 0; i < beta.size() - 1; ++i) {
      RepSoftMax tmp;
      Multiply(rbm, beta[i], &tmp);
      VReal h;
      SampleH(doc.ULen(), doc.words, v1, tmp, &h);
      VInt v2(v1.size());
      SampleV(doc, h, tmp, &v2);
      wais *= exp(MinusFreeEnergy(doc.ULen(), v2, beta[i + 1], rbm)) 
          / exp(MinusFreeEnergy(doc.ULen(), v2, beta[i], rbm));
      v1.swap(v2);
    }
    sum += wais;
    LOG_IF(INFO, k % 1000 ==0) << k << " " << Join(v1, " ");
  }
  return sum / runs;
}

double Likelihood(const Document &doc, int runs, const VReal &beta,
                                                 const RepSoftMax &rbm) {
  LOG(INFO)  << rbm.c.size();
  double z = WAis(doc, runs, beta, rbm) * pow(2, rbm.c.size()); 
  LOG(INFO) << z <<  " " << pow(2, rbm.c.size());
  LOG(INFO) << log(z);
  double p = MinusFreeEnergy(doc.ULen(), doc.counts, 1, rbm);
  return p / z;
}

double LogPartition(int doc_len, int word_num, const RepSoftMax &rep) {
  VInt v(word_num);
  v[0] = doc_len;
  VInt h(rep.c.size(), 0);
  VReal m_energy;
  do {
    do {
      double sum = 0;
      sum += Quadratic(h, v, rep.w);
      sum += doc_len * InnerProd(h, rep.c);
      sum += InnerProd(v, rep.b);
      m_energy.push_back(sum);
    } while (NextBinarySeq(&h));
  } while (NextMultiSeq(&v));
  return ::LogPartition(m_energy);
}
} // namespace ml
