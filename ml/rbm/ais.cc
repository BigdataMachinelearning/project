// Copyright 2013 yuanwujun, All Rights Reserved.
// Author: real.yuanwj@gmail.com
#include "ml/rbm/ais.h"

#include "ml/rbm/repsoftmax.h"
#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"

namespace ml {
double MinusEnergy(int len, const VInt &v, double beta, const RepSoftMax &rbm) {
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
  ::Multiply(src.b, beta, &(des->b));
  ::Multiply(src.c, beta, &(des->c));
  ::Multiply(src.w, beta, &(des->w));
}

double Partition(const Document &doc, int runs, const VReal &beta,
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
      SampleH(doc.TotalLen(), doc.words, v1, tmp, &h);
      VInt v2(v1.size());
      SampleV(doc, h, tmp, &v2);
      wais *= exp(MinusEnergy(doc.TotalLen(), v2, beta[i + 1], rbm)) 
          / exp(MinusEnergy(doc.TotalLen(), v1, beta[i], rbm));
      v1.swap(v2);
    }
    sum += wais;
    LOG_IF(INFO, k % 1000 ==0) << k << " " << Join(v1, " ");
  }
  return sum / runs;
}

double Probability(const Document &doc, int runs, const VReal &beta,
                                        const RepSoftMax &rbm) {
  double partition = Partition(doc, runs, beta, rbm) * pow(2, rbm.c.size()); 
  LOG(INFO) << partition;
  double p = MinusEnergy(doc.TotalLen(), doc.counts, 1, rbm);
  LOG(INFO) << p;
  return p / partition;
}

double MinusEnergy(int len, const VInt &v, const VInt &h,
                       const RepSoftMax &rbm) {
  double result = 0;
  result += Quadratic(v, h, rbm.w);
  LOG(INFO) << result;
  result += len * InnerProd(h, rbm.c);
  LOG(INFO) << result;
  result += InnerProd(v, rbm.b);
  LOG(INFO) << result;
  return result;
}

void Subtract(double m, VReal* v) {
  for (size_t i = 0; i < v->size(); i++) {
    v->at(i) -= m;
  }
}

void Exp(VReal* v) {
  for (size_t i = 0; i < v->size(); i++) {
    v->at(i) = exp(v->at(i));
  }
}

double LogPartition(int doc_len, int word_num, const RepSoftMax &rep) {
  VInt v(word_num);
  v[0] = doc_len;
  VInt h(rep.c.size(), 0);
  VReal m_energy;
  do {
    do {
      LOG(INFO) << Join(h, " ")  << Join(v, " ");
      LOG(INFO) << Join(rep.w, " ", "\n");
      LOG(INFO) << Join(rep.b, " ");
      LOG(INFO) << Join(rep.c, " ");
      m_energy.push_back(MinusEnergy(doc_len, h, v, rep));
    } while (NextBinarySeq(&h));
  } while (NextMultiSeq(&v));
  LOG(INFO) << Join(m_energy, " ");
  double m = Max(m_energy);
  Subtract(m, &m_energy);
  Exp(&m_energy);
  return m * log(Sum(m_energy));
}
} // namespace ml
