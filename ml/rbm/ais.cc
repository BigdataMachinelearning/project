// Copyright 2013 yuanwujun, lijiankou. All Rights Reserved.
// Author: real.yuanwj@gmail.com lijk_start@163.com
#include "ml/rbm/ais.h"

#include "ml/rbm/repsoftmax.h"
#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"

namespace ml {
void ExpectV(const Document &doc, const VReal &h, const RepSoftMax &rep,
                                  double beta, VReal* v) {
  VReal arr(v->size());
  for (size_t k = 0; k < doc.words.size(); ++k) {
    arr[k] = 0;
    for (size_t f = 0; f < h.size(); ++f) {
      arr[k] += rep.w[f][doc.words[k]] * h[f];
    }
    arr[k] += rep.b[doc.words[k]];
    arr[k] += (1 - beta) * rep.b.size();
  }
  ml::Softmax(arr, v);
}

void SampleV(const Document &doc, const VReal &h, const RepSoftMax &rbm,
                                                  double beta, VInt* v) {
  VReal expect(doc.words.size());
  ExpectV(doc, h, rbm, beta, &expect);
  for (size_t i = 0; i < doc.total; ++i) {
    v->at(Random(expect))++;
  }
}

double Potential(double len, const VInt &v, double beta, const RepSoftMax &rbm) {
  double result = 1;
  for (size_t f = 0; f < rbm.c.size(); ++f) {
    result *= (1 + exp(beta * (::InnerProd(rbm.w[f], v) + len * rbm.c[f])));
  }
  result *= exp((1 - beta) * len * log(1.0/ rbm.b.size()));
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
  VReal s;
  for(int k = 0; k < runs; ++k) {
    VInt v1(rbm.b.size());
    UniformSample(doc, &v1);// uniform shoud in the document
    double wais = 1;
    // for(size_t i = 0; i < beta.size() - 1; ++i) {
    for(size_t i = 0; i < 1; ++i) {
      RepSoftMax tmp;
      Multiply(rbm, beta[i], &tmp);
      VReal h;
      SampleH(doc.ULen(), doc.words, v1, tmp, &h);
      VInt v2(v1.size());
      SampleV(doc, h, tmp, beta[i], &v2);
      double a = Potential(doc.TLen(), v2, beta[i + 1], rbm) /
                 Potential(doc.TLen(), v2, beta[i], rbm);
      if (i == 0) {
        s.push_back(a);
      }
      wais *= a;
      v1.swap(v2);
    }
    sum += wais;
  }
  LOG(INFO) << Join(s, " ");
  return sum / runs;
}

double Likelihood(const Document &doc, int runs, const VReal &beta,
                                                 const RepSoftMax &rbm) {
  double wais = WAis(doc, runs, beta, rbm);
  double z = wais * pow(2, rbm.c.size()); 
  LOG(INFO) << z;
  double p = Potential(doc.TLen(), doc.counts, 1, rbm);
  return p / z;
}

void CreateMulti(int len, VVInt* v) {
  for (int i = 0; i < len; i++) {
    VInt tmp(len, 0);
    tmp[i] = 1;
    v->push_back(tmp);
  }
}

double LogPartition(int doc_len, int word_num, const RepSoftMax &rep) {
  VVInt v;
  CreateMulti(word_num, &v);
  VInt h(rep.c.size(), 0);
  VReal m_energy;
  for (size_t i = 0; i < v.size(); i++) {
    do {
      double sum = 0;
      sum += ::Quadratic(h, v[i], rep.w);
      sum += doc_len * ::InnerProd(h, rep.c);
      sum += InnerProd(v[i], rep.b);
      m_energy.push_back(sum);
    } while (NextBinarySeq(&h));
  }
  return ::LogPartition(m_energy);
}

double LogMultiPartition(int doc_len, int word_num, const RepSoftMax &rep) {
  VInt v(rep.b.size());
  v[0] = doc_len;
  VInt h(rep.c.size(), 0);
  VReal m_energy;
  VInt multi_num;
  do {
    do {
      multi_num.push_back(MultiNum(doc_len, v));
      double sum = 0;
      sum += Quadratic(h, v, rep.w);
      sum += doc_len * InnerProd(h, rep.c);
      sum += InnerProd(v, rep.b);
      sum += log(1.0 / rep.b.size()) * doc_len;
      m_energy.push_back(sum);
    } while (NextBinarySeq(&h));
  } while (NextMultiSeq(&v));
  return ::LogPartition(multi_num, m_energy);
}
} // namespace ml
