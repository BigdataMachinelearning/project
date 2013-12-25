// Copyright 2013 yuanwujun. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/rbm/rbm_repsoftmax.h"

#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"

namespace ml {
void RBM_RepSoftMax::Init(int f, int k, double momentum_, double eta_) {
  ml::RandomInit(f, k, &w);
  ml::RandomInit(k, &b);
  ml::RandomInit(f, &c);
  momentum = momentum_;
  eta = eta_;
}

void Update(const VInt &index, const VReal &h1, const VInt &v1,
            const VReal &h2, const VInt &v2, RBM_RepSoftMax* rbm) {
  for (size_t m = 0; m < index.size(); m++) {
    for (size_t f = 0; f < h1.size(); f++) {
      rbm->w[f][index[m]] += rbm->eta * (h1[f] * v1[index[m]] -
                                      h2[f] * v2[index[m]]) / index.size();
    }
  }
  /*
  for (VInt::size_type i = 0; i < h1.size(); i++) {
    rbm->c[i] += rbm->eta * (h1.at(i) - h2.at(i));
  }
  */
}

void ExpectV(const Document &doc, const VReal &h, const RBM_RepSoftMax &rbm,
                                                        VReal* v) {
  VReal arr(v->size());
  for (size_t k = 0; k < doc.words.size(); ++k) {
    for (size_t f = 0; f < h.size(); ++f) {
      arr[k] += rbm.w[f][doc.words[k]] * h[f] * doc.counts[k];
    }
  }
  ml::Softmax(arr, v);
}

void SampleV(const Document &doc, const VReal &h, const RBM_RepSoftMax &rbm,
                                                        VInt* v) {
  VReal expect(doc.words.size());
  ExpectV(doc, h, rbm, &expect);
  for (int i = 0; i < doc.total; ++i) {
    v->at(doc.words[Random(expect)])++;
  }
}

void ExpectH(const VInt &words, const VInt &counts, const RBM_RepSoftMax &rbm,
                                                    VReal* h) {
  h->resize(rbm.c.size());
  for (size_t f = 0; f < h->size(); f++) {
    double sum = 0.0;
    for (size_t w = 0; w < words.size(); ++w) {
      sum += rbm.w[f][words[w]] * counts[w];
    }
    sum += rbm.c[f];
    h->at(f) = Sigmoid(sum);
  }
}

void ExpectH(const Document &doc, const RBM_RepSoftMax &rbm, VReal* h) {
  ExpectH(doc.words, doc.counts, rbm, h);
}

void SampleH(const Document &doc,const RBM_RepSoftMax &rbm, VReal* h) {
  ExpectH(doc, rbm, h);	
  for (size_t i = 0; i < h->size(); i++) {
    (*h)[i] = Sample1((*h)[i]);
  }
}

void RBMLearning(const Corpus &corpus, int itern, RBM_RepSoftMax* rbm) {
  for(int iteration = 0; iteration < itern; ++iteration) {
    for(int i = 0; i < corpus.Len(); ++i) {
      VReal h1;
      SampleH(corpus.docs[i], *rbm, &h1);
      VInt v2(corpus.num_terms);
      SampleV(corpus.docs[i], h1, *rbm, &v2);
      VReal h2;
      ExpectH(corpus.docs[i].words, v2, *rbm, &h2);
      LOG_IF(INFO, i == 0) << i << ":" << Join(h2, " ");
      LOG_IF(INFO, i == 42) << i << ":" << Join(h2, " ");
      LOG_IF(INFO, i == 43) << i << ":" << Join(h2, " ");
      double var = Var(rbm->w);
      Update(corpus.docs[i].words, h1, corpus.docs[i].counts, h2, v2, rbm);
      double var2 = Var(rbm->w);
    }
  }
}
} // namespace ml
