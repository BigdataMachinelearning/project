// Copyright 2013 yuanwujun, lijiankou. All Rights Reserved.
// Author: real.yuanwj@gmail.com lijk_start@163.com
#include "ml/rbm/repsoftmax.h"

#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/info.h"
#include "ml/util.h"

namespace ml {
void RepSoftMax::Init(int f, int k, int bach, double momentum_, double eta_) {
  ml::RandomInit(f, k, &w);
  ml::RandomInit(k, &b);
  ml::RandomInit(f, &c);
  bach_size = bach;
  InitZero();
  momentum = momentum_;
  eta = eta_;
}

void RepSoftMax::InitZero() {
  dw.clear();
  db.clear();
  dc.clear();
  ::Init(w.size(), w[0].size(), 0.0, &dw);
  ::Init(b.size(), 0.0, &db);
  ::Init(c.size(), 0.0, &dc);
}

void Gradient(const VInt &index, const VReal &h1, const VInt &v1,
              const VReal &h2, const VInt &v2, RepSoftMax* rep) {
  for (size_t f = 0; f < h1.size(); ++f) {
    rep->dc[f] += index.size() * (h1[f] -  h2[f]);
    for (size_t m = 0; m < index.size(); ++m) {
      rep->dw[f][index[m]] += v1[m] * h1[f] - v2[m] * h2[f];
    }
  }
  for (size_t m = 0; m < index.size(); ++m) {
    rep->db[index[m]] += v1[m] - v2[m];
  }
}

void Update(RepSoftMax* rep) {
  double eta = rep->eta / rep->bach_size;
  for (size_t f = 0; f < rep->w.size(); f++) {
    rep->c[f] += eta * rep->dc[f];
    for (size_t m = 0; m < rep->w[0].size(); m++) {
      rep->w[f][m] += eta * rep->dw[f][m];
    }
  }
  for (size_t m = 0; m < rep->b.size(); ++m) {
    rep->b[m] += eta * rep->db[m];
  }
}

void ExpectV(const Document &doc, const VReal &h, const RepSoftMax &rep,
                                                        VReal* v) {
  VReal arr(v->size());
  for (size_t k = 0; k < doc.words.size(); ++k) {
    arr[k] = 0;
    for (size_t f = 0; f < h.size(); ++f) {
      arr[k] += rep.w[f][doc.words[k]] * h[f];
      // * doc.counts[k];
    }
    arr[k] += rep.b[doc.words[k]];
     // * doc.counts[k];
  }
  ml::Softmax(arr, v);
}

void SampleV(const Document &doc, const VReal &h, const RepSoftMax &rbm,
                                                        VInt* v) {
  VReal expect(doc.words.size());
  ExpectV(doc, h, rbm, &expect);
  for (int i = 0; i < doc.total; ++i) {
    v->at(Random(expect))++;
  }
}

void ExpectH(const VInt &words, const VInt &counts, const RepSoftMax &rbm,
                                                    VReal* h) {
  h->resize(rbm.c.size());
  for (size_t f = 0; f < h->size(); f++) {
    double sum = 0.0;
    for (size_t w = 0; w < words.size(); ++w) {
      sum += rbm.w[f][words[w]] * counts[w];
    }
    sum += words.size() * rbm.c[f];
    h->at(f) = Sigmoid(sum);
  }
}

void ExpectH(const Document &doc, const RepSoftMax &rbm, VReal* h) {
  ExpectH(doc.words, doc.counts, rbm, h);
}

void SampleH(const Document &doc,const RepSoftMax &rbm, VReal* h) {
  ExpectH(doc, rbm, h);	
  for (size_t i = 0; i < h->size(); i++) {
    (*h)[i] = Sample1((*h)[i]);
  }
}

void RBMLearning(const Corpus &corpus, int itern, RepSoftMax* rbm) {
  int count = 0;
  VVReal result;
  result.reserve(corpus.Len());
  for(int iteration = 0; iteration < itern; ++iteration) {
    result.clear();
    for(int i = 0; i < corpus.Len(); ++i) {
      count++;
      VReal h1;
      SampleH(corpus.docs[i], *rbm, &h1);
      VInt v2(corpus.docs[i].words.size());
      SampleV(corpus.docs[i], h1, *rbm, &v2);
      VReal h2;
      ExpectH(corpus.docs[i].words, v2, *rbm, &h2);
      LOG_IF(INFO, i == 0) << Join(h2, " ");
      LOG_IF(INFO, i == 43) << Join(h2, " ");
      result.push_back(h2);
      Gradient(corpus.docs[i].words, h1, corpus.docs[i].counts, h2, v2, rbm);
      if (count == rbm->bach_size) {
        count = 0;
        Update(rbm);
        rbm->InitZero();
      }
      VReal tmp;
      ::Sum(rbm->w, &tmp);
      // LOG_IF(INFO, i == 0) << Join(tmp, " ") << ":var--" << Var(rbm->w);
    }
    WriteStrToFile(Join(result, " ", "\n"), "result.txt");
  }
}

/*
void CD(const Document &doc, const RepSoftMax &rbm, int step, VInt* v) {
  VReal h1;
  SampleH(doc, rbm, &h1);
  SampleV(doc, h1, rbm, v);
  VReal h2;
  VInt r(v->size());
  for (int i = 1; i < step; i++) {
    h2.clear();
    ExpectH(doc.words, *v, rbm, &h2);
    SampleV(doc, h2, rbm, v);
    Add(*v, &r);
  }
  LOG(INFO) << CrossEntropy(r, doc.counts);
}
*/
} // namespace ml
