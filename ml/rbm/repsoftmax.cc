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

void Gradient2(int len, const VReal &h1, const VInt &v1,
              const VReal &h2, const VInt &v2, RepSoftMax* rep) {
  for (size_t f = 0; f < h1.size(); ++f) {
    rep->dc[f] += len * (h1[f] -  h2[f]);
    for (size_t m = 0; m < v1.size(); ++m) {
      rep->dw[f][m] += v1[m] * h1[f] - v2[m] * h2[f];
    }
  }
  for (size_t m = 0; m < v1.size(); ++m) {
    rep->db[m] += v1[m] - v2[m];
  }
}

void Gradient(int len, const VInt &index, const VReal &h1, const VInt &v1,
              const VReal &h2, const VInt &v2, RepSoftMax* rep) {
  for (size_t f = 0; f < h1.size(); ++f) {
    rep->dc[f] += len * (h1[f] -  h2[f]);
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
    }
    arr[k] += rep.b[doc.words[k]];
  }
  ml::Softmax(arr, v);
}

void ExpectV2(const VReal &h, const RepSoftMax &rep, VReal* v) {
  VReal arr(rep.w[0].size());
  for (size_t k = 0; k < rep.w[0].size(); ++k) {
    arr[k] = 0;
    for (size_t f = 0; f < h.size(); ++f) {
      arr[k] += rep.w[f][k] * h[f];
    }
    arr[k] += rep.b[k];
  }
  ml::Softmax(arr, v);
}

void SampleV(const Document &doc, const VReal &h, const RepSoftMax &rbm,
                                                        VInt* v) {
  VReal expect(doc.words.size());
  ExpectV(doc, h, rbm, &expect);
  for (size_t i = 0; i < doc.total; ++i) {
    v->at(Random(expect))++;
  }
}

void SampleV2(int len, const VReal &h, const RepSoftMax &rep, VInt* v) {
  VReal expect(rep.w[0].size());
  ExpectV2(h, rep, &expect);
  for (int i = 0; i < len; ++i) {
    v->at(Random(expect))++;
  }
}

void ExpectH(int len, const VInt &words, const VInt &counts,
             const RepSoftMax &rbm, VReal* h) {
  h->resize(rbm.c.size());
  for (size_t f = 0; f < h->size(); f++) {
    double sum = 0.0;
    for (size_t w = 0; w < words.size(); ++w) {
      sum += rbm.w[f][words[w]] * counts[w];
    }
    sum += len * rbm.c[f];
    h->at(f) = Sigmoid(sum);
  }
}

void ExpectH(const Document &doc, const RepSoftMax &rbm, VReal* h) {
  ExpectH(doc.total, doc.words, doc.counts, rbm, h);
}

void SampleH(int len, const VInt &words, const VInt &counts,
                      const RepSoftMax &rbm, VReal* h) {
  ExpectH(len, words, counts, rbm, h);
  for (size_t i = 0; i < h->size(); i++) {
    (*h)[i] = Sample1((*h)[i]);
  }
}
 
void SampleH(const Document &doc,const RepSoftMax &rbm, VReal* h) {
  SampleH(doc.total, doc.words, doc.counts, rbm, h);
}

void CreateTmp(const Document &doc, VInt* v) {
  for (size_t i = 0; i < doc.words.size(); i++) {
    v->at(doc.words[i]) = doc.counts[i];
  }
}

void RBMLearning(const Corpus &corpus, int itern, RepSoftMax* rep) {
  int count = 0;
  VVReal result;
  result.reserve(corpus.Len());
  // Init(2, 4, 0.1, &(rep->w));
  // rep->w[0][2] = -0.5;
  for(int k = 0; k < itern; ++k) {
    result.clear();
    for(size_t i = 0; i < corpus.Len(); ++i) {
    // for(int i = 0; i < 1; ++i) {
      // LOG_IF(INFO, i % 500 == 0) << k << " " << i;
      count++;
      VReal h1;
      SampleH(corpus.docs[i], *rep, &h1);
      // ExpectH(corpus.docs[i], *rep, &h1);
      VInt v2(rep->w[0].size());
      SampleV(corpus.docs[i], h1, *rep, &v2);
      VReal h2;
      ExpectH(corpus.docs[i].total, corpus.docs[i].words, v2, *rep, &h2);
      // LOG_IF(INFO, i == 0) << Join(h2, " ");
      // LOG_IF(INFO, i == 43) << Join(h2, " ");
      result.push_back(h2);
      // VInt tmp2(v2.size());
      // CreateTmp(corpus.docs[i], &tmp2);
      // Gradient(corpus.docs[i].total, h1, tmp2, h2, v2, rep);
      Gradient(corpus.docs[i].total, corpus.docs[i].words, h1,
         corpus.docs[i].counts,  h2, v2, rep);
      if (count == rep->bach_size) {
        count = 0;
        Update(rep);
        rep->InitZero();
      }
      VReal tmp;
      ::Sum(rep->w, &tmp);
      LOG_IF(INFO, i % 500 == 0)<< k << ":" << Join(tmp, " ") << ":var--" << Var(rep->w)
            << ":Mean-" << Mean(rep->w) << ":Sum-" << Sum(rep->w);
       LOG_IF(INFO, i == 0) << "sum:" << Sum(rep->w) << "mean:" << Mean(rep->w);
    }
    WriteStrToFile(Join(result, " ", "\n"), "result.txt");
  }
}

void RBMLearning2(const Corpus &corpus, int itern, RepSoftMax* rep) {
  int count = 0;
  VVReal result;
  result.reserve(corpus.Len());
  for(int k = 0; k < itern; ++k) {
    result.clear();
    for(size_t i = 0; i < corpus.Len(); ++i) {
      LOG(INFO) << i;
      count++;
      VReal h1;
      SampleH(corpus.docs[i], *rep, &h1);
      VInt v2(rep->w[0].size());
      SampleV2(corpus.docs[i].total, h1, *rep, &v2);
      VReal h2;
      ExpectH(corpus.docs[i].total, corpus.docs[i].words, v2, *rep, &h2);
      result.push_back(h2);
      VInt tmp2(v2.size());
      CreateTmp(corpus.docs[i], &tmp2);
      Gradient2(corpus.docs[i].total, h1, tmp2, h2, v2, rep);
      if (count == rep->bach_size) {
        count = 0;
        Update(rep);
        rep->InitZero();
      }
      VReal tmp;
      ::Sum(rep->w, &tmp);
      LOG_IF(INFO, i % 500 == 0)<< k << ":" << Join(tmp, " ") << ":var--" <<
            Var(rep->w) << ":Mean-" << Mean(rep->w) << ":Sum-" << Sum(rep->w);
    }
    WriteStrToFile(Join(result, " ", "\n"), "result.txt");
  }
}

void RepSoftmaxTest(const RepSoftMax &rbm, const Corpus &corpus, 
                                           Real partition_fc) {
  VReal PVHs;
  PVHs.reserve(corpus.Len());
  for (size_t i = 0; i < corpus.Len(); ++i) {
    VReal h1;
    SampleH(corpus.docs[i], rbm, &h1);
    const VInt &words = corpus.docs[i].words;
    const VInt &counts = corpus.docs[i].counts;
    Real D = std::accumulate(counts.begin(), counts.end(), 0.0);

    Real vbias = 0.0;
    for(size_t k = 0; k < words.size(); ++k) {
      vbias += rbm.b[words[k]] * counts[k];
    }

    Real hbias = 0.0;
    for(size_t h = 0; h < h1.size(); ++h) {
      hbias += rbm.c[h] * h1[h];
    }
    hbias *= D;

    Real fvweight = 0.0;
    for(size_t f = 0; f < h1.size(); ++f) {
      for( size_t k = 0; k < words.size(); ++k) {
        fvweight += rbm.w[f][words[k]] * h1[f] * counts[k];
      }
    }

    Real Pvh = exp(-1 * (fvweight + vbias + hbias)) / partition_fc; 
    PVHs.push_back(Pvh);
  }
  WriteStrToFile(Join(PVHs,"\n"), "test.txt");
}

void OneRep(int f, int k, RepSoftMax* rep) {
  Init(f, k, 1, &(rep->w));
  Init(k, 1, &(rep->b));
  Init(f,1, &(rep->c));
}

void ZeroRep(int f, int k, RepSoftMax* rep) {
  Init(f, k, 0, &(rep->w));
  Init(k, 0, &(rep->b));
  Init(f, 0, &(rep->c));
}

void InitRep(int f, int k, double value, RepSoftMax* rep) {
  Init(f, k, value, &(rep->w));
  Init(k, value, &(rep->b));
  Init(f, value, &(rep->c));
}
} // namespace ml
