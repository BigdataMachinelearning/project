// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/lda/lda_gibbs.h"

#include "ml/lda/lda.h"
#include "ml/lda/lda_model.h"
#include <map>

namespace ml {
// void GibbsInitSS(const LdaModel &m, LdaSuffStats* ss) {
void UpdataSuff(int m, int k, int v, int value, LdaSuffStats* suff) {
  suff->phi[k][v] += value;
  suff->theta[m][k] += value;
  suff->sum_phi[k] += value;
  suff->sum_theta[m] += value;
}

void GibbsInitSS(CorpusC &corpus, int k, VVInt* z, LdaSuffStats* ss) {
  corpus.NewLatent(z);
  MIntInt dic; // for test
  ss->Init(corpus.Len(), k, corpus.num_terms);
  for (size_t m = 0; m < corpus.Len(); m++) {
    for (size_t n = 0; n < corpus.ULen(m); n++) {
      for (int i = 0; i < corpus.Count(m, n); i++) {
        (*z)[m][n] = Random(k);
        dic[(*z)[m][n]]++; // for test
        UpdataSuff(m, (*z)[m][n], corpus.Word(m, n), 1, ss);
      }
    }
  }
  LOG(INFO) << MapToStr(dic); //for test
}

int Sampling(int m, int n, CorpusC &corpus, VVIntC &z, LdaModelC &model,
                                                       LdaSuffStats* suff) {
  int topic = z[m][n];
  int w = corpus.Word(m, n);
  UpdataSuff(m, topic, w, -1, suff);
  double Vbeta = suff->phi[0].size() * model.beta;
  double Kalpha = suff->phi.size() * model.alpha;    
  VReal p(suff->phi.size());
  for (VReal::size_type k = 0; k < p.size(); k++) {
    p[k] = (suff->phi[k][w] + model.beta) / (suff->sum_phi[k] + Vbeta) *
    (suff->theta[m][k] + model.alpha) / (suff->sum_theta[m] + Kalpha);
  }
  topic = Random(p);
  UpdataSuff(m, topic, w, 1, suff);
  return topic;
}

void ComputeTheta(LdaSuffStatsC &suff, LdaModel* model) {
  model->theta.resize(suff.theta.size());
  for (VVInt::size_type m = 0; m < suff.theta.size(); m++) {
    model->theta[m].resize(suff.theta[m].size());
    for (VInt::size_type k = 0; k < suff.theta[m].size(); k++) {
      model->theta[m][k] = (suff.theta[m][k] + model->alpha) /
      (suff.sum_theta[m] + suff.theta[m].size() * model->alpha);
    }
  }
}

void ComputePhi(const LdaSuffStats &suff, LdaModel* model) {
  model->phi.resize(suff.phi.size());
  for (VVInt::size_type k = 0; k < suff.phi.size(); k++) {
    model->phi[k].resize(suff.phi[k].size());
    for (VInt::size_type w = 0; w < suff.phi[k].size(); w++) {
      model->phi[k][w] = (suff.phi[k][w] + model->beta) / 
        (suff.sum_theta[k] + suff.phi[k].size() * model->beta);
    }
  }
}

void GibbsInfer(int Num, int k, CorpusC &corpus, LdaModel* model) {
  VVInt z;
  LdaSuffStats suff;
  GibbsInitSS(corpus, k, &z, &suff);
  for (int i = 0; i <= Num; ++i) { 
    LOG(INFO) << i;
    for (size_t m = 0; m < corpus.Len(); m++) { 
      for (size_t n = 0; n < corpus.docs[m].ULen(); n++) { 
        z[m][n] = Sampling(m, n, corpus, z, *model, &suff); 
      } 
    } 
  } 
  ComputeTheta(suff, model); 
  ComputePhi(suff, model); 
} 
} // namespace ml
