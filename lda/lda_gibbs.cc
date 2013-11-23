// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda/lda_infer_gibbs.h"

#include "lda/lda.h"
#include "lda/lda_model.h"
namespace topic {
int Sampling(int m, int n, const Corpus &corpus, VVIntC &z,
                           const LdaModel &model, LdaSuffStats* suff) {
  int topic = z[m][n];
  int w = corpus.docs[m].words[n];
  suff->ss_phi[w][topic] -= 1;
  suff->ss_theta[m][topic] -= 1;
  suff->sum_ss_phi[topic] -= 1;
  suff->sum_ss_theta[m] -= 1;
  double Vbeta = suff->ss_phi.size() * model.beta;
  double Kalpha = suff->ss_phi[0].size() * model.alpha;    
  VReal p(suff->ss_phi[0].size());
  for (VReal::size_type k = 0; k < p.size(); k++) {
    p[k] = (suff->ss_phi[w][k] + model.beta) / (suff->sum_ss_phi[k] + Vbeta) *
    (suff->ss_theta[m][k] + model.alpha) / (suff->sum_ss_theta[m] + Kalpha);
  }
  for (VReal::size_type k = 1; k < p.size(); k++) {
    p[k] += p[k - 1];
  }
  double u = ((double)random() / RAND_MAX) * p[p.size() - 1];
  for (topic = 0; topic < static_cast<int>(p.size()); topic++) {
    if (p[topic] > u) {
      break;
    }
  }
  suff->ss_phi[w][topic] += 1;
  suff->ss_theta[m][topic] += 1;
  suff->sum_ss_phi[topic] += 1;
  suff->sum_ss_theta[m] += 1;
  return topic;
}

void ComputeTheta(const LdaSuffStats &suff, LdaModel* model) {
  for (VVInt::size_type m = 0; m < suff.ss_theta.size(); m++) {
    for (VInt::size_type k = 0; k < suff.ss_theta[m].size(); k++) {
      model->theta[m][k] = (suff.ss_theta[m][k] + model->alpha) /
      (suff.sum_ss_theta[m] + suff.ss_theta[m].size() * model->alpha);
    }
  }
}

void ComputePhi(const LdaSuffStats &suff, LdaModel* model) {
  for (VVInt::size_type k = 0; k < suff.ss_phi.size(); k++) {
    for (VInt::size_type w = 0; w < suff.ss_phi[k].size(); w++) {
      model->phi[k][w] = (suff.ss_phi[w][k] + model->beta) / 
        (suff.sum_ss_theta[k] + suff.ss_phi[k].size() * model->beta);
    }
  }
}

void GibbsInfer(int Num, const Corpus &corpus, LdaModel* model) {
  VVInt z;
  LdaSuffStats suff;
  for (int i = 0; i <= Num; ++i) { 
    for (int m = 0; m < corpus.Len(); m++) { 
      for (int n = 0; n < corpus.docs[m].Len(); n++) { 
        z[m][n] = Sampling(m, n, corpus, z, *model, &suff); 
      } 
    } 
  } 
  ComputeTheta(suff, model); 
  ComputePhi(suff, model); 
} 

void GibbsInfer(const Corpus &corpus, int k, LdaSuffStats* ss, VVInt* z) {
  z->resize(corpus.Len());
  for (int m = 0; m < corpus.Len(); m++) {
    z->at(m).resize(corpus.docs[m].Len());
    for (int n = 0; n < corpus.docs[m].Len(); n++) {
      int w = corpus.docs[m].words[n];
      int topic = (int)(((double)random() / RAND_MAX) * k);
      z->at(m).at(n) = topic;
      ss->ss_phi[w][topic] += 1;
      ss->ss_theta[m][topic] += 1;
      ss->sum_ss_phi[topic] += 1;
    } 
    ss->sum_ss_theta[m] = corpus.docs[m].Len();
  }    
}
} // namespace topic
