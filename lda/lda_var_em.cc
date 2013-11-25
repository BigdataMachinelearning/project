// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda_var_em.h"

#include "base/base_head.h"
#include "lda/lda.h"
#include "lda/lda_gibbs.h"

namespace topic {
void LDA::CreateSS(StrC &t, CorpusC &c, LdaModelC &m, LdaSuffStats* ss) const {
  if (t == "seeded") {
    CorpusInitSS(c, m, ss);
  } else if (t == "random") {
    RandomInitSS(m, ss);
  }
}

double LDA::Likelihood(int d, LdaModelC &m, VRealC &gamma, VVRealC &phi) const {
  double g_sum = std::accumulate(gamma.begin(), gamma.end(), 0.0);
  double digsum = DiGamma(g_sum);
  const int &num = m.num_topics;
  VReal expect(num);
  for (int k = 0; k < num; k++) {
    expect.at(k) = DiGamma(gamma.at(k)) - digsum;
  }
  double l = lgamma(m.alpha * num) - num * lgamma(m.alpha) - lgamma(g_sum);
  for (int k = 0; k < num; k++) {
    l += (m.alpha - gamma.at(k)) * expect[k] + lgamma(gamma.at(k));
    for (int n = 0; n < corpus.DocLen(d); n++) {
      if (phi[n][k] > 0) {
        l += corpus.Count(d, n) * phi[n][k] * (expect[k] - log(phi[n][k])
                              + m.log_prob_w[k][corpus.Word(d, n)]);
      }
    }
  }
  return l;
}

void LDA::InitVar(int d, LdaModelC &m, VReal* digamma, VReal* ga,
                                                       VVReal* phi) const {
  ga->resize(m.num_topics);
  digamma->resize(m.num_topics);
  phi->resize(corpus.DocLen(d));
  for (int k = 0; k < m.num_topics; k++) {
    (*ga)[k] = m.alpha + (corpus.docs[d].total / ((double) m.num_topics));
    (*digamma)[k] = DiGamma((*ga)[k]);
  }
  for (VReal::size_type n = 0; n < phi->size(); n++) {
    phi->at(n).resize(m.num_topics);
    for (int k = 0; k < m.num_topics; k++) {
      (*phi)[n][k] = 1.0 / m.num_topics;
    }
  }
}

double LDA::Infer(int d, LdaModelC &m, VReal* ga, VVReal* phi) const {
  VReal digamma(m.num_topics);
  InitVar(d, m, &digamma, ga, phi);
  double likelihood_old = 0;
  int it = 1;
  double converged = 1;
  while ((converged > var_converged_) && (it++ < var_max_iter_)) {
    for (int n = 0; n < corpus.DocLen(d); n++) {
      double phisum = 0;
      VReal oldphi(m.num_topics);
      for (int k = 0; k < m.num_topics; k++) {
        oldphi[k] = (*phi)[n][k];
        (*phi)[n][k] = digamma[k] + m.log_prob_w[k][corpus.Word(d, n)];
        if (k > 0) {
          phisum = LogSum(phisum, (*phi)[n][k]);
        } else {
          phisum = (*phi)[n][k]; 
        }
      }
      for (int k = 0; k < m.num_topics; k++) {
        (*phi)[n][k] = exp((*phi)[n][k] - phisum);
        (*ga)[k] = (*ga)[k] + corpus.Count(d, n) * ((*phi)[n][k] - oldphi[k]);
        digamma[k] = DiGamma((*ga)[k]);
      }
    }
    double likelihood = Likelihood(d, m, *ga, *phi);
    assert(!isnan(likelihood));
    converged = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}

double LDA::DocEStep(int d, const LdaModel &m, LdaSuffStats* ss) const {
  VReal gamma;
  VVReal phi;
  double likelihood = Infer(d, m, &gamma, &phi);
  double gamma_sum = 0;
  for (int k = 0; k < m.num_topics; k++) {
    gamma_sum += gamma[k];
    ss->alpha_suffstats += DiGamma(gamma[k]);
  }
  ss->alpha_suffstats -= m.num_topics * DiGamma(gamma_sum);
  for (int n = 0; n < corpus.DocLen(d); n++) {
    for (int k = 0; k < m.num_topics; k++) {
      ss->class_word[k][corpus.Word(d, n)] += corpus.Count(d, n) * phi[n][k];
      ss->class_total[k] += corpus.Count(d, n) * phi[n][k];
    }
  }
  ss->num_docs = ss->num_docs + 1;
  return likelihood;
}
 
void LDA::RunEM(const Str &type, LdaModel* m) {
  NewLdaModel(n_topic_, corpus.num_terms, m);
  LdaSuffStats ss;
  NewLdaSuffStats(*m, &ss);
  CreateSS(type, corpus, *m, &ss);
  LdaMLE(0, ss, m);
  m->alpha = initial_alpha_;
  double converged = 1;
  double likelihood_old = 0;
  for (int i = 0; i < em_max_iter_; i++) {
    LOG(INFO) << i << " " << em_max_iter_ - i;
    double likelihood = 0;
    InitSS(*m, 0, &ss);
    for (int d = 0; d < corpus.Len(); d++) {
      likelihood += DocEStep(d, *m, &ss);
    }
    LdaMLE(estimate_alpha_, ss, m);
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihood;
  }
}

void LDA::Infer(LdaModelC &m, VVReal* ga, VVVReal* phi) const {
  for (int i = 0; i < corpus.Len(); i++) {
    VReal ga2;
    VVReal phi2;
    Infer(i, m, &ga2, &phi2);
    ga->push_back(ga2);
    phi->push_back(phi2);
  }
}

void LDA::LoadCorpus(const Str &filename) {
  corpus.LoadData(filename);
}

void LDA::Gibbs() const {
  LdaSuffStats ss;
  int k = 10;
  int it = 15000;
  LdaModel model;
  model.alpha = 0.01;
  model.beta = 0.0001;
  GibbsInfer(it, k, corpus, &model);
  WriteStrToFile(Join(model.theta, " ", "\n"), "gibs_theta");
  WriteStrToFile(Join(model.phi, " ", "\n"), "gibs_phi");
  LOG(INFO) << "over";
}
} // namespace topic
