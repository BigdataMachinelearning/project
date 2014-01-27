// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda_var_em.h"

#include <omp.h>
#include "base/base_head.h"
#include "ml/lda/lda.h"
#include "ml/lda/lda_gibbs.h"

namespace ml {
double LDA::Perplexity(const Corpus &cor, const VVReal &gamma,
                       const VVVReal &phi, const LdaModel &lda) {
  double sum = 0;
  for (size_t i = 0; i < cor.Len(); i++) {
    sum += Likelihood(cor, i, lda, gamma[i], phi[i]);
  }
  LOG(INFO) << sum << " " << cor.TermNum();
  return exp(- sum / cor.TermNum());
}

void LDA::CreateSS(StrC &t, CorpusC &c, LdaModelC &m, LdaSuffStats* ss) const {
  if (t == "seeded") {
    CorpusInitSS(c, m, ss);
  } else if (t == "random") {
    RandomInitSS(m, ss);
  }
}

double LDA::Likelihood(const Corpus &cor, int d, LdaModelC &m, VRealC &gamma,
                                                 VVRealC &phi) const {
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
    for (size_t n = 0; n < cor.ULen(d); n++) {
      if (phi[n][k] > 0) {
        l += cor.Count(d, n) * phi[n][k] * (expect[k] - log(phi[n][k])
                              + m.log_prob_w[k][cor.Word(d, n)]);
      }
    }
  }
  return l;
}

void LDA::InitVar(const Corpus &cor, int d, LdaModelC &m, VReal* digamma,
                                      VReal* ga, VVReal* phi) const {
  ga->resize(m.num_topics);
  digamma->resize(m.num_topics);
  phi->resize(cor.ULen(d));
  for (int k = 0; k < m.num_topics; k++) {
    (*ga)[k] = m.alpha + (cor.docs[d].total / ((double) m.num_topics));
    (*digamma)[k] = DiGamma((*ga)[k]);
  }
  for (VReal::size_type n = 0; n < phi->size(); n++) {
    phi->at(n).resize(m.num_topics);
    for (int k = 0; k < m.num_topics; k++) {
      (*phi)[n][k] = 1.0 / m.num_topics;
    }
  }
}

double LDA::Infer(const Corpus &cor, int d, LdaModelC &m, VReal* ga,
                                            VVReal* phi) const {
  VReal digamma(m.num_topics);
  InitVar(cor, d, m, &digamma, ga, phi);
  double likelihood_old = 0;
  int it = 1;
  double converged = 1;
  while ((converged > var_converged_) && (it++ < var_max_iter_)) {
    for (size_t n = 0; n < cor.ULen(d); n++) {
      double phisum = 0;
      VReal oldphi(m.num_topics);
      for (int k = 0; k < m.num_topics; k++) {
        oldphi[k] = (*phi)[n][k];
        (*phi)[n][k] = digamma[k] + m.log_prob_w[k][cor.Word(d, n)];
        if (k > 0) {
          phisum = LogSum(phisum, (*phi)[n][k]);
        } else {
          phisum = (*phi)[n][k]; 
        }
      }
      for (int k = 0; k < m.num_topics; k++) {
        (*phi)[n][k] = exp((*phi)[n][k] - phisum);
        (*ga)[k] = (*ga)[k] + cor.Count(d, n) * ((*phi)[n][k] - oldphi[k]);
        digamma[k] = DiGamma((*ga)[k]);
      }
    }
    double likelihood = Likelihood(cor, d, m, *ga, *phi);
    assert(!isnan(likelihood));
    converged = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}

double LDA::DocEStep(const Corpus &cor, int d, LdaModelC &m,
                     LdaSuffStats* ss) const {
  VReal gamma;
  VVReal phi;
  double likelihood = Infer(cor, d, m, &gamma, &phi);
  double gamma_sum = 0;
  for (int k = 0; k < m.num_topics; k++) {
    gamma_sum += gamma[k];
    ss->alpha_suffstats += DiGamma(gamma[k]);
  }
  ss->alpha_suffstats -= m.num_topics * DiGamma(gamma_sum);
  for (size_t n = 0; n < cor.ULen(d); n++) {
    for (int k = 0; k < m.num_topics; k++) {
      ss->class_word[k][cor.Word(d, n)] += cor.Count(d, n) * phi[n][k];
      ss->class_total[k] += cor.Count(d, n) * phi[n][k];
    }
  }
  ss->num_docs = ss->num_docs + 1;
  return likelihood;
}
 
void LDA::RunEM(const Str &type, const Corpus &train,
                const Corpus &test, LdaModel* m) {
  NewLdaModel(n_topic_, train.num_terms, m);
  LdaSuffStats ss;
  NewLdaSuffStats(*m, &ss);
  CreateSS(type, train, *m, &ss);
  LdaMLE(0, ss, m);
  m->alpha = initial_alpha_;
  double converged = 1;
  double likelihood_old = 0;
  for (int i = 0; i < em_max_iter_; i++) {
    LOG(INFO) << i << " " << em_max_iter_ - i;
    double likelihood = 0;
    InitSS(*m, 0, &ss);
    for (size_t d = 0; d < train.Len(); d++) {
      likelihood += DocEStep(train, d, *m, &ss);
    }
    LdaMLE(estimate_alpha_, ss, m);
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihood;

    VVReal gamma2;
    VVVReal phi2;
    Infer(test, *m, &gamma2, &phi2);
    LOG(INFO) << Perplexity(test, gamma2, phi2, *m);
  }
}

void LDA::Infer(const Corpus &cor, LdaModelC &m, VVReal* ga,
                                                 VVVReal* phi) const {
  for (size_t i = 0; i < cor.Len(); i++) {
    VReal ga2;
    VVReal phi2;
    Infer(cor, i, m, &ga2, &phi2);
    ga->push_back(ga2);
    phi->push_back(phi2);
  }
}

void LDA::Gibbs(const Corpus &cor) const {
  LdaSuffStats ss;
  int k = 10;
  int it = 15000;
  LdaModel model;
  model.alpha = 0.01;
  model.beta = 0.0001;
  GibbsInfer(it, k, cor, &model);
  WriteStrToFile(Join(model.theta, " ", "\n"), "gibs_theta");
  WriteStrToFile(Join(model.phi, " ", "\n"), "gibs_phi");
  LOG(INFO) << "over";
}
} // namespace ml 
