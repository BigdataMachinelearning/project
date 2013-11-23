// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda_var_em.h"

#include "base/base_head.h"
#include "lda/lda.h"

namespace topic {
double LDA::DocEStep(int d, const LdaModel &model, double* gamma, double** phi,
                                                           LdaSuffStats* ss) {
  double likelihood = Infer(d, model, gamma, phi);
  double gamma_sum = 0;
  for (int k = 0; k < model.num_topics; k++) {
    gamma_sum += gamma[k];
    ss->alpha_suffstats += DiGamma(gamma[k]);
  }
  ss->alpha_suffstats -= model.num_topics * DiGamma(gamma_sum);
  for (int n = 0; n < corpus.DocLen(d); n++) {
    for (int k = 0; k < model.num_topics; k++) {
      ss->class_word[k][corpus.Word(d, n)] += corpus.Count(d, n) * phi[n][k];
      ss->class_total[k] += corpus.Count(d, n) * phi[n][k];
    }
  }
  ss->num_docs = ss->num_docs + 1;
  return likelihood;
}

void LDA::CreateSS(const Str &type, const Corpus &c,
                   const LdaModel &m, LdaSuffStats* ss) const {
  if (type == "seeded") {
    CorpusInitSS(c, m, ss);
  } else if (type == "random") {
    RandomInitSS(m, ss);
  }
}

double LDA::Likelihood(int d, const LdaModel &m, const VReal &gamma,
                                      const VVReal &phi) const {
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

double LDA::Likelihood(int d, const LdaModel &m, double** phi, double* gamma) {
  const int &num_topic = m.num_topics;
  const double &alpha = m.alpha;
  double dig[num_topic];
  double gamma_sum = 0;
  for (int k = 0; k < num_topic; k++) {
    dig[k] = DiGamma(gamma[k]);
    gamma_sum += gamma[k];
  }
  double digsum = DiGamma(gamma_sum);
  double likelihood = lgamma(alpha * num_topic) - num_topic * lgamma(alpha)
	                                        - lgamma(gamma_sum);
  for (int k = 0; k < num_topic; k++) {
    likelihood += (alpha - 1)*(dig[k] - digsum) + lgamma(gamma[k])
	                              - (gamma[k] - 1)*(dig[k] - digsum);
    for (int n = 0; n < corpus.DocLen(d); n++) {
      if (phi[n][k] > 0) {
        likelihood += corpus.Count(d, n) *
          (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
          + m.log_prob_w[k][corpus.Word(d, n)]));
      }
    }
  }
  return likelihood;
}

void LDA::InitVar(int d, const LdaModel &m, VReal* digamma,
                         VReal* ga, VVReal* phi) const {
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

void LDA::InitVar(const Document &doc, const LdaModel &model,
             double* digamma, double* gamma, double** phi) const {
  for (int k = 0; k < model.num_topics; k++) {
    gamma[k] = model.alpha + (doc.total / ((double) model.num_topics));
    digamma[k] = DiGamma(gamma[k]);
    for (size_t n = 0; n < doc.words.size(); n++) {
      phi[n][k] = 1.0 / model.num_topics;
    }
  }
}

double LDA::Infer(int d, const LdaModel &m, double* gamma, double** phi) {
  double converged = 1;
  double digamma_gam[m.num_topics];
  InitVar(corpus.docs[d], m, digamma_gam, gamma, phi);
  double likelihood_old = 0;
  int it = 1;
  while ((converged > var_converged_) && (it++ < var_max_iter_)) {
    for (int n = 0; n < corpus.DocLen(d); n++) {
      double phisum = 0;
      double oldphi[m.num_topics];
      for (int k = 0; k < m.num_topics; k++) {
        oldphi[k] = phi[n][k];
        phi[n][k] = digamma_gam[k] + m.log_prob_w[k][corpus.Word(d, n)];
        if (k > 0) {
          phisum = LogSum(phisum, phi[n][k]);
        } else {
          phisum = phi[n][k]; 
        }
      }
      for (int k = 0; k < m.num_topics; k++) {
        phi[n][k] = exp(phi[n][k] - phisum);
        gamma[k] = gamma[k] + corpus.Count(d, n) * (phi[n][k] - oldphi[k]);
        digamma_gam[k] = DiGamma(gamma[k]);
      }
    }
    double likelihood = Likelihood(d, m, phi, gamma);
    assert(!isnan(likelihood));
    converged = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}
 
void LDA::RunEM(const Str &type, double** gamma, double** phi) {
  LdaModel model;
  NewLdaModel(n_topic_, corpus.num_terms, &model);
  LdaSuffStats ss;
  NewLdaSuffStats(model, &ss);
  CreateSS(type, corpus, model, &ss);
  LdaMLE(0, ss, &model);
  model.alpha = initial_alpha_;
  double converged = 1;
  double likelihood_old = 0;
  int i = 0;
  while (((converged < 0) || (converged > em_converged_) ||
                          (i <= 2)) && (i++ <= em_max_iter_)) {
    // LOG(INFO) << i;
    std::cout << "repeat " << i << std::endl;
    double likelihood = 0;
    InitSS(model, 0, &ss);
    for (int d = 0; d < corpus.Len(); d++) {
      likelihood += DocEStep(d, model, gamma[d], phi, &ss);
    }
    LdaMLE(estimate_alpha_, ss, &model);
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihood;
  }
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
  int i = 0;
  while (((converged < 0) || (converged > em_converged_) ||
                          (i <= 2)) && (i++ <= em_max_iter_)) {
    // LOG(INFO) << i;
    std::cout << "repeat " << i << std::endl;
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
} // namespace topic
