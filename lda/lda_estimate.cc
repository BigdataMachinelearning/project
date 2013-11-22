// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "lda_estimate.h"

#include <assert.h>
#include <cstring>
#include <cstdlib>

#include "base/base_head.h"
#include "lda/lda.h"

namespace topic {
double LDA::DocEStep(document* doc, double* gamma, double** phi,
                     lda_model* model, lda_suffstats* ss) {
  double likelihood = Inference(doc, model, gamma, phi);
  double gamma_sum = 0;
  for (int k = 0; k < model->num_topics; k++) {
    gamma_sum += gamma[k];
    ss->alpha_suffstats += DiGamma(gamma[k]);
  }
  ss->alpha_suffstats -= model->num_topics * DiGamma(gamma_sum);
  for (int n = 0; n < doc->length; n++) {
    for (int k = 0; k < model->num_topics; k++) {
      ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[n][k];
      ss->class_total[k] += doc->counts[n]*phi[n][k];
    }
  }
  ss->num_docs = ss->num_docs + 1;
  return likelihood;
}

void LDA::CreateSS(const Str &type, const Corpus &c,
                   LdaModel* m, LdaSuffStats* ss) {
  if (type == "seeded") {
    CorpusInitSS(c, *m, ss);
  } else if (type == "random") {
    RandomInitSS(*m, ss);
  }
  LdaMLE(0, m, ss);
  m->alpha = initial_alpha_;
}
 
void LDA::RunEM(const Str &type, const Corpus &corpus,
                double** gamma, double** phi) {
  LdaModel model;
  NewLdaModel(n_topic_, corpus.num_terms, &model);
  LdaSuffStats ss;
  NewLdaSuffStats(model, &ss);
  CreateSS(type, corpus, &model, &ss);
  double converged = 1;
  double likelihood_old = 0;
  int i = 0;
  while (((converged < 0) || (converged > em_converged_) ||
                          (i <= 2)) && (i++ <= em_max_iter_)) {
    // LOG(INFO) << i;
    std::cout << "repeat " << i << std::endl;
    double likelihood = 0;
    InitSS(model, 0, &ss);
    for (int d = 0; d < corpus.num_docs; d++) {
      likelihood += DocEStep(&(corpus.docs[d]), gamma[d], phi, &model, &ss);
    }
    LdaMLE(estimate_alpha_, &model, &ss);
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0) {
      var_max_iter_ = var_max_iter_ * 2;
    }
    likelihood_old = likelihood;
  }
}

double LDA::Likelihood(const Document &doc, const LdaModel &m, 
                       const VVReal &phi, const VReal &gamma) const {
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
    for (int n = 0; n < doc.length; n++) {
      if (phi[n][k] > 0) {
        l += doc.counts[n] * phi[n][k] * (expect[k] - log(phi[n][k])
                              + m.log_prob_w[k][doc.words[n]]);
      }
    }
  }
  return l;
}

double LDA::Likelihood(document* doc, lda_model* model, double** phi,
                                                        double* gamma) {
  const int &num_topic = model->num_topics;
  const double &alpha = model->alpha;
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
    for (int n = 0; n < doc->length; n++) {
      if (phi[n][k] > 0) {
        likelihood += doc->counts[n]*
          (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
          + model->log_prob_w[k][doc->words[n]]));
      }
    }
  }
  return likelihood;
}

void InitVar(const Document &doc, const LdaModel &model,
             double* digamma, double* gamma, double** phi) {
  for (int k = 0; k < model.num_topics; k++) {
    gamma[k] = model.alpha + (doc.total / ((double) model.num_topics));
    digamma[k] = DiGamma(gamma[k]);
    for (int n = 0; n < doc.length; n++) {
      phi[n][k] = 1.0 / model.num_topics;
    }
  }
}

double LDA::Inference(document* doc, lda_model* model, double* gamma,
                                                       double** phi) {
  double converged = 1;
  double digamma_gam[model->num_topics];
  InitVar(*doc, *model, digamma_gam, gamma, phi);
  double likelihood_old = 0;
  int it = 1;
  while ((converged > var_converged_) && (it++ < var_max_iter_)) {
    for (int n = 0; n < doc->length; n++) {
      double phisum = 0;
      double oldphi[model->num_topics];
      for (int k = 0; k < model->num_topics; k++) {
        oldphi[k] = phi[n][k];
        phi[n][k] = digamma_gam[k] + model->log_prob_w[k][doc->words[n]];
        if (k > 0) {
          phisum = LogSum(phisum, phi[n][k]);
        } else {
          phisum = phi[n][k]; 
        }
      }
      for (int k = 0; k < model->num_topics; k++) {
        phi[n][k] = exp(phi[n][k] - phisum);
        gamma[k] = gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
        digamma_gam[k] = DiGamma(gamma[k]);
      }
    }
    double likelihood = Likelihood(doc, model, phi, gamma);
    assert(!isnan(likelihood));
    converged = (likelihood_old - likelihood) / likelihood_old;
    likelihood_old = likelihood;
  }
  return likelihood_old;
}
} // namespace topic
