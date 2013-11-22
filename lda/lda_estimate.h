// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_ESTIMATE_H
#define LDA_ESTIMATE_H
#include "base/base_head.h"

#include "lda/lda.h"
#include "lda/lda_model.h"
#include "cokus.h"

const int LAG = 5;
namespace topic {
class LDA {
 public:
  inline void Init(float em_converged, int em_max_iter, int estimate_alpha,
                   int var_max_iter, int var_converged_,
                   double initial_alpha, int n_topic);
  double DocEStep(const Document &doc, double* gamma, double** phi,
                                 LdaModel* model, LdaSuffStats* ss);
  void RunEM(const Str &mode, const Corpus &corpus,
             double** var_gamma, double** phi);
  void CreateSS(const Str &mode, const Corpus &c,
                LdaModel* model, LdaSuffStats* ss);
  double Inference(const Document &, LdaModel*, double*, double**);
  double Likelihood(const Document &, LdaModel*, double**, double*);
  double Likelihood(const Document &doc, const LdaModel &m, 
                    const VVReal &phi, const VReal &gamma) const;
 private:
  float em_converged_;
  int em_max_iter_;
  int estimate_alpha_;
  double initial_alpha_;
  int var_max_iter_;
  int n_topic_;
  int var_converged_;
};

void LDA::Init(float em_converged, int em_max_iter, int estimate_alpha,
                                   int var_max_iter, int var_converged,
                                   double initial_alpha, int n_topic) {
  em_converged_ = em_converged;
  em_max_iter_ = em_max_iter;
  estimate_alpha_ = estimate_alpha;
  initial_alpha_ = initial_alpha;
  n_topic_ = n_topic;
  var_converged_ = var_converged;
  var_max_iter_ = var_max_iter;
}
} // namespace topic
#endif
