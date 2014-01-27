// copyright 2013 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_LDA_VAR_EM_H
#define ML_LDA_VAR_EM_H
#include "base/base_head.h"

#include "ml/lda/lda.h"
#include "ml/lda/lda_model.h"
#include "ml/lda/cokus.h"

const int LAG = 5;
namespace ml {
class LDA {
 public:
  void CreateSS(const Str &type, const Corpus &c, const LdaModel &m,
                                 LdaSuffStats* ss) const;
  double Likelihood(const Corpus &cor, int d, LdaModelC &m, VRealC &gamma,
                                                 VVRealC &phi) const;
  double Infer(const Corpus &cor, int d, LdaModelC &m, VReal* ga,
                                            VVReal* phi) const;

  double Perplexity(const Corpus &cor, const VVReal &gamma,
                       const VVVReal &phi, const LdaModel &lda);
  void RunEM(const Str &type, const Corpus &train,
                const Corpus &test, LdaModel* m);

  void Infer(LdaModelC &m, VVReal* ga, VVVReal* phi) const;
  void Infer(const Corpus &cor, LdaModelC &m, VVReal* ga, VVVReal* phi) const;
 
  void Gibbs(const Corpus &cor) const;

  inline void Init(float em_converged, int em_max_iter, int estimate_alpha,
                   int var_max_iter, int var_converged_,
                   double initial_alpha, int n_topic);
 private:
  void InitVar(const Corpus &cor, int d, LdaModelC &m, VReal* digamma,
                                      VReal* ga, VVReal* phi) const;
  void InitVar(const Document &doc, const LdaModel &model,
               double* digamma, double* gamma, double** phi) const;

  double DocEStep(const Corpus &cor, int d, LdaModelC &m,
                     LdaSuffStats* ss) const;
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
} // namespace ml
#endif // ML_LDA_VAR_EM_H
