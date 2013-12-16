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
  inline void Init(float em_converged, int em_max_iter, int estimate_alpha,
                   int var_max_iter, int var_converged_,
                   double initial_alpha, int n_topic);
  void RunEM(const Str &mode, double** var_gamma, double** phi);
  void CreateSS(const Str &type, const Corpus &c, const LdaModel &m,
                                 LdaSuffStats* ss) const;
  double Infer(int d, LdaModelC &m, VReal* ga, VVReal* phi) const;
  double Infer(LdaModelC &m, VReal* ga, VVReal* phi) const;
  double Likelihood(int d, LdaModelC &m, VRealC &gamma, VVRealC &phi) const;
  void LoadCorpus(const Str &filename);
  inline int Len() const;
  inline int MaxCorpusLen() const;
  void RunEM(const Str &type, LdaModel* m) ;
  void Infer(LdaModelC &m, VVReal* ga, VVVReal* phi) const;
  inline void AddDoc(const Document &doc);
  void Gibbs() const;
 private:
  void InitVar(int d, const LdaModel &model, VReal* digamma, VReal* gamma,
                                             VVReal* phi) const;
  void InitVar(const Document &doc, const LdaModel &model,
               double* digamma, double* gamma, double** phi) const;
  double DocEStep(int d, const LdaModel &model, LdaSuffStats* ss) const;
  float em_converged_;
  int em_max_iter_;
  int estimate_alpha_;
  double initial_alpha_;
  int var_max_iter_;
  int n_topic_;
  int var_converged_;
  Corpus corpus;
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

inline int LDA::Len() const {
  return corpus.Len();
}

inline int LDA::MaxCorpusLen() const {
  return corpus.MaxCorpusLen();
}

inline void LDA::AddDoc(const Document &doc) {
  corpus.docs.push_back(doc);
}
} // namespace ml
#endif // ML_LDA_VAR_EM_H
