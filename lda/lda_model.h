#ifndef LDA_MODEL_H
#define LDA_MODEL_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "base/base_head.h"
#include "lda/lda.h"
#include "cokus.h"

namespace topic {
inline double Alhood(double a, double ss, int d, int k) {
  return(d * (lgamma(k * a) - k * lgamma(a)) + (a - 1) * ss);
}

inline double DAlhood(double a, double ss, int d, int k) {
  return(d * (k * DiGamma(k * a) - k * DiGamma(a)) + ss);
}

inline double D2Alhood(double a, int d, int k) {
  return ( d * (k * k * TriGamma(k * a) - k * TriGamma(a)));
}

double OptAlpha(double ss, int d, int k);
void NewLdaModel(int, int, LdaModel* m);
void NewLdaSuffStats(const LdaModel &model, LdaSuffStats* ss);
void CorpusInitSS(const Corpus &c, const LdaModel &m, LdaSuffStats* ss);
void RandomInitSS(const LdaModel &m, LdaSuffStats* ss);
void InitSS(const LdaModel &model, double value, LdaSuffStats* ss);
void LdaMLE(int estimate_alpha, const LdaSuffStats &ss, LdaModel* m);
} // namespace topic
#endif // LDA_MODEL_H
