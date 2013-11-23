// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_LDA_INFER_GIBBS_H
#define LDA_LDA_INFER_GIBBS_H
#include "base/base_head.h"
namespace topic {
class Corpus;
class LdaModel;
class LdaSuffStats;
int Sampling(const Corpus &corpus, VVIntC &z, LdaSuffStats* suff);
void GibbsInfer(int Num, double alpha, double beta, int m, int n,
                    VVIntC &corpus, LdaModel* model);
void GibbsInfer(const Corpus &corpus, LdaSuffStats* ss, VVInt* z);
} // namespace topic
#endif // LDA_LDA_INFER_GIBBS_H
