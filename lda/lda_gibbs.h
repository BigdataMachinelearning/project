// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_LDA_GIBBS_H
#define LDA_LDA_GIBBS_H
#include "base/base_head.h"
#include "lda/lda.h"
namespace topic {
int Sampling(CorpusC &corpus, VVIntC &z, LdaSuffStats* suff);
void GibbsInfer(int Num, int k, CorpusC &corpus, LdaModel* model);
void GibbsInitSS(CorpusC &corpus, int k, VVInt* z, LdaSuffStats* ss);
} // namespace topic
#endif // LDA_LDA_GIBBS_H
