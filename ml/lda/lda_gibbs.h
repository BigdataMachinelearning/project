// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_LDA_LDA_GIBBS_H
#define ML_LDA_LDA_GIBBS_H
#include "base/base_head.h"
#include "ml/lda/lda.h"
namespace ml {
int Sampling(CorpusC &corpus, VVIntC &z, LdaSuffStats* suff);
void GibbsInfer(int Num, int k, CorpusC &corpus, LdaModel* model);
void GibbsInitSS(CorpusC &corpus, int k, VVInt* z, LdaSuffStats* ss);
} // namespace topic
#endif // ML_LDA_LDA_GIBBS_H
