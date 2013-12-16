// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef LDA_POWER_LAW_H_
#define LDA_POWER_LAW_H_
#include "base/base_head.h"
namespace ml {
class Corpus;
void EM(CorpusC &c, int K, VVReal* theta, VReal* alpha);
}  // namespace ml
#endif // LDA_POWER_LAW_H_
