// Copyright 2013 yuanwujun, lijiankou. All Rights Reserved.
// Author: real.yuanwj@gmail.com lijk_start@163.com
#ifndef ML_RBM_AIS_H_
#define ML_RBM_AIS_H_
#include "base/base_head.h"
#include "repsoftmax.h"
namespace ml{
class Document;
double Partition(const Document &doc, int runs, const VReal &beta,
                                        const RepSoftMax &rep);
void UniformSample(const Document &doc, VInt* v);
double Probability(const Document &doc, int runs, const VReal &beta,
                                        const RepSoftMax &rbm);
double LogPartition(int doc_len, int word_num, const RepSoftMax &rep);
}; // namespace ml
#endif // ML_RBM_AIS_H_
