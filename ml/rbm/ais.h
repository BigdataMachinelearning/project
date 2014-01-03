// Copyright 2013 yuanwujun, lijiankou. All Rights Reserved.
// Author: real.yuanwj@gmail.com lijk_start@163.com
#ifndef ML_ANNEALING_IMPORTANT_SAMPLING_H_
#define ML_ANNEALING_IMPORTANT_SAMPLING_H_

#include "base/base_head.h"
#include "repsoftmax.h"
namespace ml{
double AISEstimate(int runs, const VReal &belts,const RepSoftMax &rbm);
};
#endif // ML_ANNEALING_IMPORTANT_SAMPLING_H_
