// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_BASE_H_
#define BASE_BASE_H_
#include <cmath>
#include "base/type.h"
void Init(int len, double value, VReal* des);
void Init(int row, int col, double value, VVReal* des);
void Cumulate(VReal* des);
int Random(const VReal &data);
int Random(int k);
#endif  // BASE_BASE_H_
