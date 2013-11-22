// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_BASE_H_
#define BASE_BASE_H_ 
#include <vector>

typedef std::vector<int> VInt;
typedef double Real;
typedef std::vector<Real> VReal;
typedef std::vector<VReal> VVReal;

inline void Init(int len, double value, VReal* des) {
  for (int i = 0; i < len; i++) {
    des->push_back(value);
  }
}

inline void Init(int row, int col, double value, VVReal* des) {
  for (int i = 0; i < row; i++) {
    VReal tmp;
    Init(col, value, &tmp);
    des->push_back(tmp);
  }
}
#endif // BASE_H_
