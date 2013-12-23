// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/util.h"

namespace ml {
void Softmax(const VReal &a, VReal *b) {
  for(size_t i = 0; i < a.size(); ++i){
    double s = 0;
    for(size_t j = 0; j < a.size(); ++j) {
      s += exp(a[j] - a[i]);
    }
    b[0][i] = 1 / s;
  }
}

int Sample(const VReal &a){
  double r = Random1();
  for(size_t i = 0; i < a.size(); ++i) {
    r -= a[i];
    if(r < 0) {
      return i + 1;
    }
  }
  return a.size();
}

int SoftmaxSample(const VReal &a, VReal *b) {
  VReal e(a.size());
  Softmax(a, &e);
  return Sample(e);
}

double NormalSample() {
  static double V1, V2, S;
  static int phase = 0;
  double X;
  if (phase == 0) {
    do {
      double U1 = (double)rand() / RAND_MAX;
      double U2 = (double)rand() / RAND_MAX;
      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    X = V1 * sqrt(-2 * log(S) / S);
  } else {
    X = V2 * sqrt(-2 * log(S) / S);
  }
  phase = 1 - phase;
  return X;
}

void RandomInit(int len, VReal* des) {
  des->resize(len);
  for (int i = 0; i < len; i++) {
    des->at(i) = NormalSample() / 100;
  }
}

void RandomInit(int row, int col, VVReal* des) {
  for (int i = 0; i < row; i++) {
    VReal tmp;
    RandomInit(col, &tmp);
    des->push_back(tmp);
  }
}

void RandomInit(int len1, int len2, int len3, VVVReal* des) {
  for (int i = 0; i < len1; i++) {
    VVReal tmp;
    RandomInit(len2, len3, &tmp);
    des->push_back(tmp);
  }
}

double Mean(const VReal &v) {
  return Sum(v) / v.size();
}

double Mean(const VVReal &v) {
  VReal tmp;
  Append(v, &tmp);
  return Mean(tmp);
}

double Mean(const VVVReal &v) {
  VReal tmp;
  Append(v, &tmp);
  return Mean(tmp);
}

double Var(const VReal &v) {
  double mean = Mean(v);
  double sum = 0;
  for (size_t i = 0; i < v.size(); i++) {
    sum += Square(v[i] - mean);
  }
  return sum / v.size();
}

double Var(const VVReal &v) {
  VReal tmp;
  Append(v, &tmp);
  return Var(tmp);
}

double Var(const VVVReal &v) {
  VReal tmp;
  Append(v, &tmp);
  return Var(tmp);
}

double Sum(const VReal &v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}
} // namespace ml