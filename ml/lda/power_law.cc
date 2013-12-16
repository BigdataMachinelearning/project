// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"

#include "ml/document.h"
#include "ml/lda/power_law.h"
namespace ml {
void EStep(CorpusC &c, VVRealC &theta, VReal& alpha, int K, VVVReal* z) { 
  for(int m = 0; m < c.Len(); m++) {
  // for(int m = 0; m < 1; m++) {
    for(int n = 0; n < c.DocLen(m); n++) {
      for(int k = 0; k < K; k++) {
        (*z)[m][n][k] = theta[m][k] * pow(c.Count(m, n), -alpha[k])
             / (alpha[k] - 1);
      }
      double sum = std::accumulate((*z)[m][n].begin(), (*z)[m][n].end(), 0.0);
      for(int k = 0; k < K; k++) {
        (*z)[m][n][k] /= sum;
      }
    }
  }
}

void MStepAlpha(CorpusC &c, VVVRealC &z, VReal* alpha) {
  for (size_t k = 0; k < z[0][0].size(); k++) {
    double tmp1 = 0;
    double tmp2 = 0;
    for (int m = 0;  m < c.Len(); m++) {
      for (int n = 0; n < c.DocLen(m); n++) {
        tmp1 += z[m][n][k];
        tmp2 += z[m][n][k] * log(static_cast<double>(c.Count(m, n)));
      }
    }
    (*alpha)[k] = 1 - tmp1 / tmp2;
  }
}

void Normalize(VReal* data) {
  double sum = std::accumulate(data->begin(), data->end(), 0.0);
  for (VReal::size_type i = 0; i < data->size(); i++) {
    data->at(i) /= sum;
  }
}

void MStepTheta(CorpusC &c, VVVRealC &z, VVReal* theta) {
  for (int m = 0;  m < c.Len(); m++) {
    for (size_t k = 0; k < z[0][0].size(); k++) {
      (*theta)[m][k] = 0;
      for (int n = 0; n < c.DocLen(m); n++) {
        (*theta)[m][k] += z[m][n][k];
      }
    }
    Normalize(&(theta->at(m)));
  }
}

void EM(CorpusC &c, int K, VVReal* theta, VReal* alpha) {
  Init(K, 5, alpha);
  theta->resize(c.Len());
  for (int i = 0; i < c.Len(); i++) {
    theta->at(i).resize(K);
    for (int k = 0; k < K; k++) {
      theta->at(i).at(k) = 0.1;
    }
    theta->at(i).at(5) = 0.2; 
    theta->at(i).at(1) = 0.2; 
    theta->at(i).at(0) = 0.7; 
  }
  int iter = 100;
  VVVReal z;
  c.NewLatent(&z, K);
  for(int i = 0; i < iter; i++) {
    LOG(INFO) << i << " " << iter - i;
    EStep(c, *theta, *alpha, K, &z);
    LOG(INFO) << Join(*alpha, " ");
    MStepAlpha(c, z, alpha);
    MStepTheta(c, z, theta);
  }
}
}  // namespace ml
