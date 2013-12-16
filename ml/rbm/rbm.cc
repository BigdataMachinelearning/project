// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
namespace ml {
void RBM::Init(int f, int m, int k, double momentum_, double eta_) {
  ::Init(f, m, k, 0.01, &w1);
  ::Init(m, k, 0.0, &b1);
  ::Init(f, 0.0, &c1);
  momentum = momentum_;
  eta = eta_;
}

void Update(const VInt &item, const VReal &h1, const VReal &v1,
            const VReal &h2, const VReal &v2, RBM* rbm) {
  for (VInt::size_type m = 0; m < item.size(); m++) {
    if (v1[m] != v2[m]) {
      rbm->b1[item[m]][v1[m]] += rbm->eta;
      rbm->b1[item[m]][v2[m]] -= rbm->eta;
    }
    for (VInt::size_type f = 0; f < h1.size(); f++) {
      if (v1[m] == v2[m]) {
        rbm->w1[f][item[m]][v1[m]] += rbm->eta * (h1[f] - h2[f]);
      } else {
        rbm->w1[f][item[m]][v1[m]] += rbm->eta * h1[f];
        rbm->w1[f][item[m]][v2[m]] -= rbm->eta * h2[f];
      }
    }
  }
  for (VInt::size_type i = 0; i < h1.size(); i++) {
    rbm->c1[i] += rbm->eta * (h1.at(i) - h2.at(i));
  }
}

void ExpectV(const VInt &item, const VReal &h, const RBM &rbm, VVReal* v) {
  v->resize(item.size());
  for (VVInt::size_type i = 0; i < item.size(); ++i) {
    VReal arr(rbm.w1[0][0].size());
    for (size_t k = 0; k < arr.size(); k++) {
      arr[k] = 0;
      for (size_t f = 0; f < h.size(); f++) {
        arr[k] += rbm.w1[f][item[i]][k] * h[f];
      }
      arr[k] += rbm.b1[item[i]][k];
    }
    SoftMax(arr, &(v->at(i)));
  }
}

void SampleV(const VInt &item, const VReal &h, const RBM &rbm, VReal* v) {
  VVReal tmp;
  ExpectV(item, h, rbm, &tmp);
  v->resize(item.size());
  for (VVReal::size_type i = 0; i < tmp.size(); ++i) {
    v->at(i) = Random(tmp[i]);
  }
}

void ExpectH(const VInt &item, const VReal &rating, const RBM &rbm, VReal* h) {
  h->resize(rbm.c1.size());
  for (size_t f = 0; f < h->size(); f++) {
    double sum = 0.0;
    for (VReal::size_type i = 0; i < rating.size(); ++i) {
      sum += rbm.w1[f][item[i]][rating[i]];
    }
    sum += rbm.c1[f];
    h->at(f) = Sigmoid(sum);
  }
}

void SampleH(const VInt &item, const VReal &rating, const RBM &rbm, VReal* h) {
  VReal tmp;
  ExpectH(item, rating, rbm, &tmp);
  h->resize(rbm.c1.size());
  for (size_t i = 0; i < h->size(); i++) {
    h->at(i) = RandSample(tmp.at(i));
  }
}

void RBMTrain(const User &train, const User &test, int iter_num, RBM* rbm) {
  for (int i = 0; i < iter_num; i++) {
    Time time;
    time.Start();
    for (size_t j = 1; j < train.item.size(); j++) {
      VReal h1;
      SampleH(train.item[j], train.rating[j], *rbm, &h1);
      VReal v2;
      SampleV(train.item[j], h1, *rbm, &v2);
      VReal h2;
      ExpectH(train.item[j], v2, *rbm, &h2);
      Update(train.item[j], h1, train.rating[j], h2, v2, rbm);
      LOG_IF(INFO, j % 500 == 0) << i << " " << RBMTest(train, test, *rbm)
           << " " << RBMTest(train, train, *rbm) << ":" << time.GetTime();
    }
  }
}
}  // namespace ml
