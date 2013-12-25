// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"
namespace ml2 {
void RBM::Init(int f, int m, int k, double momentum_, double eta_) {
  ml::RandomInit(f, m, k, &w1);
  ml::RandomInit(m, k, &b1);
  ml::RandomInit(f, &c1);
  InitZero();
  momentum = momentum_;
  eta = eta_;
  LOG(INFO) << ml::Var(w1);
  LOG(INFO) << ml::Mean(w1);
}

void RBM::InitZero() {
  dw.clear();
  db.clear();
  dc.clear();
  ::Init(w1.size(), w1[0].size(), w1[0][0].size(), 0.0, &dw);
  ::Init(b1.size(), b1[0].size(), 0.0, &db);
  ::Init(c1.size(), 0.0, &dc);
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

void Update(RBM* rbm) {
  double eta = rbm->eta / rbm->bach_size;
  for (size_t m = 0; m < rbm->w1[0].size(); m++) {
    for (size_t k = 0; k < rbm->w1[0][0].size(); k++) {
      rbm->b1[m][k] += eta * rbm->db[m][k];
      for (size_t f = 0; f < rbm->w1.size(); f++) {
        rbm->w1[f][m][k] += eta * rbm->dw[f][m][k];
      }
    }
  }
  for (VInt::size_type i = 0; i < rbm->c1.size(); i++) {
    rbm->c1[i] += eta * rbm->dc[i];
  }
}

void Gradient(const VInt &item, const VReal &h1, const VReal &v1,
              const VReal &h2, const VReal &v2, RBM* rbm) {
  for (size_t m = 0; m < item.size(); m++) {
    if (v1[m] != v2[m]) {
      rbm->db[item[m]][static_cast<int>(v1[m] - 1)] += 1;
      rbm->db[item[m]][static_cast<int>(v2[m] - 1)] -= 1;
    }
    for (size_t f = 0; f < h1.size(); f++) {
      if (v1[m] == v2[m]) {
        rbm->dw[f][item[m]][static_cast<int>(v1[m] - 1)] += h1[f] - h2[f];
      } else {
        rbm->dw[f][item[m]][static_cast<int>(v1[m] - 1)] += h1[f];
        rbm->dw[f][item[m]][static_cast<int>(v2[m] - 1)] -= h2[f];
      }
    }
  }
  for (VInt::size_type i = 0; i < h1.size(); i++) {
    rbm->dc[i] += h1.at(i) - h2.at(i);
  }
}

void ExpectV(const VInt &item, const VReal &h, const RBM &rbm, VVReal* v) {
  v->resize(item.size());
  for (size_t i = 0; i < item.size(); ++i) {
    VReal arr(rbm.w1[0][0].size());
    for (size_t k = 0; k < arr.size(); k++) {
      arr[k] = 0;
      for (size_t f = 0; f < h.size(); f++) {
        arr[k] += rbm.w1[f][item[i]][k] * h[f];
      }
      arr[k] += rbm.b1[item[i]][k];
    }
    v->at(i).resize(arr.size());
    ml::Softmax(arr, &(v->at(i)));
  }
}

void SampleV(const VInt &item, const VReal &h, const RBM &rbm, VReal* v) {
  VVReal expect;
  ExpectV(item, h, rbm, &expect);
  v->resize(item.size());
  for (VVReal::size_type i = 0; i < expect.size(); ++i) {
    // v->at(i) = Random(expect[i]) + 1;
    int x = Random(expect[i]) + 1;
    Count2[x]++;
    v->at(i) = x;
  }
}

void ExpectH(const VInt &item, const VReal &rating, const RBM &rbm, VReal* h) {
  h->resize(rbm.c1.size());
  for (size_t f = 0; f < h->size(); f++) {
    double sum = 0.0;
    for (size_t m = 0; m < rating.size(); ++m) {
      sum += rbm.w1[f][item[m]][rating[m] - 1];
    }
    sum += rbm.c1[f];
    h->at(f) = Sigmoid(sum);
  }
}

void SampleH(const VInt &item, const VReal &rating, const RBM &rbm, VReal* h) {
  ExpectH(item, rating, rbm, h);
  for (size_t i = 0; i < h->size(); i++) {
    h->at(i) = Sample1(h->at(i));
    if (h->at(i) == 0) {
      Count[0]++;
    } else {
      Count[1]++;
    }
  }
}

void RBMLearning(const User &train, const User &test, int iter_num, RBM* rbm) {
  for (int i = 0; i < iter_num; i++) {
    Time time;
    time.Start();
    for (size_t j = 1; j < train.item.size(); j++) {
      LOG_IF(INFO, j % 500 == 1) << i << " " << ml::RBMTest(train, test, *rbm)
           << " " << ml::RBMTest(train, train, *rbm) << " " << time.GetTime();
      VReal h1;
      SampleH(train.item[j], train.rating[j], *rbm, &h1);
      VReal v2;
      SampleV(train.item[j], h1, *rbm, &v2);
      VReal h2;
      ExpectH(train.item[j], v2, *rbm, &h2);
      Update(train.item[j], h1, train.rating[j], h2, v2, rbm);
    }
  }
}

void RBMLearning(const User &train, const User &test, int iter_num, 
                                    int bach_size, RBM* rbm) {
  int sample = 0;
  for (int i = 0; i < iter_num; i++) {
    Time time;
    time.Start();
    Count.clear();
    Count2.clear();
    for (size_t j = 1; j < train.item.size(); j++) {
      sample++;
      VReal h1;
      SampleH(train.item[j], train.rating[j], *rbm, &h1);
      // ExpectH(train.item[j], train.rating[j], *rbm, &h1);
      VReal v2;
      SampleV(train.item[j], h1, *rbm, &v2);
      VReal h2;
      ExpectH(train.item[j], v2, *rbm, &h2);
      Gradient(train.item[j], h1, train.rating[j], h2, v2, rbm);
      if (sample == bach_size) {
        Update(rbm);
        rbm->InitZero();
        sample = 0;
      }
    }
    // LOG(INFO) << ml::Var(rbm->w1) << " " << ml::Var(rbm->dw) << "--"
      //        << ml::Mean(rbm->w1) << " " << ml::Mean(rbm->dw);
    double error = ml::RBMTest(train, test, *rbm);
    if (error < 0.96) {
      rbm->eta = 0.01;
    }
    if (error < 0.95) {
      rbm->eta = 0.005;
    }
    LOG(INFO) << i << ":"  << error 
        << " " << ml::RBMTest(train, train, *rbm) << " " << time.GetTime();
    // LOG(INFO) << JoinValue(Count2.begin(), Count2.end());
    // LOG(INFO) << i << " " << Count[0] << " " << Count[1];
  }
}
// test 
}  // namespace ml
