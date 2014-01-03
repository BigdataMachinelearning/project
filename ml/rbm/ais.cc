// Copyright 2013 yuanwujun, All Rights Reserved.
// Author: real.yuanwj@gmail.com
#include "ml/rbm/repsoftmax.h"
#include "ml/rbm/ais.h"

#include "ml/document.h"
#include "ml/rbm/rbm_util.h"
#include "ml/info.h"
#include "ml/util.h"

#include <cmath>

namespace ml {
void SampleV(const VReal &h,const int wn,const RepSoftMax &rbm,const Real belt,VReal* v) {
  VReal expect(wn);
  VReal arr(v->size());
  for (size_t k = 0; k < expect.size(); ++k) {
    arr[k] = 0;
    for (size_t f = 0; f < h.size(); ++f) {
      arr[k] += rbm.w[f][k] * h[f] * belt;
    }
    arr[k] += rbm.b[k] * belt;
  }
  ml::Softmax(arr, &expect);
  
  for (int i = 0; i < wn; ++i) {
    v->at(Random(expect))++;
  }
}

void SampleH(const VReal& wl,const int wn,const RepSoftMax &rbm, const Real belt, VReal* h) {
  h->resize(rbm.c.size());
  for (size_t f = 0; f < h->size(); f++) {
    double sum = 0.0;
    for (size_t w = 0; w < wl.size(); ++w) {
      sum += rbm.w[f][w] * wl[w] * belt;
    }
    sum += rbm.c[f] * wn * belt;
    h->at(f) = Sigmoid(sum);
  }
  
  for (size_t i = 0; i < h->size(); i++) {
    (*h)[i] = Sample1((*h)[i]);
  }
}

void UniformSample(VReal& v,const RepSoftMax& rbm) {
  const int base = 2;
  for( size_t i = 0; i < v.size(); ++i ) {
    v[i] = random() % static_cast<long int>(powl(base, rbm.c.size()));
  }
}

double CalculateP(const VReal& v,const Real belt,const ml::RepSoftMax& rbm) {
  const int fhidden = rbm.c.size();
  const int kwords = rbm.b.size();
  double result = 1;
  for(int f = 0; f < fhidden; ++f) {
    double sum = 0.0;
    for(int k = 0; k < kwords; ++k) {
      sum += rbm.w[f][k] * v[k];
    }
    double factor = exp(sum * belt) + 1;
    result *= factor;
  }
  return result;
}

double AISEstimate(const int runs, const VReal& belts,const ml::RepSoftMax& rbm) {
  const int ncorpus = rbm.b.size();
  VVReal vtransition(belts.size());
  for(size_t i = 0; i < belts.size(); ++i) {
    vtransition[i].resize(ncorpus);
  }
  VReal wais;
  Init(runs, 1, &wais);
  for(int k = 0; k < runs; ++k) {
    UniformSample(vtransition[0],rbm);
    for(size_t i = 0; i < belts.size() - 1; ++i) {
      VReal h;
      SampleH(vtransition[i],ncorpus,rbm,belts[i],&h);
      SampleV(h,ncorpus,rbm,belts[i],&vtransition[i+1]);
    }
    for(size_t i = 1; i < belts.size(); ++i ) {
      double pratio = CalculateP(vtransition[i],belts[i],rbm) / 
                      CalculateP(vtransition[i],belts[i - 1],rbm);
      wais[k] *= pratio;
    }
    LOG(INFO) << wais[k];
  }
  return std::accumulate(wais.begin(), wais.end(), 0.0) / runs;
}
} // namespace ml
