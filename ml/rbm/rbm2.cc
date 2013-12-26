#include "base/base_head.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"
#include "ml/rbm/rbm2.h"
namespace ml {
using Eigen::MatrixXd;
using Eigen::VectorXd;
RBM::RBM(const SpMat &train, int nv, int nh, int nsoftmax){
   W.resize(nsoftmax);
   dW.resize(nsoftmax);
   for(size_t i=0; i<W.size(); ++i){
     W[i].resize(nh, nv);
     dW[i].resize(nh, nv);
     RandomInit(&W[i]);
   }
   VVVReal tmp;
   Convert(W, &tmp);
   LOG(INFO) << Var(tmp);
   LOG(INFO) << Mean(tmp);
   bv.resize(nsoftmax, nv);
   dv.resize(nsoftmax, nv);
   RandomInit(&bv);
   bh.resize(nh);
   dh.resize(nh);
   bh.setZero();
   v0.resize(nv);
   h0.resize(nh);
   vk.resize(nv);
   hk.resize(nh);
}

void RBM::Expectv(const VectorXd &h, const SpVec &t, SpVec *v) {
  VReal a(W.size());
  VReal b(W.size());
  v->setZero();
  for (SpVec::InnerIterator it(t); it; ++it){
    int i = it.index();
    for(size_t k = 0; k < W.size(); ++k) {
      a[k] = bv(k, i) +  W[k].col(i).dot(h);
    }
    ml::Softmax(a, &b);
    v->insert(i) = ExpectSoftmax(b);
  }
}

void RBM::Samplev(const VectorXd &h, const SpVec &t, SpVec *v) {
  VReal a(W.size());
  VReal b(W.size());
  v->setZero();
  for (SpVec::InnerIterator it(t); it; ++it){
    int i = it.index();
    for(size_t k = 0; k < W.size(); ++k) {
      a[k] = bv(k, i) +  W[k].col(i).dot(h);
    }
    ml::Softmax(a, &b);
    v->insert(i) = ml::Sample(b);
  }
}

void RBM::Expecth(const SpVec &v, VectorXd *h) {
  double s;
  for (int j = 0; j < h->rows(); ++j) {
    s = 0;
    for (SpVec::InnerIterator it(v); it; ++it) {
      s += W[it.value() - 1](j, it.index());
    }
    s += bh[j];
    (*h)[j] = Sigmoid(s);
  }
}

void RBM::Sampleh(const SpVec &v, VectorXd *h){
  Expecth(v, h);
  for (int i = 0; i < h->size(); ++i) {
    (*h)[i] = Sample1(h[0][i]);
  }
}

void RBM::PartGrad(const SpVec &v, const VectorXd &h, const double &coeff){
  for(SpVec::InnerIterator it(v); it; ++it){
    dv(it.value() - 1, it.index()) += coeff;
    for (int j = 0; j < h.rows(); ++j) {
      dW[it.value() - 1](j, it.index()) += coeff * h[j];
    }
  }
  for (int j = 0; j < h.rows(); ++j) {
    dh(j) += coeff * h[j];
  }
}

void RBM::InitGradient(){
  for(size_t k=0; k<dW.size(); ++k) {
    dW[k].setZero();
  }
  dh.setZero();
  dv.setZero();
}

void RBM::UpdateGradient(double alpha, int batch_size) {
  double r = alpha / batch_size;
  for(size_t k = 0; k < dW.size(); ++k) {
    W[k] += r * dW[k];
  }
  bh += r * dh;
  bv += r * bv;
}

void RBM::Gradient(const SpVec &x, int step) {
  Sampleh(x, &h0);
  Samplev(h0, x, &vk);
  for (int k = 0; k < step - 1; ++k) {
    Sampleh(vk, &hk);
    Samplev(hk, x, &vk);
  }
  Expecth(vk, &hk);
  PartGrad(x,  h0, 1);
  PartGrad(vk, hk, -1);
}

void RBM::Train(const SpMat &train, const SpMat &test, int niter, double alpha,
                                                       int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 0;
  for (int i = 0; i < niter; ++i) {
    Count.clear();
    if (i % 50 == 0) {
      nCD++;
    }
    for (int n = 0; n < train.cols(); n++) {
      Gradient(train.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    LOG(INFO) << i << " " << 
              Predict(train, train) << " " << Predict(train, test);
  }
}

double RBM::Predict(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    Expecth(train.col(n), &h0);
    Expectv(h0, test.col(n), &v0);
    v0 -= test.col(n);
    rmse += v0.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}
} // namespace ml
