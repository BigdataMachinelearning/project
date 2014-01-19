#ifndef ML_PMF_
#define ML_PMF_
#include "base/base_head.h"
#include "ml/eigen.h"

namespace ml {
class PMF {
 public:
  PMF(double eta_, double lambda_) :lambda(lambda_), eta(eta_) {}
  inline void Gradient(const SpMat &mat, const EMat &u, EMat* v) const;
  inline void Gradient2(const SpMat &mat, const EMat &u, EMat* v) const;
  inline void Learning(int it_num, const SpMat &u_v, const SpMat &test, EMat* u,
                                   EMat* v);
  inline void Gradient3(const SpMat &mat, EMat* v, EMat* u) const;
  inline double Test(const SpMat &mat, const EMat &u, const EMat &v) const;
 private:
  double lambda;
  double eta;
};

void PMF::Gradient(const SpMat &mat, const EMat &v, EMat* u) const {
  for (int m = 0; m < u->cols(); ++m) {
    EVec grd(u->col(m).size());
    grd.setZero();
    double s = 0;
    for (SpMatInIt it(mat, m); it; ++it) {
      grd += (u->col(m).transpose() * v.col(it.index()) - it.value())
             * v.col(it.index());
      s++;
    }
    grd += lambda * u->col(m);
    u->col(m) -= eta * grd / s;
  }
}

void PMF::Gradient2(const SpMat &mat, const EMat &v, EMat* u) const {
  for (int i = 0; i < u->cols(); ++i) {
    for (SpMatInIt it(mat, i); it; ++it) {
      // U.col(u) += alpha * (e*pv - lambda*pu);
      //V.col(v) += alpha * (e*pu - lambda*pv);
      double e = u->col(i).dot(v.col(it.index())) - it.value();
      u->col(i) -= eta * e * v.col(it.index());
    }
  }
}

void PMF::Gradient3(const SpMat &mat, EMat* v, EMat* u) const {
  for (int i = 0; i < u->cols(); ++i) {
    for (SpMatInIt it(mat, i); it; ++it) {
      EVec pu = u->col(i);
      double e = pu.dot(v->col(it.index())) - it.value();
      u->col(i) -= eta * e * v->col(it.index());
      v->col(it.index()) -= eta * e * pu;
    }
  }
}

void PMF::Learning(int it_num, const SpMat &u_v, const SpMat &test, EMat* u,
                                                 EMat* v) {
  v->setRandom();
  u->setRandom();
  // SpMat v_u = u_v.transpose();
  for (int i = 0; i < it_num; i++) {
    // EMat tmp(*u);
    //Gradient2(u_v, *v, u);
    //Gradient2(v_u, tmp, v);
    Gradient3(u_v, v, u);
    eta *= 0.99;
    LOG(INFO) << i << "  " << Test(u_v, *u, *v) << ":" << Test(test, *u, *v);
  }
}

double PMF::Test(const SpMat &mat, const EMat &u, const EMat &v) const {
  double rmse = 0;
  for (int m = 0; m < mat.cols(); ++m) {
    for (SpMatInIt it(mat, m); it; ++it) {
      double a = Sigmoid(u.col(m).transpose() * v.col(it.index()));
      if (a < 0.2) {
        a = 0.2;
      }
      rmse += Square(a * 5 - it.value());
    }
  }
  return std::sqrt(rmse / mat.nonZeros());
}
} // namespace ml
#endif // ML_PMF_
