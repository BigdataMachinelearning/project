// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_MATRIX_FACTORIZATION_
#define ML_MATRIX_FACTORIZATION_
#include "base/base_head.h"
#include "ml/eigen.h"

namespace ml {
struct MF {
  EMat u;
  EMat v;
  EVec bu;
  EVec bv;
};

inline void RandomInit(int u, int v, int k, MF* mf) {
  mf->u.resize(k, u);
  mf->v.resize(k, v);
  mf->bu.resize(u);
  mf->bv.resize(v);
}

inline double Test(const SpMat &mat, const EMat &u, const EMat &v) {
  double rmse = 0;
  for (int m = 0; m < mat.cols(); ++m) {
    for (SpMatInIt it(mat, m); it; ++it) {
      double a = u.col(m).dot(v.col(it.index()));
      a = a < 1 ? 1 : a;
      a = a > 5 ? 5 : a;
      rmse += Square(a - it.value());
    }
  }
  return std::sqrt(rmse / mat.nonZeros());
}

inline double Test(const SpMat &mat, const MF &mf) {
  double rmse = 0;
  for (int m = 0; m < mat.cols(); ++m) {
    for (SpMatInIt it(mat, m); it; ++it) {
      double a = mf.u.col(m).dot(mf.v.col(it.index())) + mf.bu(m) +
                                             mf.bv(it.index());
      a = a < 1 ? 1 : a;
      //a = a > 5 ? 5 : a;
      a = a > 2 ? 2 : a;
      rmse += Square(a - it.value());
    }
  }
  return std::sqrt(rmse / mat.nonZeros());
}

inline void SGD(int it_num, double eta, double lambda, const SpMat &u_v,
                            const SpMat &test, EMat* u, EMat* v) {
  for (int i = 0; i < it_num; ++i) {
    for (int j = 0; j < u->cols(); ++j) {
      for (SpMatInIt it(u_v, j); it; ++it) {
        EVec pu = u->col(j);
        double e = pu.dot(v->col(it.index())) - it.value();
        u->col(j) -= eta * (e * v->col(it.index()) + lambda * pu);
        v->col(it.index()) -= eta * (e * pu + v->col(it.index()) * lambda);
      }
    }
    LOG_IF(INFO, i % 100 == 0) << i << "  " << Test(u_v, *u, *v) <<
                               ":" << Test(test, *u, *v);
  }
}

inline void BGD(double eta, double lambda, const SpMat &mat, const EMat &v,
                                                             EMat* u) {
  for (int m = 0; m < u->cols(); ++m) {
    EVec grd(u->col(m).size());
    grd.setZero();
    double s = 0;
    for (SpMatInIt it(mat, m); it; ++it) {
      grd += (u->col(m).dot(v.col(it.index())) - it.value())
             * v.col(it.index());
      s++;
    }
    grd += lambda * u->col(m) * s;
    u->col(m) -= eta * grd / s;
  }
}

inline void BGD(int it_num, double eta, double lambda, const SpMat &u_v,
                            const SpMat &test, EMat* u, EMat* v) {
  SpMat v_u = u_v.transpose();
  for (int i = 0; i < it_num; i++) {
    EMat tmp_u(*u);
    BGD(eta, lambda, u_v, *v, u);
    BGD(eta, lambda, v_u, tmp_u, v);
    LOG_IF(INFO, i % 100 == 0) << i << "  " << Test(u_v, *u, *v) <<
                               ":" << Test(test, *u, *v);
  }
}

inline void SGD(int it_num, double eta, double lambda, const SpMat &rating,
                            const SpMat &test, MF* mf) {
  for (int i = 0; i < it_num; ++i) {
    for (int j = 0; j < mf->u.cols(); ++j) {
      for (SpMatInIt it(rating, j); it; ++it) {
        EVec pu = mf->u.col(j);
        const int &k = it.index();
        double e = pu.dot(mf->v.col(k)) + mf->bu[j] + mf->bv[k] - it.value();
        mf->u.col(j) -= eta * (e * mf->v.col(k) + lambda * pu);
        mf->bu[j] -= eta * (e + lambda * mf->bu[j]);
        mf->v.col(k) -= eta * (e * pu + mf->v.col(k) * lambda);
        mf->bv[k] -= eta * (e + lambda * mf->bv[k]);
      }
    }
    LOG_IF(INFO, i % 100 == 0) << i << "  " << Test(rating, *mf) <<
                               ":" << Test(test, *mf);
  }
}
} // namespace ml
#endif // ML_MATRIX_FACTORIZATION_
