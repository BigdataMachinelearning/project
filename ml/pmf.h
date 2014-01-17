#ifndef ML_RBM_PMF_
#define ML_RBM_PMF_
#include "ml/eigen.h"

namespace ml {
class PMF {
 public:
inline void Gradient(const SpMat &mat, int m, const EMat &v, EVec* grd);
inline void Gradient(const SpMat &mat, EMat* u, EMat* v);
inline void Learning(const SpMat &train, const SpMat &test, EMat* u, EMat* v);
inline void Test(const SpMat &mat, EMat* u, EMat *v);
 private:
  double lambda_u;
  double lambda_v;
  double eta;
  int it_num;
}

void PMF::Gradient(const SpMat &mat, EMat* u, EMat* v) {
  for (int m = 0; m < u->cols(); ++m) {
    EVec grd;
    Grdient(mat, m, *v, &grd);
    u->col(m) -= eta * grd;
  }
  for (int m = 0; m < v->cols(); ++m) {
    EVec grd;
    Grdient(mat, m, *u, &grd);
    v->col(m) -= eta * grd;
  }
}

void PMF::Gradient(const SpMat &mat, int m, const EMat &v, EVec* grd) {
  EMat sum(u.size(), u.size());
  for (SpVecInIt it(mat.col(m)); it; ++it) {
    sum += v->col(it->index()) * v->col(it->index()).transpose();
  }
  (*grd) = sum * mat.col(m);
  for (SpVecInIt it(mat.col(m)); it; ++it) {
    (*grd) -= it->value() * v->col(it->index());
  }
  (*grd) = lambda_u * u;
}

void PMF::Learning(const SpMat &train, const SpMat &test, EMat* u, EMat* v) {
  for (int i = 0; i < it_num; i++) {
    Gradient(mat, u, v);
    Test(test, u, v);
  }
}

void PMF::Test(const SpMat &mat, EMat* u, EMat *v) {
  double rmse = 0;
  for (int m = 0; m < u->cols(); ++m) {
    for (SpVecInIt it(mat.col(m)); it; ++it) {
      rmse += Square(u->col(m).transpose() * v->col(it->index() - it->value());
    }
  }
  return std::sqrt(rmse / mat.nonZeros());
}
} // namespace ml
#endif // ML_RBM_PMF_
