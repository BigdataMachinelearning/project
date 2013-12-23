#ifndef ML_RBM_RBM2_
#define ML_RBM_RBM2_
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
using namespace std;
namespace ml {
using Eigen::MatrixXd;
using Eigen::VectorXd;
class RBM {
 public:
  RBM(const SpMat &train, int nv, int nh, int nsoftmax);
  void Train(const SpMat &train, const SpMat &test, int niter,
                                 double alpha, int batch_size);
  double Predict(const SpMat &train, const SpMat &test);
 public:
  SpVec v0, vk;
  VectorXd h0, hk;
 private:
  std::vector<MatrixXd> W, dW;
  MatrixXd bv, dv;
  VectorXd bh, dh;
  void InitGradient();
  void UpdateGradient(double alpha, int batch_size);
  void Expecth(const SpVec &v, VectorXd *h);
  void Sampleh(const SpVec &v, VectorXd *h);
  void Expectv(const VectorXd &h, const SpVec &t, SpVec *v);
  void Samplev(const VectorXd &h, const SpVec &t, SpVec *v);
  void PartGrad(const SpVec &v, const VectorXd &h, const double &coeff);
  void Gradient(const SpVec &x, int step);
};
} // namespace ml
#endif // ML_RBM_RBM2_
