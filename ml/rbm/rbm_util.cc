// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/rbm/rbm_util.h"

#include "ml/rbm/rbm.h"
#include "base/base_head.h"
namespace ml {
void LoadMovieLen(const Str &name, User* user) {
  Str str;
  ReadFileToStr(name, &str);
  VStr vec;
  SplitStr(str, "\n", &vec);
  for(size_t i = 0; i < vec.size(); i++) {
    VStr terms;
    SplitStr(vec.at(i), " ", &terms);
    if (terms.size() < 2) {
      terms.clear();
      SplitStr(vec.at(i), "\t", &terms);
    }
    if(terms.size() >= 3) {
      size_t id = static_cast<size_t>(StrToInt(terms[0]));
      if(id >= user->item.size()) {
        user->item.resize(id + 1);
        user->rating.resize(id + 1);
      }
      user->item.at(StrToInt(terms[0])).push_back(StrToInt(terms[1]));
      user->rating.at(StrToInt(terms[0])).push_back(StrToInt(terms[2]));
    }
  }
}

void LoadBaidu(const Str &name, User* user) {
  Str str;
  ReadFileToStr(name, &str);
  VStr vec;
  SplitStr(str, "\n", &vec);
  user->item.resize(vec.size() + 1);
  user->rating.resize(vec.size() + 1);
  for(VStr::size_type i = 0; i < vec.size(); i++) {
    VStr terms;
    SplitStr(vec.at(i), " ", &terms);
    int id = StrToInt(terms[0]);
    for(VStr::size_type j = 1; j < terms.size(); j++) {
      VStr tmp;
      SplitStr(terms.at(j), ":", &tmp);
      (user->item)[id].push_back(StrToInt(tmp[0]));
      (user->rating)[id].push_back(StrToInt(tmp[1]));
    }
  }
}

void SaveBaidu(const Str &name, const User &user) {
  std::list<Str> tmp;
  for (size_t i = 1; i < user.item.size(); i++) {
    LOG_IF(INFO, i % 100 == 0) << i;
    for (size_t j = 0; j < user.item[i].size(); j++) {
      VReal tmp2;
      tmp2.push_back(i);
      tmp2.push_back(user.item[i][j]);
      tmp2.push_back(user.rating[i][j]);
      tmp.push_back(Join(tmp2, " "));
    }
  }
  LOG(INFO) << tmp.size();
  WriteStrToFile(Join(tmp, "\n"), name);
}

void SplitData(const User &user, double value, User* train, User* test) {
  train->item.resize(user.item.size());
  train->rating.resize(user.item.size());
  test->item.resize(user.item.size());
  test->rating.resize(user.item.size());
  for(size_t i = 0; i < user.item.size(); i++) {
    for(VVInt::size_type j = 0; j < user.item[i].size(); ++j) {
      if(Random1() < value) {
        (train->item)[i].push_back(user.item[i][j]);
        (train->rating)[i].push_back(user.rating[i][j]);
      } else {
        (test->item)[i].push_back(user.item[i][j]);
        (test->rating)[i].push_back(user.rating[i][j]);
      }
    }
  }
}

size_t Size(const VVInt &item) {
  size_t sum = 0;
  for(size_t i = 0; i < item.size(); i++) {
    sum += item[i].size();
  }
  return sum;
}

void LoadBaidu(const Str &name, double pro, User* train, User* test) {
  User user;
  LoadBaidu(name, &user);
  SplitData(user, pro, train, test);
}

double SquareError(const VReal &lhs, const VReal &rhs) {
  double sum = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    sum += Square(lhs[i] - rhs[i]);
  }
  return sum;
}

double RBMTest(const User &train, const User &test, const ml2::RBM &rbm) {
  double sum = 0;
  for (size_t i = 1; i < test.item.size(); ++i) {
    VReal h1;
    ml2::ExpectH(train.item[i], train.rating[i], rbm, &h1);
    VVReal predict;
    ml2::ExpectV(test.item[i], h1, rbm, &predict);
    VReal v2;
    for (VVReal::size_type j = 0; j < predict.size(); ++j) {
      v2.push_back(Expect(predict[j]));
    }
    sum += SquareError(test.rating[i], v2);
  }
  return std::sqrt(sum / Size(test.item));
}

void ReadData(const Str &path, int rows, int cols, SpMat *mat) {
  FILE *fin = fopen(path.c_str(), "r");
  int u;
  int v;
  float r;
  int m = 0;
  int n = 0;
  std::vector<T> tripletList;
  while(fscanf(fin, "%d %d %f", &u, &v, &r) > 0) {
    tripletList.push_back(T(v, u, r));
    m = u > m ? u : m;
    n = v > n ? v : n;
  }
  if (rows == 0) { 
    rows = n + 1;
  }
  if (cols == 0) {
    cols = m + 1;
  }
  mat->resize(rows, cols);
  mat->setFromTriplets(tripletList.begin(), tripletList.end());
}

int MaxItemId(const User &user) {
  int id = -1;
  for (size_t i = 0; i < user.item.size(); i++) {
    for (size_t j = 0; j < user.item[i].size(); j++) {
      if (user.item[i][j] > id) {
        id = user.item[i][j];
      }
    }
  }
  return id;
}

void RandomInit(MatrixXd* des) {
  for(int i = 0; i < des->rows(); ++i) {
    for(int j = 0; j< des->cols(); ++j) {
      des[0](i, j) = NormalSample() / 100;
    }
  }
}

void Convert(const std::vector<MatrixXd> &src, VVVReal* des) {
  Init(src.size(), src[0].rows(), src[0].cols(), 0.0, des);
  for (size_t i = 0; i < src.size(); ++i) {
    for(int j = 0; j < src[i].rows(); ++j) {
      for(int k = 0; k < src[i].cols(); ++k) {
        (*des)[i][j][k] = src[i](j, k);
      }
    }
  }
}
}  // namespace ml
