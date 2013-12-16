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
  for(VStr::size_type i = 0; i < vec.size(); i++) {
    VStr terms;
    SplitStr(vec.at(i), "\t", &terms);
    if(terms.size() == 4) {
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

double SoftMax(const VReal &data, double value) {
  VReal tmp(data.size());
  for(VReal::size_type i = 0; i < data.size(); i++) {
    tmp.at(i) = exp(data.at(i) - value);
  }
  return 1 / std::accumulate(tmp.begin(), tmp.end(), 0.0);
}

void SoftMax(const VReal &data, VReal* des) {
  des->resize(data.size());
  for(VReal::size_type i = 0; i < data.size(); i++) {
    des->at(i) = SoftMax(data, data.at(i));
  }
}

double SquareError(const VReal &lhs, const VReal &rhs) {
  double sum = 0;
  for (VReal::size_type i = 0; i < lhs.size(); ++i) {
    sum += Square(lhs[i] - rhs[i]);
  }
  return sum;
}

double RBMTest(const User &train, const User &test, const RBM &rbm) {
  double sum = 0;
  for (size_t i = 1; i < test.item.size(); ++i) {
    VReal h1;
    ExpectH(train.item[i], train.rating[i], rbm, &h1);
    VVReal predict;
    ExpectV(test.item[i], h1, rbm, &predict);
    VReal v2;
    for (VVReal::size_type j = 0; j < predict.size(); ++j) {
      v2.push_back(Expect(predict[j]));
    }
    sum += SquareError(test.rating[i], v2);
  }
  return std::sqrt(sum / Size(test.item));
}
}  // namespace ml
