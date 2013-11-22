// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/join.h"

Str Join(const VStr &vec, const Str &del) {
  return JoinStr(vec.begin(), vec.end(), del);		   
}

Str Join(const VVStr &vec, const Str &del1, const Str &del2) {
  VStr tmp;
  for (VVStr::size_type i = 0; i < vec.size(); i++) {
    tmp.push_back(Join(vec.at(i), del1));
  }
  return Join(tmp, del2);
}

Str Join(const VReal &data, const Str &del) {
  VStr tmp;
  for (VReal::size_type i = 0; i < data.size(); i++) {
    tmp.push_back(ToStr(data[i]));
  }
  return Join(tmp, " ");
}

Str Join(double* str, int len) {
  VStr tmp;
  for (int i = 0; i < len; i++) {
    tmp.push_back(ToStr(str[i]));
  }
  return Join(tmp, " ");
}

Str Join(double** str, int len1, int len2) {
  VStr tmp;
  for (int i = 0; i < len1; i++) {
    tmp.push_back(Join(str[i], len2));
  }
  return Join(tmp, "\n");
}
