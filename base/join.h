// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_JOIN_H_
#define BASE_JOIN_H_
#include "base/string_util.h"

#include "base/base.h"

template <typename Iter>
inline Str JoinStr(Iter beg, Iter end, const Str &del) {
  Str str;
  for (Iter it = beg; it != end; ++it) {
    str.append(*it);
    str.append(del);
  }
  return str;
}

Str Join(const  VStr &vec, const Str &del);
Str Join(const VVStr &vec, const Str &del1, const Str &del2);
Str Join(const VReal &data, const Str &del);
Str Join(double* str, int len1);
Str Join(double** str, int len1, int len2);
#endif // BASE_JOIN_H_
