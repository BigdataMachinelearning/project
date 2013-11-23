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
Str Join(const VVReal &data, const Str &del, const Str &del2);
Str Join(const VVVReal &data, StrC &del1, StrC &del2, StrC &del3);
Str Join(double* str, int len1);
Str Join(double** str, int len1, int len2);

template <typename T>
inline Str MapToStr(T beg, T end) {
  VStr vec1;
  for (T it = beg; it != end; ++it) {
    VStr vec2;
    vec2.push_back(ToStr(it->first));
    vec2.push_back(ToStr(it->second));
    vec1.push_back(Join(vec2, " "));
  }
  return Join(vec1, "\n");
}

inline Str MapToStr(const MIntInt &src) {
  return MapToStr(src.begin(), src.end());
}
#endif // BASE_JOIN_H_
