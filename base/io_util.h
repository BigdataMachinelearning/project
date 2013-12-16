// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_IO_UTIL_H_
#define BASE_IO_UTIL_H_
#include <fstream>
#include "base/string_util.h"

inline void ReadFileToStr(const Str &file, Str* str) {
  std::ifstream in(file.c_str());
  std::istreambuf_iterator<char> beg(in);
  std::istreambuf_iterator<char> end;
  str->assign(beg, end);
  in.close();
}

inline void ReadFileToStr(const Str &file, const Str &del, VStr* data) {
  Str str;
  ReadFileToStr(file, &str);
  SplitStr(str, del, data);
}

inline Str ReadFileToStr(const Str &file) {
  Str str;
  ReadFileToStr(file, &str);
  return str;
}

inline void WriteStrToFile(const Str &str, const Str &file) {
  std::ofstream o(file.c_str());
  o << str;
  o.close();
}

/*
inline void ReadFile(const Str &file, int size, char* des) {
  std::ofstream out(file.c_str(), std::ios::binary);
  out.write((char*)(&a[0]), sizeof(a));
  out.close();

  VInt v2(2);
  std::ifstream in("test", std::ios::binary);
  in.read((char*)(&v2[0]), sizeof(v2));
  in.close( );
  LOG(INFO) << Join(v2, " ");
  LOG(INFO) << sizeof(a);
  LOG(INFO) << sizeof(v2);
}
*/
#endif  // BASE_IO_UTIL_H_
