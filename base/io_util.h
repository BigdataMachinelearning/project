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
#endif // BASE_IO_UTIL_H_
