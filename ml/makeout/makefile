BaseDir:=../..
include build_makefile
WorkDir:=$(BaseDir)/ml

name:=document_unittest
$(name)_update_dep:=\
  base.o\
  document.o\
  join.o\
  string_util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=util_unittest
$(name)_update_dep:=\
  base.o\
  join.o\
  string_util.o\
  util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=document_unittest
name:=util_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
