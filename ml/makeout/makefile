BaseDir:=../..
include build_makefile
WorkDir:=$(BaseDir)/ml

name:=document_unittest
$(name)_update_dep:=\
  base.o\
  document.o\
  join.o\
  util.o\
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

name:=info_unittest
$(name)_update_dep:=\
  base.o\
  join.o\
  string_util.o\
  util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=eigen_unittest
$(name)_update_dep:=\
  util.o\
  eigen.o
$(name)_remain_dep:=\
  base.o\
  join.o\
  string_util.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=eigen_app
$(name)_update_dep:=\
  util.o\
  eigen.o
$(name)_remain_dep:=\
  base.o\
  join.o\
  string_util.o
include post_app_makefile

name:=info_unittest
name:=util_unittest
name:=document_unittest
name:=eigen_unittest
name:=eigen_app
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
