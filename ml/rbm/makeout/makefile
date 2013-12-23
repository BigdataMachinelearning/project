BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/rbm

name:=rbm_unittest
$(name)_update_dep:=\
  rbm.o\
  util.o\
  rbm2.o\
  rbm_util.o
$(name)_remain_dep:=\
  join.o\
  base.o\
  string_util.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=rbm_app
$(name)_update_dep:=\
  rbm2.o\
  rbm.o\
  util.o\
  rbm_util.o
$(name)_remain_dep:=\
  join.o\
  base.o\
  string_util.o\
  $(GTestDir)/gtest-all.o
include post_app_makefile

name:=rbm_unittest
name:=rbm_app
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
