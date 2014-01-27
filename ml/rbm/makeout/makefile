BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/rbm

name:=rbm_unittest
$(name)_update_dep:=\
  rbm2.o\
  rbm_util.o\
  eigen.o\
  rbm.o
$(name)_remain_dep:=\
  repsoftmax.o\
  probability.o\
  base.o\
  document.o\
  util.o\
  join.o\
  string_util.o\
  ais.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=rbm_app
$(name)_update_dep:=\
  rbm_util.o\
  rbm.o
$(name)_remain_dep:=\
  ais.o\
  rbm2.o\
  join.o\
  eigen.o\
  util.o\
  repsoftmax.o\
  probability.o\
  string_util.o\
  document.o\
  base.o\
  $(GTestDir)/gtest-all.o
include post_app_makefile

name:=rbm_unittest
name:=rbm_app
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
