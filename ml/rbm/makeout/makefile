BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/rbm

name:=rbm_unittest
$(name)_update_dep:=\
  repsoftmax.o\
  probability.o\
  ais.o
$(name)_remain_dep:=\
  rbm.o\
  base.o\
  rbm_util.o\
  document.o\
  util.o\
  rbm2.o\
  join.o\
  string_util.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=rbm_app
$(name)_update_dep:=\
  ais.o
$(name)_remain_dep:=\
  rbm_util.o\
  util.o\
  repsoftmax.o\
  probability.o\
  join.o\
  rbm2.o\
  rbm.o\
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
