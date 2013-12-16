BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/rbm

name:=rbm_unittest
$(name)_update_dep:=\
  rbm.o\
  string_util.o\
  base.o\
  join.o\
  rbm_util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=rbm_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
