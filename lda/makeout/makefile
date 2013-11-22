BaseDir:=../..
include build_makefile
WorkDir:=$(BaseDir)/lda

name:=lda_unittest
$(name)_update_dep:=\
  cokus.o\
  join.o\
  lda.o\
  lda_estimate.o\
  lda_infer_gibbs.o\
  lda_model.o\
  string_util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=lda_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
