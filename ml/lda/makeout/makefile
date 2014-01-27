BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/lda

name:=lda_unittest
$(name)_update_dep:=\
  document.o\
  lda.o\
  lda_model.o\
  lda_var_em.o\
  lda_gibbs.o
$(name)_remain_dep:=\
  base.o\
  util.o\
  cokus.o\
  join.o\
  string_util.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=power_law_unittest
$(name)_update_dep:=\
  string_util.o
$(name)_remain_dep:=\
  base.o\
  cokus.o\
  document.o\
  join.o\
  lda.o\
  lda_model.o\
  lda_var_em.o\
  lda_gibbs.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=lda_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
