BaseDir:=../..
include build_makefile
WorkDir:=$(BaseDir)/lda

name:=lda_unittest
$(name)_update_dep:=\
  lda.o\
  lda_model.o\
  lda_var_em.o\
  lda_gibbs.o\
  join.o\
  string_util.o
$(name)_remain_dep:=\
  cokus.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=lda_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
