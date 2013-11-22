BaseDir:=../..
include build_makefile
WorkDir:=$(BaseDir)/base

name:=base_unittest
$(name)_update_dep:=\
  join.o\
  string_util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=base_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
