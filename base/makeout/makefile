include makefile.config
BaseDir:=../..
include build_makefile
WorkDir:=$(BaseDir)/base

name:=base_unittest
$(name)_update_dep:=\
  base.o\
  join.o\
  probability.o\
  string_util.o
$(name)_remain_dep:=\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=base_app
$(name)_update_dep:=
$(name)_remain_dep:=\
  $(BaseDir)/lib/base.so\
  $(GTestDir)/gtest-all.o
include post_app_makefile

name:=base_app
name:=base_unittest
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)

name:=$(WorkDir)/base.cc\
      $(WorkDir)/join.cc\
      $(WorkDir)/string_util.cc
.PHONY:shared
shared:
	$(CC) $(CXXFLAGS2) $(name) -shared $(Include) -o $(BaseDir)/lib/base.so
