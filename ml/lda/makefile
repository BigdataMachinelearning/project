BaseDir:=../
cmd:=\
  ctags -R *

.PHONY:main
main:
	$(cmd); cd makeout; make clean; make

name:=./run/lda_unittest.sh
name:=./run/power_law_unittest.sh
.PHONY:run
run:
	$(name)
