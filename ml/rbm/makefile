BaseDir:=../
cmd:=\
  ctags -R *

.PHONY:main
main:
	$(cmd); cd makeout; make clean; make

name:=./run/rbm_app.sh
name:=./run/repsoftmax_app.sh
name:=./run/rbm_unittest.sh
.PHONY:run
run:
	$(name)
