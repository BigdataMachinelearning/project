BaseDir:=../
cmd:=\
  ctags -R *

.PHONY:main
main:
	$(cmd); cd makeout; make clean; make

name:=./run/info_unittest.sh
name:=./run/util_unittest.sh
name:=./run/document_unittest.sh
name:=./run/eigen_unittest.sh
name:=./run/eigen_app.sh
.PHONY:run
run:
	$(name)
