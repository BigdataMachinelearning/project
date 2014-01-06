BaseDir:=../
cmd:=\
  ctags -R *

.PHONY:main
main:
	$(cmd); cd makeout; make clean; make

name:=./run/info_unittest.sh
name:=./run/document_unittest.sh
name:=./run/util_unittest.sh
.PHONY:run
run:
	$(name)
