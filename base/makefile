BaseDir:=../
cmd:=\
  ctags -R *

.PHONY:main
main:
	$(cmd); cd makeout; make clean; make

name:=./run/base_unittest.sh
.PHONY:run
run:
	$(name)

.PHONY:clean
clean:
	cd makeout; make clean

.PHONY:shared
shared:
	cd makeout; make shared
