GTEST_DIR:=.
gtest-all.o:
	g++ -I ${GTEST_DIR}/include -I ${GTEST_DIR} -c ${GTEST_DIR}/src/gtest-all.cc -lpthread
.PHONY:clean
clean:
	rm *.o
