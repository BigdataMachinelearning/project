#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_unittest.o
  --gtest_filter=RBMTest.BaiduTest
  --gtest_filter=RBMTest.BaiduLoadTest
  --gtest_filter=RBMTest.LoadMovieLenTest
  --gtest_filter=RBMTest.MovieLenTest
  --gtest_filter=EigenRBMTest.MovieLenTest
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd

