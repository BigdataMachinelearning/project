#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_unittest.o
  --gtest_filter=EigenRBMTest.MovieLenTest
  --gtest_filter=RepSoftmaxTest.AisTest
  --gtest_filter=AisTest.UniformSampleTest
  --gtest_filter=Ais.AisTest
  --gtest_filter=Ais.WAisTest
  --gtest_filter=Ais.LogPartitionTest
  --gtest_filter=RBMTest.BaiduTest
  --gtest_filter=RBMTest.BaiduLoadTest
  --gtest_filter=RBMTest.LoadMovieLenTest
  --gtest_filter=RBMTest.MovieLenTest
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd

