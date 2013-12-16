#! /bin/sh -f
Include="$HOME/google-library/include"
Lib="$HOME/google-library/lib:$HOME/lib/gmp/lib:/home/lijk/lib/lib/"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_unittest.o
  --gtest_filter=RBMTest.BaiduLoadTest
  --gtest_filter=RBMTest.LoadMovieLenTest
  --gtest_filter=RBMTest.SoftMaxTest
  --gtest_filter=RBMTest.MovieLenTest
  --gtest_filter=RBMTest.BaiduTest
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd

