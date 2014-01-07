#! /bin/sh -f
Include="$HOME/google-library/include"
Lib="$HOME/google-library/lib:$HOME/lib/gmp/lib:/home/lijk/lib/lib/"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/lda_unittest.o
  --gtest_filter=LDATest.LDATest
  --gtest_filter=LDATest.GibbsTest
  --gtest_filter=LDATest.LikelihoodTest
  --gtest_filter=LDATest.ReadDataTest
  --gtest_filter=LDATest.VAREMTest
  "
exec $cmd

