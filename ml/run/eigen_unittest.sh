#! /bin/sh -f
Include="$HOME/google-library/include"
Lib="$HOME/google-library/lib:$HOME/lib/gmp/lib:/home/lijk/lib/lib/"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/eigen_unittest.o
  --gtest_filter=Eigen.ReadDataTest
  "
exec $cmd

