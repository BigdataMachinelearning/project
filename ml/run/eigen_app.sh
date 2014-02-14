#! /bin/sh -f
Include="$HOME/google-library/include"
Lib="$HOME/google-library/lib:$HOME/lib/gmp/lib:/home/lijk/lib/lib/"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/eigen_app.o
  --train_path=../data/movielen_train.txt
  --test_path=../data/movielen_test.txt
  --train_path=rbm/tmp/fengxing/data/train
  --test_path=rbm/tmp/fengxing/data/test
  --k=10
  --lambda=0.1
  --eta=0.0001
  "
exec $cmd
