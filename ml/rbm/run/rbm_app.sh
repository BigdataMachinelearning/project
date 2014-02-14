#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.01
  --bach_size=200
  --m=2000
  --k=2
  --hidden=150
  --it_num=300
  --type=softmax
  --type=stl
  --type=eigen
  --train_path=tmp/fengxing/data/train
  --test_path=tmp/fengxing/data/test
  --train_path=tmp/train_g20.txt
  --test_path=tmp/test_g20.txt
  --train_path=../../data/movielen/movielen_train2.txt
  --test_path=../../data/movielen/movielen_test2.txt
  --train_path=../../data/movielen/movielen_train3.txt
  --test_path=../../data/movielen/movielen_test3.txt
  "
gdb="
  gdb ./makeout/rbm_app.o
  "

exec $cmd
#exec $gdb

