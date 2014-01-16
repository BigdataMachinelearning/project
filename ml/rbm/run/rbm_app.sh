#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.3
  --bach_size=200
  --m=2000
  --k=5
  --hidden=150
  --it_num=1000000
  --type=softmax
  --type=stl
  --type=eigen
  --train_path=tmp/train_g20.txt
  --test_path=tmp/test_g20.txt
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd

