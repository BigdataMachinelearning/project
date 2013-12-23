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
  --it_num=10000
  --type=eigen
  --type=stl
  --train_path=data/train_g20.txt
  --test_path=data/test_g20.txt
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd

