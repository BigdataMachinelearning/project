#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.0000001
  --bach_size=100
  --k=50
  --it_num=5
  --type=eigen
  --type=stl
  --type=softmax
  --train_path=test
  --train_path=../../data/ap.dat
  --algorithm_type=1
  "
gdb="
  gdb ./makeout/rbm_app.o
  "

exec $cmd
#exec $gdb

