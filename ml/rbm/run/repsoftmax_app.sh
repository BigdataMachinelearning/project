#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.001
  --eta=0.01
  --bach_size=200
  --k=10
  --it_num=5
  --type=eigen
  --type=stl
  --type=softmax
  --train_path=../../data/ap.dat
  --train_path=../data/document_demo
  --algorithm_type=1
  "
gdb="
  gdb ./makeout/rbm_app.o
  "

exec $cmd
#exec $gdb

