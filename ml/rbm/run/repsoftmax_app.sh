#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.001
  --eta=0.01
  --beta_beg=0.001
  --bach_size=2
  --k=1
  --it_num=10
  --type=eigen
  --type=stl
  --type=softmax
  --train_path=../../data/ap.dat
  --train_path=../data/document_demo
  --train_path=test
  --algorithm_type=1
  --ais_run=1000
  "
gdb="
  gdb ./makeout/rbm_app.o
  "

exec $cmd
#exec $gdb
