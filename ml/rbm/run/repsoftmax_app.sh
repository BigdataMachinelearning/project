#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.1
  --bach_size=200
  --m=2000
  --k=10
  --hidden=150
  --it_num=10000
  --type=eigen
  --type=stl
  --type=softmax
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd

