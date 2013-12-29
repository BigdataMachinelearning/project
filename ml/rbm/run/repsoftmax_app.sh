#! /bin/sh -f
Lib="$HOME/google-library/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.001
  --bach_size=2
  --k=2
  --it_num=2000
  --type=eigen
  --type=stl
  --type=softmax
  "
gdb="
  gdb ./makeout/rbm_app.o
  "

exec $cmd
#exec $gdb

