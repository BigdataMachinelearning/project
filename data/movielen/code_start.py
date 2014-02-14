def Modify(src, des):
  f = open(src, 'rb')
  f2 = open(des, 'wb')
  for l in f:
    x = l.split(' ')
    if int(x[2]) > 4:
      x[2] = str(2)
    else:
      x[2] = str(1)
    f2.write(' '.join(x))
    f2.write('\n')

def main():
  src_path = 'movielen_train.txt'
  des_path = 'movielen_train2.txt'
  Modify(src_path, des_path)

main()
