# Neural network pipeline: matmul -> add bias -> relu
t1 = matmul(x, W, m:1, n:4, k:4)
t2 = add(t1, b)
t3 = relu(t2)
return t3
