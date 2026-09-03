# Matrix multiplication followed by add
x = matmul(A, B, m:2, n:2, k:2)
result = add(x, bias)
return result
