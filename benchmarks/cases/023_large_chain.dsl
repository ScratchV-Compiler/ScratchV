# Long chained computation: (a + b) * relu(a - b) / gelu(x)
t1 = add(a, b)
t2 = sub(a, b)
t3 = relu(t2)
t4 = mul(t1, t3)
t5 = gelu(x)
t6 = div(t4, t5)
return t6
