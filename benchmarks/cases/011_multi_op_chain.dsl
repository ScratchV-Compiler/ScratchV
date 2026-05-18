# Multi-operation chain: (a + b) * (a - b)
t1 = add(a, b)
t2 = sub(a, b)
t3 = mul(t1, t2)
return t3
