a = const(1.5, dtype:f64)
b = const(2.0, dtype:f64)
c = const(3.0, dtype:f64)
d = const(4.0, dtype:f64)
left = mul(a, b)
right = mul(c, d)
result = add(left, right)
return result
