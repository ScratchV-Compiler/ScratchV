a = const(1.5, dtype:f64)
b = const(2.0, dtype:f64)
c = const(3.25, dtype:f64)
partial = add(a, b)
result = add(partial, c)
return result
