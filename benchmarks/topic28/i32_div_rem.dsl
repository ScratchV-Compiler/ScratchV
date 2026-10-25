a = const(-17, dtype:i32)
b = const(5, dtype:i32)
quotient = div(a, b)
remainder = rem(a, b)
result = add(quotient, remainder)
return result
