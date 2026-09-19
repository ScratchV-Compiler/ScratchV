# Low-pressure dependency chain: values die quickly and should not spill.
t00 = add(a, b)
t01 = mul(t00, c)
t02 = sub(t01, d)
t03 = add(t02, a)
t04 = mul(t03, b)
t05 = sub(t04, c)
t06 = add(t05, d)
t07 = mul(t06, a)
t08 = sub(t07, b)
t09 = add(t08, c)
return t09
