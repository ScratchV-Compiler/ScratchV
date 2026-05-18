# if-else with multiple ops in branches
if (a > b):
  t1 = add(a, b)
  t2 = mul(t1, 2.0)
  c = relu(t2)
else:
  t1 = sub(a, b)
  c = relu(t1)
endif
return c
