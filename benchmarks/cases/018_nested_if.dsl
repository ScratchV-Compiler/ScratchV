# Nested if-else (extended parser)
if (a > b):
  if (a > 0):
    c = add(a, b)
  else:
    c = mul(a, b)
  endif
else:
  c = sub(a, b)
endif
return c
