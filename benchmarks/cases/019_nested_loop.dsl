# Nested for loops
for i = 0, 4
  for j = 0, 2
    t1 = mul(x, y)
    acc = add(acc, t1)
  endfor
endfor
return acc
