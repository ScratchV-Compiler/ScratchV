# while loop accumulation with multiple ops
while (i < 5):
  t1 = mul(x, y)
  acc = add(acc, t1)
  i = add(i, 1)
endwhile
return acc
