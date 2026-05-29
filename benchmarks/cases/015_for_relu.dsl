# Apply relu in a loop
for i = 0, 4
  t1 = relu(x)
  y = add(y, t1)
endfor
return y
