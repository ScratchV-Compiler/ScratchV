# Loop with chained add inside the body
for i = 0, 4
x = add(a, b)
y = add(x, c)
endfor
return y
