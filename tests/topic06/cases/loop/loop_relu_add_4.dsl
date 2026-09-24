# Loop with add followed by ReLU in the body
for i = 0, 4
x = add(input, bias)
y = relu(x)
endfor
return y
