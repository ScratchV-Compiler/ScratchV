# Add followed by two ReLU stages
x = add(input, bias)
y = relu(x)
result = relu(y)
return result
