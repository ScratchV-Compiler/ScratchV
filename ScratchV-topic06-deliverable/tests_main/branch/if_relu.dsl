# Branch selects whether to apply ReLU after an add.
sum = add(a, b)
if use_relu
result = relu(sum)
return result
else
return sum
endif
