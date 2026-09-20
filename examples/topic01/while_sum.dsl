# Expected result: 15
i = add(1, 0)
total = add(0, 0)
while (i <= 5):
    total = add(total, i)
    i = add(i, 1)
endwhile
return total
