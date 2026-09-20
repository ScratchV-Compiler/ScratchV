# Expected result: 6
i = add(0, 0)
total = add(0, 0)
while (i < 3):
    j = add(0, 0)
    while (j < 2):
        if (i >= 0):
            total = add(total, 1)
        else:
            total = sub(total, 1)
        endif
        j = add(j, 1)
    endwhile
    i = add(i, 1)
endwhile
return total
