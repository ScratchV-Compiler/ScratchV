# return后死代码 -> 不可达块
c = add(a, b)
return c
dead = mul(a, b)
return dead
