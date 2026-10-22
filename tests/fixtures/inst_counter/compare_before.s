.text
main:
  add a0, a1, a2
  add a0, a0, a3
  lw a4, 0(sp)
  sw a4, 4(sp)
  beq a0, a1, .L1
  j .L2
.L1:
  li a0, 1
.L2:
  ret
