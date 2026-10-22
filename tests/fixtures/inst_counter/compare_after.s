.text
main:
  add a0, a1, a2
  # one add removed
  lw a4, 0(sp)
  # one sw removed
  beq a0, a1, .L1
  ret
.L1:
  li a0, 1
  ret
