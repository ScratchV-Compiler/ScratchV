.text
.globl main
main:
  add a0, a1, a2
  addi a0, a0, 10
  sub t0, a0, a1
  mul t1, t0, a2
  div t2, t1, a0
  srai t3, t2, 2
  lui a3, 0x1000
  lw a4, 0(sp)
  sw a4, 4(sp)
  beq a0, a1, .L_exit
  j main
.L_exit:
  ret
  li a5, 100
  mv a6, a5
  nop
