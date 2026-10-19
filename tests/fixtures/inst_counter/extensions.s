.text
main:
  # FP instructions
  fadd.s fa0, fa1, fa2
  fmadd.d fa3, fa4, fa5, fa6
  fsqrt.s fa7, fa0
  flw fa1, 0(sp)
  fsd fa2, 8(sp)
  fcvt.w.s a0, fa0
  fmv.x.w a1, fa1

  # Atomic instructions
  lr.w a2, (a3)
  sc.w a4, a2, (a3)
  amoadd.w.aq a5, a1, (a3)
  amoswap.d.aqrl a6, a7, (a3)

  # System instructions
  fence
  fence.i
  csrrw t0, mstatus, t1
  ecall
  ebreak

  # Compressed instructions
  c.addi a0, 1
  c.lw a1, 0(sp)
  c.beqz a0, .L_c
  c.j .L_c
  c.nop

.L_c:
  ret
