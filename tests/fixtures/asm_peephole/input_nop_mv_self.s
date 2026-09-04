# Eliminate no-ops and identity moves
.text
.globl main
main:
  nop
  mv t0, t0
  addi t0, t0, 1
  ret
