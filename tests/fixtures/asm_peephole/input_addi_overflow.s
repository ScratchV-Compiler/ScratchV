# Reject fusion when immediate sum overflows signed 12-bit range
.text
.globl main
main:
  addi t0, t0, 2000
  addi t0, t0, 2000
  ret
