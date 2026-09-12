# Fuse adjacent add-immediate with hex literals
.text
.globl main
main:
  addi t0, t0, 0x10
  addi t0, t0, 0x20
  ret
