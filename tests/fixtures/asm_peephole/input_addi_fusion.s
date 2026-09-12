# Fuse adjacent add-immediate pair into one
.text
.globl main
main:
  addi t0, t0, 3
  addi t0, t0, 5
  ret
