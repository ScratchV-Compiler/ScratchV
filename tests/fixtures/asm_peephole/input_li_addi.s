# Fold load-immediate followed by add-immediate
.text
.globl main
main:
  li t0, 10
  addi t0, t0, 5
  ret
