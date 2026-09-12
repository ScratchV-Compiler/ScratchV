# Always-taken branch should become unconditional jump
.text
.globl main
main:
  beq x0, x0, target
target:
  ret
