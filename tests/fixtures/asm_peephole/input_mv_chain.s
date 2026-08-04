# Collapse redundant move chain through temporary
.text
.globl main
main:
  mv t0, t1
  mv t2, t0
  ret
