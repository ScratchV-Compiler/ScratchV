# Clean sequence: optimizer should change nothing
.text
.globl main
main:
  add t0, t1, t2
  sub t3, t4, t5
  ret
