# Deterministic RV32 scheduling case; initial state is supplied by the runner.
# a0=1024, t2=7, t3=0, memory[a0]=9, memory[a0+4]=0.
# The independent addi can fill the load-use stall: model cycles 5 -> 4.
# Expected output: t0=9, t1=16, t3=1, memory[a0+4]=16.
.text
scheduler_feature:
  lw t0, 0(a0)
  add t1, t0, t2
  addi t3, t3, 1
  sw t1, 4(a0)
