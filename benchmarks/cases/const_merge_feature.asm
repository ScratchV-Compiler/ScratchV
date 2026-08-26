.text
entry:
  # Rule A: fold an adjacent lui/addi pair into a source-level li.
  lui x5, 1
  addi x5, x5, 2

  # Rule B: the second identical lui is redundant because x6 is unchanged.
  lui x6, 2
  addi x7, x0, 7
  lui x6, 2
  addi x6, x6, 3
