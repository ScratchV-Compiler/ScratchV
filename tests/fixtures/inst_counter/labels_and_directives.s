# Fixture for testing labels, directives, comments and blank lines

.text
.file "test.c"
.cfi_startproc
.option arch, +c

loop:
  # comment in loop
.L1:
1:
name$part:
loop2: addi t0, t0, -1
.L2: bnez t0, .L2
10: nop

.data
.word 0x13
.byte 0x01, 0x02
.string "hello"
.cfi_endproc
