# Loop + register pressure regression test.
# Accumulator (acc) survives across loop back-edges while the loop body
# creates a chain of dependent intermediates from x.
# This tests that labels at the loop header don't corrupt register
# mappings — the original _flush_regs() bug tore live ranges here.
for i = 0, 10
  v1 = add(x, x)
  v2 = add(v1, x)
  v3 = add(v2, v1)
  v4 = add(v3, v2)
  v5 = add(v4, v3)
  v6 = add(v5, v4)
  acc = add(acc, v6)
endfor
return acc
