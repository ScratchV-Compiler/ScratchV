# Stress test: 32 simultaneously-live virtual registers
# Fan-in tree over 32 distinct intermediate values using input variables
# (no constants, to avoid the pre-existing add-with-immediate emitter bug).
#
# Peak liveness is at the first reduction instruction, where all 32
# intermediate vregs are still alive: 32 > 19 (greedy) / 27 (linear scan).
# This triggers the spill/reload path in both register allocators.

# Layer 1: 32 values — each is x*2 (add of same register, valid RISC-V)
v01 = add(x, x)
v02 = add(x, x)
v03 = add(x, x)
v04 = add(x, x)
v05 = add(x, x)
v06 = add(x, x)
v07 = add(x, x)
v08 = add(x, x)
v09 = add(x, x)
v10 = add(x, x)
v11 = add(x, x)
v12 = add(x, x)
v13 = add(x, x)
v14 = add(x, x)
v15 = add(x, x)
v16 = add(x, x)
v17 = add(x, x)
v18 = add(x, x)
v19 = add(x, x)
v20 = add(x, x)
v21 = add(x, x)
v22 = add(x, x)
v23 = add(x, x)
v24 = add(x, x)
v25 = add(x, x)
v26 = add(x, x)
v27 = add(x, x)
v28 = add(x, x)
v29 = add(x, x)
v30 = add(x, x)
v31 = add(x, x)
v32 = add(x, x)

# Layer 2: 16 reductions — PEAK LIVENESS: all 32 vregs still alive
r01 = add(v01, v02)
r02 = add(v03, v04)
r03 = add(v05, v06)
r04 = add(v07, v08)
r05 = add(v09, v10)
r06 = add(v11, v12)
r07 = add(v13, v14)
r08 = add(v15, v16)
r09 = add(v17, v18)
r10 = add(v19, v20)
r11 = add(v21, v22)
r12 = add(v23, v24)
r13 = add(v25, v26)
r14 = add(v27, v28)
r15 = add(v29, v30)
r16 = add(v31, v32)

# Layer 3: 8 reductions
s01 = add(r01, r02)
s02 = add(r03, r04)
s03 = add(r05, r06)
s04 = add(r07, r08)
s05 = add(r09, r10)
s06 = add(r11, r12)
s07 = add(r13, r14)
s08 = add(r15, r16)

# Layer 4: 4 reductions
t01 = add(s01, s02)
t02 = add(s03, s04)
t03 = add(s05, s06)
t04 = add(s07, s08)

# Layer 5: 2 reductions
u01 = add(t01, t02)
u02 = add(t03, t04)

# Layer 6: final reduction
final = add(u01, u02)
return final
