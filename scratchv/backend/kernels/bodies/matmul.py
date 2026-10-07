# -*- coding: utf-8 -*-
"""C[i][j] = sum_k A[i][k] * B[k][j]

入口：a0 = [A(N*N) 后面紧跟 B(N*N)]，a1 = C，a2 = N

O0 档：通用三重循环，对任意 N 正确。

⚠️ 这里有一个真实的翻车记录：置零的暂存寄存器曾经写死成 `t6`，
而 `t6` 存的是行步长 N*4 —— 一置零，步长就没了，10 个数据点全部算错，
而汇编器不报任何错。所以 `zero_acc` 的暂存寄存器必须由调用点指定。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue
from scratchv.backend.kernels.dtypes import zero_acc, mac_instrs


def build(target, dtype) -> list[str]:
    L = dtype.load
    S = dtype.store
    ACC, T1, T2 = dtype.acc, dtype.tmp1, dtype.tmp2

    return [
        *prologue(),
        '    blez a2, .Lret',
        '    mul  t0, a2, a2',              # t0 = N * N
        '    slli t0, t0, 2',               # t0 = N * N * 4  （B 的起始偏移）
        '    add  t5, a0, t0',              # t5 = &B[0]
        '    slli t6, a2, 2',               # t6 = 行步长 = N * 4
        '    mv   t3, a0',                  # t3 = &A[i][0]
        '    mv   t4, a1',                  # t4 = &C[i][0]
        '    li   t1, 0',                   # i = 0
        '.Li:',
        '    li   t2, 0',                   # j = 0
        '.Lj:',
        *zero_acc(dtype, 's0'),             # ★ 用 s0，不能用 t6（见模块 docstring）
        '    mv   a3, t3',                  # a3 = &A[i][0]
        '    slli a4, t2, 2',               # a4 = j * 4
        '    add  a4, t5, a4',              # a4 = &B[0][j]
        '    li   a5, 0',                   # k = 0
        '.Lk:',
        f'    {L}  {T1}, 0(a3)',            # 读 A[i][k]
        f'    {L}  {T2}, 0(a4)',            # 读 B[k][j]
        *mac_instrs(dtype),                 # ★ 乘加：f32 是 2 条，q16 是 3 条（含 >>16）
        '    addi a3, a3, 4',               # A 的 k 方向 +1
        '    add  a4, a4, t6',              # B 的 k 方向 +1 行
        '    addi a5, a5, 1',               # k++
        '    bne  a5, a2, .Lk',             # k != N 就继续
        '    slli t0, t2, 2',               # 算出 C[i][j] 的地址
        '    add  t0, t4, t0',
        f'    {S}  {ACC}, 0(t0)',           # 写回
        '    addi t2, t2, 1',               # j++
        '    bne  t2, a2, .Lj',
        '    add  t3, t3, t6',              # A 的行指针 +1 行
        '    add  t4, t4, t6',              # C 的行指针 +1 行
        '    addi t1, t1, 1',               # i++
        '    bne  t1, a2, .Li',
        *epilogue(),
    ]
