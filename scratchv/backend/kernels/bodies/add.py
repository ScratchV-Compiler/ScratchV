# -*- coding: utf-8 -*-
"""C[i] = A[i] + B[i]

入口：a0 = [A(N) 后面紧跟 B(N)]，a1 = C，a2 = N

O0 档：通用循环，对任意 N 正确，不做任何按形状的特化。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue


def build(target, dtype) -> list[str]:
    L = dtype.load           # 载入助记符：f32 是 FLW，q16 是 LW
    S = dtype.store          # 存储助记符
    ADD = dtype.add          # 加法助记符
    T1, T2 = dtype.tmp1, dtype.tmp2

    return [
        *prologue(),
        '    blez a2, .Lret',                # N <= 0 直接返回
        '    slli t0, a2, 2',                # t0 = N * 4
        '    add  t1, a0, t0',               # t1 = &B[0]
        '.Lloop:',
        f'    {L}  {T1}, 0(a0)',             # 读 A[i]
        f'    {L}  {T2}, 0(t1)',             # 读 B[i]
        f'    {ADD} {T1}, {T1}, {T2}',       # 算
        f'    {S}  {T1}, 0(a1)',             # 写 C[i]
        '    addi a0, a0, 4',                # 三个指针各前进一个元素
        '    addi t1, t1, 4',
        '    addi a1, a1, 4',
        '    addi a2, a2, -1',               # 剩余个数 -1
        '    bnez a2, .Lloop',
        *epilogue(),
    ]
