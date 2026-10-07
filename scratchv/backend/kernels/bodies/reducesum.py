# -*- coding: utf-8 -*-
"""out[0] = sum x[i]

入口：a0 = x(N)，a1 = out，a2 = N

O0 档：通用循环，对任意 N 正确。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue
from scratchv.backend.kernels.dtypes import zero_acc


def build(target, dtype) -> list[str]:
    L = dtype.load
    S = dtype.store
    ADD = dtype.add
    ACC, T1 = dtype.acc, dtype.tmp1

    return [
        *prologue(),
        '    blez a2, .Lret',
        *zero_acc(dtype, 't6'),                 # t6 在这道题里没别的用途，安全
        '.Lloop:',
        f'    {L}  {T1}, 0(a0)',
        f'    {ADD} {ACC}, {ACC}, {T1}',
        '    addi a0, a0, 4',
        '    addi a2, a2, -1',
        '    bnez a2, .Lloop',
        f'    {S}  {ACC}, 0(a1)',
        *epilogue(),
    ]
