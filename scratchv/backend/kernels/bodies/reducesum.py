# -*- coding: utf-8 -*-
"""out[0] = sum x[i]

入口：a0 = x(N)，a1 = out，a2 = N

两个档：
  unroll == 1  通用循环，每元素 5 条指令
  unroll > 1   展开 u 次 + 余数尾循环。**不需要知道 N**。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue
from scratchv.backend.kernels.dtypes import zero_acc


def build(target, dtype, unroll: int = 1) -> list[str]:
    L = dtype.load
    S = dtype.store
    ADD = dtype.add
    ACC, T1 = dtype.acc, dtype.tmp1

    if unroll == 1:
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

    if unroll & (unroll - 1):
        raise ValueError(f'unroll 必须是 2 的幂，收到 {unroll}')

    step = unroll * 4
    sh = unroll.bit_length() - 1

    body = []
    for k in range(unroll):
        body += [
            f'    {L}  {T1}, {k * 4}(a0)',
            f'    {ADD} {ACC}, {ACC}, {T1}',
        ]

    tail = [f'    {L}  {T1}, 0(a0)', f'    {ADD} {ACC}, {ACC}, {T1}']

    return [
        *prologue(),
        '    blez a2, .Lret',
        *zero_acc(dtype, 't6'),
        f'    srli t2, a2, {sh}',            # t2 = N / unroll
        '    beqz t2, .Ltail',
        f'    slli t3, t2, {sh + 2}',
        '    add  t3, a0, t3',               # 展开段结束地址
        '.Lloop:',
        *body,
        f'    addi a0, a0, {step}',
        '    bne  a0, t3, .Lloop',
        '.Ltail:',
        f'    andi a2, a2, {unroll - 1}',
        '    beqz a2, .Lout',
        '.Ltail_loop:',
        *tail,
        '    addi a0, a0, 4',
        '    addi a2, a2, -1',
        '    bnez a2, .Ltail_loop',
        '.Lout:',
        f'    {S}  {ACC}, 0(a1)',
        *epilogue(),
    ]
