# -*- coding: utf-8 -*-
"""out[0] = sum x[i]

入口：a0 = x(N)，a1 = out，a2 = N

两个档：
  unroll == 1  通用循环，每元素 5 条指令
  unroll > 1   展开 u 次 + 余数尾循环。**不需要知道 N**。

骨架与 `add` 共用（`loopgen.unrolled_loop`）。与 `add` 的区别只有三处，全部从
参数进去：只前进一个指针（`ptrs=['a0']`）、累加器清零、以及余数为 0 时跳到
`.Lout` 而不是 `.Lret`（累加器要在那里写出去）。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue, unrolled_loop
from scratchv.backend.kernels.dtypes import zero_acc


def build(target, dtype, unroll: int = 1) -> list[str]:
    L = dtype.load
    S = dtype.store
    ADD = dtype.add
    ACC, T1 = dtype.acc, dtype.tmp1

    single = [
        f'    {L}  {T1}, 0(a0)',
        f'    {ADD} {ACC}, {ACC}, {T1}',
    ]

    if unroll == 1:
        return [
            *prologue(),
            '    blez a2, .Lret',
            *zero_acc(dtype, 't6'),                 # t6 在这道题里没别的用途，安全
            '.Lloop:',
            *single,
            '    addi a0, a0, 4',
            '    addi a2, a2, -1',
            '    bnez a2, .Lloop',
            f'    {S}  {ACC}, 0(a1)',
            *epilogue(),
        ]

    body: list[str] = []
    for k in range(unroll):
        body += [
            f'    {L}  {T1}, {k * 4}(a0)',
            f'    {ADD} {ACC}, {ACC}, {T1}',
        ]

    return [
        *prologue(),
        *unrolled_loop(
            unroll=unroll,
            prologue=zero_acc(dtype, 't6'),
            body=body,
            tail=single,
            ptrs=['a0'],                            # 只有一个输入流要前进
            exit_label='.Lout',                     # 余数为 0 也要去写出累加器
        ),
        '.Lout:',
        f'    {S}  {ACC}, 0(a1)',
        *epilogue(),
    ]
