# -*- coding: utf-8 -*-
"""C[i] = A[i] + B[i]

入口：a0 = [A(N) 后面紧跟 B(N)]，a1 = C，a2 = N

两个档：
  unroll == 1  通用循环，每元素 9 条指令
  unroll > 1   展开 u 次 + 余数尾循环；偏移变立即数，循环开销摊到 u 个元素上
               每元素 ≈ 4 + 5/u 条指令。**不需要知道 N** —— 尾循环处理 N % u。

骨架（幂校验、轮数、尾循环）在 `loopgen.unrolled_loop`，与 `reducesum` 共用。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue, unrolled_loop


def build(target, dtype, unroll: int = 1) -> list[str]:
    L = dtype.load
    S = dtype.store
    ADD = dtype.add
    T1, T2 = dtype.tmp1, dtype.tmp2

    # 单元素体：读 A、读 B、加、写 C。展开体与尾体共用这一段。
    single = [
        f'    {L}  {T1}, 0(a0)',
        f'    {L}  {T2}, 0(t1)',
        f'    {ADD} {T1}, {T1}, {T2}',
        f'    {S}  {T1}, 0(a1)',
    ]

    if unroll == 1:
        return [
            *prologue(),
            '    blez a2, .Lret',                # N <= 0 直接返回
            '    slli t0, a2, 2',                # t0 = N * 4
            '    add  t1, a0, t0',               # t1 = &B[0]
            '.Lloop:',
            *single,
            '    addi a0, a0, 4',                # 三个指针各前进一个元素
            '    addi t1, t1, 4',
            '    addi a1, a1, 4',
            '    addi a2, a2, -1',               # 剩余个数 -1
            '    bnez a2, .Lloop',
            *epilogue(),
        ]

    body: list[str] = []
    for k in range(unroll):
        off = k * 4
        body += [
            f'    {L}  {T1}, {off}(a0)',
            f'    {L}  {T2}, {off}(t1)',
            f'    {ADD} {T1}, {T1}, {T2}',
            f'    {S}  {T1}, {off}(a1)',
        ]

    return [
        *prologue(),
        *unrolled_loop(
            unroll=unroll,
            prologue=['    slli t0, a2, 2',      # t0 = N * 4
                      '    add  t1, a0, t0'],    # t1 = &B[0]
            body=body,
            tail=single,
            ptrs=['a0', 't1', 'a1'],             # 三个指针每轮各前进 u 个元素
        ),
        *epilogue(),
    ]
