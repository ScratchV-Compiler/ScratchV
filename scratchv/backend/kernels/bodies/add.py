# -*- coding: utf-8 -*-
"""C[i] = A[i] + B[i]

入口：a0 = [A(N) 后面紧跟 B(N)]，a1 = C，a2 = N

两个档：
  unroll == 1  通用循环，每元素 9 条指令
  unroll > 1   展开 u 次 + 余数尾循环；偏移变立即数，循环开销摊到 u 个元素上
               每元素 ≈ 4 + 5/u 条指令。**不需要知道 N** —— 尾循环处理 N % u。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue, unrolled_body


def build(target, dtype, unroll: int = 1) -> list[str]:
    L = dtype.load
    S = dtype.store
    ADD = dtype.add
    T1, T2 = dtype.tmp1, dtype.tmp2

    if unroll == 1:
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

    if unroll & (unroll - 1):
        raise ValueError(f'unroll 必须是 2 的幂，收到 {unroll}')

    step = unroll * 4                # 每轮前进的字节数
    sh = unroll.bit_length() - 1     # log2(unroll)

    return [
        *prologue(),
        '    blez a2, .Lret',
        '    slli t0, a2, 2',                # t0 = N * 4
        '    add  t1, a0, t0',               # t1 = &B[0]
        f'    srli t2, a2, {sh}',            # t2 = N / unroll（完整轮数）
        '    beqz t2, .Ltail',               # N < unroll：直接走尾循环
        f'    slli t3, t2, {sh + 2}',        # t3 = 轮数 * unroll * 4
        '    add  t3, a0, t3',               # t3 = 展开段结束的地址
        '.Lloop:',
        *unrolled_body(dtype, unroll),       # 偏移全是立即数
        f'    addi a0, a0, {step}',          # 三个指针每轮只前进一次
        f'    addi t1, t1, {step}',
        f'    addi a1, a1, {step}',
        '    bne  a0, t3, .Lloop',
        '.Ltail:',
        f'    andi a2, a2, {unroll - 1}',    # 余数 = N % unroll
        '    beqz a2, .Lret',
        '.Ltail_loop:',
        f'    {L}  {T1}, 0(a0)',
        f'    {L}  {T2}, 0(t1)',
        f'    {ADD} {T1}, {T1}, {T2}',
        f'    {S}  {T1}, 0(a1)',
        '    addi a0, a0, 4',
        '    addi t1, t1, 4',
        '    addi a1, a1, 4',
        '    addi a2, a2, -1',
        '    bnez a2, .Ltail_loop',
        *epilogue(),
    ]
