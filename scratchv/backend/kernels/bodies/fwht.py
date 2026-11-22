# -*- coding: utf-8 -*-
"""out = H_N · x（快速哈达玛变换，正向）

入口：a0 = x(N)，a1 = out，a2 = N（2 的幂）

**O0 基线**：完全照 `reference/fwht.c` 的三重循环写，不做任何优化。

  ① 先把输入拷到输出（此后在原地变换）
  ② span = 1, 2, 4, … < N
       i = 0, 2·span, 4·span, … < N
         j = 0 .. span-1
           u = out[i+j]，v = out[i+j+span]
           out[i+j]     = u + v
           out[i+j+span] = u - v

每个元素都重新算一遍地址（不做指针强度削减）、循环不展开、不分块 ——
这就是"O0 水平"。后续优化（展开、把 i+j 的地址递推出来、蝶形向量化）
都在这个基线之上做，而不是替换它。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue


def build(target, dtype, unroll: int = 1) -> list[str]:
    # 基线固定不展开。展开是后续优化，先保证这份汇编只描述算法本身。
    if unroll != 1:
        raise ValueError('fwht 的 O0 基线不支持展开（unroll 必须为 1）')

    L, S = dtype.load, dtype.store
    ADD, SUB = dtype.add, dtype.sub
    U, V, W = dtype.tmp1, dtype.tmp2, dtype.prod      # 三个互不相同的浮点寄存器

    return [
        *prologue(),
        '    blez a2, .Lret',                        # N <= 0 直接返回
        # ---- ① 拷贝：out[i] = x[i] --------------------------------------
        '    mv   t0, a2',                           # 剩余个数
        '    mv   t1, a0',                           # 源指针
        '    mv   t2, a1',                           # 目的指针
        '.Lcopy:',
        f'    {L}  {U}, 0(t1)',
        f'    {S}  {U}, 0(t2)',
        '    addi t1, t1, 4',
        '    addi t2, t2, 4',
        '    addi t0, t0, -1',
        '    bnez t0, .Lcopy',
        # ---- ② 蝶形：span 从 1 翻倍到 N --------------------------------
        '    li   t0, 1',                            # span
        '.Lspan:',
        '    bge  t0, a2, .Lret',                    # span >= N → 结束
        '    li   t1, 0',                            # i
        '.Li:',
        '    bge  t1, a2, .Lspan_step',              # i >= N → 下一级
        '    li   t2, 0',                            # j
        '.Lj:',
        '    bge  t2, t0, .Li_step',                 # j >= span → 下一组
        '    add  t3, t1, t2',                       # idx = i + j
        '    slli t3, t3, 2',                        # idx * 4
        '    add  t3, a1, t3',                       # &out[i+j]
        '    add  t4, t1, t2',
        '    add  t4, t4, t0',                       # idx2 = i + j + span
        '    slli t4, t4, 2',
        '    add  t4, a1, t4',                       # &out[i+j+span]
        f'    {L}  {U}, 0(t3)',                      # u
        f'    {L}  {V}, 0(t4)',                      # v
        f'    {ADD} {W}, {U}, {V}',                  # u + v
        f'    {S}  {W}, 0(t3)',
        f'    {SUB} {W}, {U}, {V}',                  # u - v
        f'    {S}  {W}, 0(t4)',
        '    addi t2, t2, 1',                        # j++
        '    j    .Lj',
        '.Li_step:',
        '    slli t3, t0, 1',                        # 2 * span
        '    add  t1, t1, t3',                       # i += 2 * span
        '    j    .Li',
        '.Lspan_step:',
        '    slli t0, t0, 1',                        # span <<= 1
        '    j    .Lspan',
        *epilogue(),
    ]
