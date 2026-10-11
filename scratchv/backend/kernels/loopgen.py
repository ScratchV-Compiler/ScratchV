# -*- coding: utf-8 -*-
"""把一段叶子算子套进循环里，产出汇编文本。"""


def prologue(entry: str = 'cnn_entry') -> list[str]:
    """每个 .s 的开头模板。`.option norvc` 不能少 —— 压缩指令会被判整题 0 分。"""
    return [
        '    .option norvc',
        '    .option norelax',
        '',
        '    .text',
        '    .balign 4',
        f'    .globl {entry}',
        f'    .type {entry}, @function',
        f'{entry}:',
    ]


def epilogue() -> list[str]:
    return ['.Lret:', '    ret', '']


def unrolled_body(dtype, unroll: int) -> list[str]:
    """生成展开 unroll 次的【逐元素】循环体：读 → 算 → 写。

    只适用于 add 这类一对一映射的题。偏移是立即数（`k * 4`），
    不含指针前进和计数——那两个由调用方在每轮做一次。
    """
    out: list[str] = []
    for k in range(unroll):
        off = k * 4
        out += [
            f'    {dtype.load}  {dtype.tmp1}, {off}(a0)',
            f'    {dtype.load}  {dtype.tmp2}, {off}(t1)',
            f'    {dtype.add} {dtype.tmp1}, {dtype.tmp1}, {dtype.tmp2}',
            f'    {dtype.store} {dtype.tmp1}, {off}(a1)',
        ]
    return out


def unrolled_loop(*, unroll: int, prologue: list[str], body: list[str],
                  tail: list[str], ptrs: list[str],
                  exit_label: str = '.Lret') -> list[str]:
    """把一段【单元素循环体】套成「展开 u 轮 + 余数尾循环」——**对任意 N 正确**。

    这是 `add` 与 `reducesum` 共用的骨架：幂校验、轮数计算、展开段结束地址、
    余数尾循环。两题的差别只有三处，都从参数进来：

      prologue    循环前的一次性准备（算 B 基址、累加器清零…）。
                  出来后 `a0` 必须指向第一个待处理元素。
      body        展开体，一**轮** u 个元素；偏移用立即数 `0/4/…/(u-1)*4`，从 `a0` 起算。
      tail        单体体，同样从 `a0` 起算。**它是"任意 N 正确"的关键**
                  —— 比赛的数据点若都是 u 的倍数，这段永远不执行。
      ptrs        需要每轮前进的指针寄存器，如 `['a0','t1','a1']`。
                  **`a0` 必须在里面**（循环哨兵用它）。尾循环里这些指针按单元素前进。
      exit_label  余数为 0 时跳到哪。`add` 跳 `.Lret`（不需要收尾）；
                  `reducesum` 跳 `.Lout`（要在那里把累加器写出去）。

    寄存器占用：`t2`（轮数）、`t3`（结束地址）。调用方不要用这两个。
    """
    if unroll < 1 or unroll & (unroll - 1):
        raise ValueError(f'unroll 必须是 2 的幂，收到 {unroll}')
    if 'a0' not in ptrs:
        raise ValueError("a0 必须在内 ptrs 里——循环哨兵要用它")
    step = unroll * 4
    sh = unroll.bit_length() - 1           # log2(unroll)
    return [
        '    blez a2, .Lret',
        *prologue,
        f'    srli t2, a2, {sh}',           # t2 = N / unroll（完整轮数）
        '    beqz t2, .Ltail',              # N < unroll：直接走尾循环
        f'    slli t3, t2, {sh + 2}',       # t3 = 轮数 * unroll * 4
        '    add  t3, a0, t3',              # 展开段结束的地址
        '.Lloop:',
        *body,
        *[f'    addi {p}, {p}, {step}' for p in ptrs],
        '    bne  a0, t3, .Lloop',
        '.Ltail:',
        f'    andi a2, a2, {unroll - 1}',   # 余数 = N % unroll
        f'    beqz a2, {exit_label}',
        '.Ltail_loop:',
        *tail,
        *[f'    addi {p}, {p}, 4' for p in ptrs],
        '    addi a2, a2, -1',
        '    bnez a2, .Ltail_loop',
    ]
