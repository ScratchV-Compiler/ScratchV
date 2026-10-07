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
