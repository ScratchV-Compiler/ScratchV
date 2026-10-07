# -*- coding: utf-8 -*-
"""目标机器描述：唯一允许写指令助记符和寄存器名的地方。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetDesc:
    """一台目标机器的全部静态事实。"""

    name: str                    # 这台机器的名字，如 'rv32imf'
    march: str                   # 编译时传给 clang 的 -march
    mabi: str                    # 编译时传给 clang 的 -mabi
    int_regs: tuple[str, ...]    # 整数寄存器池（指针、计数用；不含 a0/a1/a2）
    fp_regs: tuple[str, ...]     # 浮点寄存器池（算浮点用；没有就留空）
    load: dict[str, str]         # 数值类型 → 载入指令
    store: dict[str, str]        # 数值类型 → 存储指令
    imm_max: int                 # 立即数上限（12 位有符号）
    line_bytes: int              # 一级缓存行大小


# 整数寄存器：a0/a1/a2 是平台传进来的参数，不动；其余都可用。
INT_POOL = ('s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
            't0', 't1', 't2', 't3', 't4', 't5',
            'a3', 'a4', 'a5', 'a6', 'a7')

# 浮点寄存器：f0 到 f31
FP_POOL = tuple(f'f{i}' for i in range(32))


TARGETS: dict[str, TargetDesc] = {
    'rv32im': TargetDesc(
        name='rv32im', march='rv32im', mabi='ilp32',
        int_regs=INT_POOL, fp_regs=(),
        load={'int32': 'lw'}, store={'int32': 'sw'},
        imm_max=2047, line_bytes=64,
    ),
    'rv32imf': TargetDesc(
        name='rv32imf', march='rv32imf', mabi='ilp32',
        int_regs=INT_POOL, fp_regs=FP_POOL,
        load={'int32': 'lw', 'f32': 'flw'},
        store={'int32': 'sw', 'f32': 'fsw'},
        imm_max=2047, line_bytes=64,
    ),
}
