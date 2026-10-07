# -*- coding: utf-8 -*-
"""组装：题名 → 完整的 .s 文本。"""

from scratchv.backend.kernels.bodies import BODIES, PROBLEMS
from scratchv.backend.kernels.dtypes import POLICIES
from scratchv.backend.kernels.target import TARGETS


# ═══════════════════════════════════════════════════════════════════════════
# 两张实测表。**每个值都必须连同「当时按什么形状假设、什么目标选的」一起记。**
#
# 为什么：平台不公布逐个数据点的规模（DEVELOPMENT.md §0.6），只公布区间。
# 所以"取总 cost 最小的"这句话里的「总」需要说明白是对着哪些 N 量的 ——
# 否则几个月后没人知道这个数字为什么是这个数，也没人知道它什么时候失效。
# ═══════════════════════════════════════════════════════════════════════════

# 展开因子（DEVELOPMENT.md §5.1）。
#
# 【采样】range:64:4096:8  →  N = {64, 640, 1216, 1792, 2368, 2944, 3520, 4096}
#         **这是本仓库自己选的采样，不是平台的数据点。**
#         平台的逐点规模保密（只公布区间），而且**会变** —— 写作期间就被整体换过一次。
#         所以任何"总 cost"都必须写明是对着哪条采样量的。
# 【目标】总 cost 最小。
#
# 扫描结果（总 cost 取上述采样点之和，用 /root/workspace/riscv_matmul/probe.py 量）：
#   add-fp32        u: 1→197431  2→147649  4→131279  8→123079  16→119159  32→117559  64→117479
#   reducesum-fp32  u: 4→58305   8→54220   16→52260   32→51460
#
# ⚠️ 换个目标，答案就变。按"**最坏相对损失**"口径，两条曲线都是 **u=16 最稳健**：
#
#     add-fp32        u=16 → 最坏偏离 102.2%      u=64 → 128.7%（总量最优，但 N=64 那点差 28.7%）
#     reducesum-fp32  u=16 → 最坏偏离 102.0%      u=32 → 109.3%（总量最优）
#
#   从 32 换到 16：总量只让约 1.4%，而最坏情况从 10.4% 降到 2.2%。
#   **看不到数据点的场合应该填 16；现状填 32 是按"总量"这个目标选的。**
UNROLL: dict[str, int] = {
    'add': 32,           'add-fp32': 32,
    'reducesum': 32,     'reducesum-fp32': 32,
    'matmul': 1,         'matmul-fp32': 1,
}

# 寄存器分块 (mr, nr)（DEVELOPMENT.md §6.3）。(0,0) = 走通用三重循环。
#
# 【采样】range:4:64:8  →  N = {4, 13, 21, 30, 38, 47, 55, 64}
# 【目标】总 cost 最小。
#
# ⚠️ 分块只在 `N % 4 == 0` 时生效，其余整题退回通用实现。在那条采样上
#    **8 个点只有 2 个（4 和 64）真正走分块**，其余 6 个是 1.00x（一步没走到）。
#    数据点允许任意整数之后，这条约束的代价被放大 —— 见 §6.3。
#
# 容量约束：mr*nr + mr + nr + 1 个**数据**寄存器，从 TargetDesc 对应银行取。
#   f32：数据在浮点组（32 个）→ (4,4) 需要 25 个，放得下
#   q16：数据在整数组（23 个）→ (4,4) 需要 25 个，放不下，暂时走通用实现
BLOCKING: dict[str, tuple[int, int]] = {
    'matmul-fp32': (4, 4),
}


def build_program(problem: str, unroll: int | None = None) -> str:
    """按题名生成一份自包含的 .s。

    `unroll` 省略时取 UNROLL 表里的实测值。
    """
    if problem not in PROBLEMS:
        known = ', '.join(sorted(PROBLEMS))
        raise KeyError(f'未知题名 {problem!r}；已知：{known}')

    body_name, dtype_name, target_name = PROBLEMS[problem]
    body = BODIES[body_name]
    dtype = POLICIES[dtype_name]
    target = TARGETS[target_name]
    u = UNROLL.get(problem, 1) if unroll is None else unroll
    blk = BLOCKING.get(problem, (0, 0))
    if body_name == 'matmul':
        return '\n'.join(body.build(target, dtype, u, blk))
    return '\n'.join(body.build(target, dtype, u))
