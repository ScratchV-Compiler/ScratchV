# -*- coding: utf-8 -*-
"""组装：题名 → 完整的 .s 文本。"""

from scratchv.backend.kernels.bodies import BODIES, PROBLEMS
from scratchv.backend.kernels.dtypes import POLICIES
from scratchv.backend.kernels.target import TARGETS


# 实测选定的展开因子（第 5 章）。改 target 后必须重新扫。
#
# 扫描结果（内测比赛2，总 cost 取 10 个数据点之和）：
#   add-fp32        u: 1→181787  2→136049  4→121113  8→113615  16→110091  32→108779  64→109023 128→110796
#   reducesum-fp32  u: 1→91989   4→53959   8→50240   16→48478   32→47822   64→47944  128→48854
# 两条曲线都在 u=32 拐头：再大，循环体占的指令缓存行变多，取指未命中反超省下的循环开销。
# 逐点看，小 N（64/128）偏好 u=8，N≥256 偏好 u=32 —— 见 DEVELOPMENT.md 第 6 章。
UNROLL: dict[str, int] = {
    'add': 32,           'add-fp32': 32,
    'reducesum': 32,     'reducesum-fp32': 32,
    'matmul': 1,         'matmul-fp32': 1,
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
    return '\n'.join(body.build(target, dtype, u))
