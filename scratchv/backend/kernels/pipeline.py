# -*- coding: utf-8 -*-
"""组装：题名 → 完整的 .s 文本。"""

from scratchv.backend.kernels.bodies import BODIES, PROBLEMS
from scratchv.backend.kernels.dtypes import POLICIES
from scratchv.backend.kernels.target import TARGETS


def build_program(problem: str) -> str:
    """按题名生成一份自包含的 .s。"""
    if problem not in PROBLEMS:
        known = ', '.join(sorted(PROBLEMS))
        raise KeyError(f'未知题名 {problem!r}；已知：{known}')

    body_name, dtype_name, target_name = PROBLEMS[problem]
    body = BODIES[body_name]
    dtype = POLICIES[dtype_name]
    target = TARGETS[target_name]
    return '\n'.join(body.build(target, dtype))
