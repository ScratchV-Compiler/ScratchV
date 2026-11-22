# -*- coding: utf-8 -*-
"""题册：题名 → （用哪个 body，哪种数值，哪台机器）。"""

from scratchv.backend.kernels.bodies import add, fwht, matmul, reducesum

BODIES = {
    'add': add,
    'reducesum': reducesum,
    'matmul': matmul,
    'fwht': fwht,
}

# 题名 → (body 名, 数值类型, 目标机器)
PROBLEMS = {
    'add-fp32':       ('add',       'f32', 'rv32imf'),
    'reducesum-fp32': ('reducesum', 'f32', 'rv32imf'),
    'matmul-fp32':    ('matmul',    'f32', 'rv32imf'),

    # 内测比赛1 的三题：只换数值类型和目标机器，body 一行都不用改。
    'add':            ('add',       'q16', 'rv32im'),
    'reducesum':      ('reducesum', 'q16', 'rv32im'),
    'matmul':         ('matmul',    'q16', 'rv32im'),

    # ScratchV杯（场次 demo）第一题：哈达玛变换，FP32。
    # 当前是 O0 基线（不展开、不分块），见 bodies/fwht.py。
    'fwht':           ('fwht',      'f32', 'rv32imf'),
}
