# -*- coding: utf-8 -*-
"""C[i][j] = sum_k A[i][k] * B[k][j]

入口：a0 = [A(N*N) 后面紧跟 B(N*N)]，a1 = C，a2 = N

两档：
  blocking == (0, 0)   通用三重循环 —— 每条 MAC 约 8 条指令（f32）
  blocking == (mr, nr) 寄存器分块 —— 每条 MAC 约 2.5 条指令

⚠️ 两个真实的翻车记录（见 DEVELOPMENT.md 第 3 章）：
  1. 置零的暂存寄存器曾经写死成 `t6`，而 `t6` 存的是行步长 N*4 —— 10 个数据点
     全部算错，汇编器不报任何错。
  2. 曾经把"乘"建模成单个 `mul` 助记符，但一次乘加在 f32 与 q16 下**形状不同**
     （q16 要先右移 16 位）。见 dtypes.DtypePolicy.mac。
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue
from scratchv.backend.kernels.dtypes import zero_acc, mac_instrs


# ── 通用三重循环 ────────────────────────────────────────────────────────────

def _generic(target, dtype) -> list[str]:
    L = dtype.load
    S = dtype.store
    ACC, T1, T2 = dtype.acc, dtype.tmp1, dtype.tmp2

    return [
        '.Lgeneric:',
        '    blez a2, .Lret',
        '    mul  t0, a2, a2',              # t0 = N * N
        '    slli t0, t0, 2',               # t0 = N * N * 4  （B 的起始偏移）
        '    add  t5, a0, t0',              # t5 = &B[0]
        '    slli t6, a2, 2',               # t6 = 行步长 = N * 4
        '    mv   t3, a0',                  # t3 = &A[i][0]
        '    mv   t4, a1',                  # t4 = &C[i][0]
        '    li   t1, 0',                   # i = 0
        '.Li:',
        '    li   t2, 0',                   # j = 0
        '.Lj:',
        *zero_acc(dtype, 's0'),             # 用 s0，不能用 t6（见模块 docstring）
        '    mv   a3, t3',                  # a3 = &A[i][0]
        '    slli a4, t2, 2',               # a4 = j * 4
        '    add  a4, t5, a4',              # a4 = &B[0][j]
        '    li   a5, 0',                   # k = 0
        '.Lk:',
        f'    {L}  {T1}, 0(a3)',            # 读 A[i][k]
        f'    {L}  {T2}, 0(a4)',            # 读 B[k][j]
        *mac_instrs(dtype),                 # 乘加：f32 是 2 条，q16 是 3 条（含 >>16）
        '    addi a3, a3, 4',               # A 的 k 方向 +1
        '    add  a4, a4, t6',              # B 的 k 方向 +1 行
        '    addi a5, a5, 1',               # k++
        '    bne  a5, a2, .Lk',             # k != N 就继续
        '    slli t0, t2, 2',               # 算出 C[i][j] 的地址
        '    add  t0, t4, t0',
        f'    {S}  {ACC}, 0(t0)',           # 写回
        '    addi t2, t2, 1',               # j++
        '    bne  t2, a2, .Lj',
        '    add  t3, t3, t6',              # A 的行指针 +1 行
        '    add  t4, t4, t6',              # C 的行指针 +1 行
        '    addi t1, t1, 1',               # i++
        '    bne  t1, a2, .Li',
        *epilogue(),
    ]


# ── 寄存器分块 ──────────────────────────────────────────────────────────────

def blocked_regs_needed(mr: int, nr: int) -> int:
    """分块需要多少个**数据**寄存器：累加器 + 一行 A + 一行 B + 一个乘积落点。"""
    return mr * nr + mr + nr + 1


def _blocked(target, dtype, mr: int, nr: int) -> list[str]:
    """(mr × nr) 寄存器分块。

    数据寄存器从 `TargetDesc` 对应银行取 —— 浮点的累加器在浮点组、
    指针在整数组，两个银行各自独立。**不要写死容量。**
    """
    pool = target.fp_regs if dtype.bank == 'fp' else target.int_regs
    need = blocked_regs_needed(mr, nr)
    if len(pool) < need:
        raise ValueError(
            f'{target.name}/{dtype.name} 上放不下 {mr}x{nr} 分块：'
            f'需要 {need} 个 {dtype.bank} 寄存器，只有 {len(pool)} 个')

    acc = list(pool[:mr * nr])
    av = list(pool[mr * nr:mr * nr + mr])
    bv = list(pool[mr * nr + mr:mr * nr + mr + nr])
    prod = pool[mr * nr + mr + nr]        # 乘积落点，与 av/bv 都不同

    L, S = dtype.load, dtype.store

    # ── 整数侧寄存器（循环机制，与数据寄存器天然分属两个银行）──
    STRIDE = 't0'                       # 行步长 N*4
    a_base = ['s0', 's1', 's2', 's3'][:mr]      # A 的 mr 行基址
    a_cur = ['s4', 's5', 's6', 's7'][:mr]       # A 的 mr 个游标
    b_cur, b_base, c_base = 's8', 's9', 's10'
    I, J, K, TMP, ADDR = 't1', 't2', 't3', 't4', 'a3'
    MAINI, MAINJ = 'a4', 't5'                   # 完整块的行/列边界

    def zero(reg: str) -> str:
        return f'    fmv.w.x {reg}, zero' if dtype.bank == 'fp' else f'    li   {reg}, 0'

    def reset_rows() -> list[str]:
        """把 A 的行基址与 C 的块基址复位到第 0 行。②③ 段要靠它回到起点。"""
        lines = [f'    mv   {a_base[0]}, a0']
        for r in range(1, mr):
            lines.append(f'    add  {a_base[r]}, {a_base[r - 1]}, {STRIDE}')
        lines.append(f'    mv   {c_base}, a1')
        return lines

    unit = max(mr, nr)                  # 两者都是 2 的幂，取大者即最小公倍数的掩码位数
    out = [
        '    blez a2, .Lret',
        f'    li   {TMP}, {unit}',
        f'    bltu a2, {TMP}, .Lgeneric',         # N < 分块粒度：太小，不值得分块
        '',
        f'    slli {STRIDE}, a2, 2',              # STRIDE = N*4
        f'    mul  {TMP}, a2, a2',
        f'    slli {TMP}, {TMP}, 2',              # N*N*4
        f'    add  {b_base}, a0, {TMP}',          # &B[0][0]
        # 完整块边界：向下取整到 mr / nr。用 `andi x, x, -4` 取 N 的低位清零。
        f'    andi {MAINI}, a2, {-mr}',
        f'    andi {MAINJ}, a2, {-nr}',
    ]

    # ── ① 主块：i ∈ [0, mainI)，j ∈ [0, mainJ) ──────────────────────────────
    out += reset_rows()
    out += [f'    li   {I}, 0', '.Bli:', f'    li   {J}, 0', '.Blj:']
    for r in range(mr):
        out.append(f'    mv   {a_cur[r]}, {a_base[r]}')
    out += [
        f'    slli {TMP}, {J}, 2',
        f'    add  {b_cur}, {b_base}, {TMP}',     # &B[0][j]
    ]
    for reg in acc:
        out.append(zero(reg))
    out += [f'    li   {K}, 0', '.Blk:']
    for r in range(mr):
        out.append(f'    {L}  {av[r]}, 0({a_cur[r]})')
    for c in range(nr):
        out.append(f'    {L}  {bv[c]}, {c * 4}({b_cur})')
    for r in range(mr):
        for c in range(nr):
            out += mac_instrs(dtype, acc[r * nr + c], av[r], bv[c], prod)
    for r in range(mr):
        out.append(f'    addi {a_cur[r]}, {a_cur[r]}, 4')
    out += [
        f'    add  {b_cur}, {b_cur}, {STRIDE}',
        f'    addi {K}, {K}, 1',
        f'    bne  {K}, a2, .Blk',
        f'    slli {TMP}, {J}, 2',
        f'    add  {ADDR}, {c_base}, {TMP}',      # &C[i][j]
    ]
    for r in range(mr):
        for c in range(nr):
            out.append(f'    {S}  {acc[r * nr + c]}, {c * 4}({ADDR})')
        if r != mr - 1:
            out.append(f'    add  {ADDR}, {ADDR}, {STRIDE}')
    out += [
        f'    addi {J}, {J}, {nr}',
        f'    bne  {J}, {MAINJ}, .Blj',           # j 到 mainJ（不是 N）
        f'    slli {TMP}, {STRIDE}, {mr.bit_length() - 1}',   # mr 行 = mr * STRIDE
    ]
    for reg in a_base:
        out.append(f'    add  {reg}, {reg}, {TMP}')
    out += [
        f'    add  {c_base}, {c_base}, {TMP}',
        f'    addi {I}, {I}, {mr}',
        f'    bne  {I}, {MAINI}, .Bli',           # i 到 mainI（不是 N）
    ]

    # ── ② 右带：i ∈ [0, mainI)，j ∈ [mainJ, N) —— 每列一个 mr×1 块 ──────────
    # ⚠️ 循环是「先做后判」，所以 mainJ == N（右边没有余列）时必须**整段跳过**，
    #    否则会从第 N 列写出去。这一条在 N 是 4 的倍数时必然触发。
    out.append(f'    beq  {MAINJ}, a2, .Brs')
    out += reset_rows()
    out += [f'    li   {I}, 0', '.Bri:', f'    mv   {J}, {MAINJ}', '.Brj:']
    for r in range(mr):
        out.append(f'    mv   {a_cur[r]}, {a_base[r]}')
    out += [
        f'    slli {TMP}, {J}, 2',
        f'    add  {b_cur}, {b_base}, {TMP}',
    ]
    for r in range(mr):
        out.append(zero(acc[r]))
    out += [f'    li   {K}, 0', '.Brk:']
    for r in range(mr):
        out.append(f'    {L}  {av[r]}, 0({a_cur[r]})')
    out.append(f'    {L}  {bv[0]}, 0({b_cur})')
    for r in range(mr):
        out += mac_instrs(dtype, acc[r], av[r], bv[0], prod)
    for r in range(mr):
        out.append(f'    addi {a_cur[r]}, {a_cur[r]}, 4')
    out += [
        f'    add  {b_cur}, {b_cur}, {STRIDE}',
        f'    addi {K}, {K}, 1',
        f'    bne  {K}, a2, .Brk',
        f'    slli {TMP}, {J}, 2',
        f'    add  {ADDR}, {c_base}, {TMP}',      # &C[i][j]，这一列
    ]
    for r in range(mr):
        out.append(f'    {S}  {acc[r]}, 0({ADDR})')
        if r != mr - 1:
            out.append(f'    add  {ADDR}, {ADDR}, {STRIDE}')
    out += [
        f'    addi {J}, {J}, 1',                  # 一列一列走
        f'    bne  {J}, a2, .Brj',                # j 到 N
        f'    slli {TMP}, {STRIDE}, {mr.bit_length() - 1}',
    ]
    for reg in a_base:
        out.append(f'    add  {reg}, {reg}, {TMP}')
    out += [
        f'    add  {c_base}, {c_base}, {TMP}',
        f'    addi {I}, {I}, {mr}',
        f'    bne  {I}, {MAINI}, .Bri',           # i 到 mainI
        '.Brs:',
    ]

    # ── ③ 底带：i ∈ [mainI, N)，j ∈ [0, N) —— 标量 ─────────────────────────
    # 同理：mainI == N（下边没有余行）时整段跳过。
    out.append(f'    beq  {MAINI}, a2, .Lret')
    out += [
        f'    mul  {TMP}, {MAINI}, {STRIDE}',     # mainI 行处
        f'    add  {a_base[0]}, a0, {TMP}',       # &A[mainI][0]
        f'    add  {c_base}, a1, {TMP}',          # &C[mainI][0]
        f'    mv   {I}, {MAINI}',
        '.Bbi:',
        f'    li   {J}, 0',
        '.Bbj:',
        zero(acc[0]),
        f'    mv   {ADDR}, {a_base[0]}',          # &A[i][0]
        f'    slli {TMP}, {J}, 2',
        f'    add  {b_cur}, {b_base}, {TMP}',     # &B[0][j]
        f'    li   {K}, 0',
        '.Bbk:',
        f'    {L}  {av[0]}, 0({ADDR})',
        f'    {L}  {bv[0]}, 0({b_cur})',
        *mac_instrs(dtype, acc[0], av[0], bv[0], prod),
        f'    addi {ADDR}, {ADDR}, 4',
        f'    add  {b_cur}, {b_cur}, {STRIDE}',
        f'    addi {K}, {K}, 1',
        f'    bne  {K}, a2, .Bbk',
        f'    slli {TMP}, {J}, 2',
        f'    add  {TMP}, {c_base}, {TMP}',
        f'    {S}  {acc[0]}, 0({TMP})',
        f'    addi {J}, {J}, 1',
        f'    bne  {J}, a2, .Bbj',
        f'    add  {a_base[0]}, {a_base[0]}, {STRIDE}',
        f'    add  {c_base}, {c_base}, {STRIDE}',
        f'    addi {I}, {I}, 1',
        f'    bne  {I}, a2, .Bbi',
        '    j    .Lret',                         # 三段跑完就出去，别落进通用实现
        '',
    ]
    return out


# ── 入口 ────────────────────────────────────────────────────────────────────

def build(target, dtype, unroll: int = 1, blocking: tuple[int, int] = (0, 0)) -> list[str]:
    """blocking == (0,0) 走通用三重循环；否则走 (mr × nr) 寄存器分块。

    两条路径共用同一个 `.Lret: ret`（由 `_generic` 的 epilogue 提供）。
    """
    head = prologue()
    if blocking == (0, 0):
        return [*head, *_generic(target, dtype)]

    mr, nr = blocking
    if mr > 4:
        raise ValueError('分块的行数上限是 4（A 的行基址/游标各写死了 4 个）')
    return [*head, '', *_blocked(target, dtype, mr, nr), *_generic(target, dtype)]
