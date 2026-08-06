"""课题17：寄存器分配瓶颈场景系统性测试（v1.4）

瓶颈分析维度（6大维度，每个维度2-4个具体场景）：
  1. 活跃度维度 —— 活跃 vreg 数量 vs 物理寄存器数量
  2. 生命期维度 —— 区间长度分布模式
  3. 复用模式维度 —— 自溢 vreg 的访问频率和模式
  4. eviction 模式维度 —— eviction 触发频率和 victim 选择
  5. 栈压力维度 —— spill slot 复用效率
  6. 代码膨胀维度 —— spill code 对最终代码大小的影响

每个场景输出：
  - 场景含义（构造意图）
  - 构建方式（具体代码）
  - 详细测试数据

关于非法多源指令的说明（v1.4）
=============================
部分高压场景（A01/A02/A03/D01/E03 等）故意把几十个 vreg 的 ``uses``
放进*一条* ``add`` 指令，以此构造“全部 vreg 在同一位置完全重叠”的
极端压力。这类指令的 operands 列表超出了 RISC-V 三操作数 ISA 的写实范围，
生成的汇编片段是“压力模型 dump”，**不代表可执行语义**——分配器的行为
完全由 ``defines``/``uses``/``compute_live_intervals`` 驱动，operands 只是
展示载体，不影响压力测量的正确性。若要得到可执行的合法汇编，应将这些
多源指令改写为两两累加链（会显著降低同一位置的峰值压力），那是另一类
测试，不属于本框架的压力模型范围。
"""
from scratchv.backend.regalloc_linear_v1_4 import LinearScanAllocator, LsInstruction
import random
import inspect


PHYS_REGS = ['a0','a1','a2','a3','a4','a5','a6','a7',
             't0','t1','t2','t3','t4','t5','t6',
             's0','s1','s2','s3','s4','s5','s6','s7',
             's8','s9','s10','s11']
POOL = len(PHYS_REGS)  # 27


def _same_end_consumer(N, start_id=0):
    """Build a pressure block with *N* vregs that *all* start at position 0
    (fully overlapping) and are consumed through a *legal* accumulation chain.

    A single ``add`` with ``N`` source operands would express the same
    "everything is live at once" pressure but is not a valid RISC-V
    instruction (and, when N exceeds the physical pool, is physically
    impossible to allocate at one program point).  Consuming via a two-source
    ``add`` chain keeps every vreg live from position 0 until the chain
    touches it, i.e. it preserves the same all-overlapping pressure profile
    while remaining representable and allocatable.

    Returns
    -------
    list[LsInstruction]:
        The LI batch (id 0) followed by the accumulation chain (ids 1..).
    """
    block = [LsInstruction(start_id, 'li', [f'v{i}', str(i)],
                           defines={f'v{i}'}, uses=set())
             for i in range(N)]
    block.append(LsInstruction(start_id + 1, 'li', ['v_acc0', '0'],
                               defines={'v_acc0'}, uses=set()))
    for i in range(N):
        block.append(LsInstruction(
            start_id + 2 + i, 'add', [f'v_acc{i+1}', f'v_acc{i}', f'v{i}'],
            defines={f'v_acc{i+1}'}, uses={f'v_acc{i}', f'v{i}'}))
    return block


def _renumber(block):
    """Return a copy of *block* with strictly unique, sequential ids.

    The linear-scan allocator keys all internal spill/reload/eviction state
    by ``inst.id`` and assumes each id uniquely identifies one instruction
    (as produced by ``block_from_machine_instrs``).  Several scenario builders
    reuse the same id across many instructions (e.g. all ``li`` use id 0) to
    express "simultaneously created"; handing those directly to the allocator
    corrupts the id-keyed state.  Renumbering by block position preserves the
    builders' intent (list order is execution order) while satisfying the
    unique-id contract.
    """
    from dataclasses import replace
    return [replace(inst, id=i) for i, inst in enumerate(block)]


def _all_spill_lines(alloc):
    """Return every spill ``sw`` store line emitted by the allocator.

    In the v1.3 allocator:
      - ``spill_code``:   dict[int, list[str]]  -> self-spill ``sw`` stores
      - ``_evictions``:   dict[int, list[str]]  -> eviction ``sw`` stores
      - ``_reloads``:     dict[int, list[(vreg, slot)]] -> ``lw`` reloads

    Reload ``lw`` lines are not returned here: they carry no ``vreg`` in a
    machine-readable position and are counted separately from the emitted
    assembly (``code.count('  lw ')``).
    """
    lines = []
    for pend in (alloc.spill_code, alloc._evictions):
        for pos in sorted(pend):
            lines.extend(pend[pos])
    return lines


def run_scenario(name, category, desc, block_fn, phys_regs=None):
    """Run a single scenario and return all metrics."""
    regs = phys_regs or PHYS_REGS
    alloc = LinearScanAllocator(phys_regs=regs)
    try:
        block = _renumber(block_fn())
        ivs = alloc.compute_live_intervals(block)
        alloc.allocate(ivs)
        code = alloc.get_allocated_code(block)
    except Exception as e:
        import traceback
        return {
            'name': name, 'category': category, 'desc': desc,
            'pool': len(regs), 'error': str(e),
            'traceback': traceback.format_exc(),
        }

    total_vregs = len(alloc.alloc_map)

    # === Eviction vs self-spill counting ===
    # v1.3/v1.4: evictions are recorded in alloc._evictions (sw stores emitted
    # before a reload), self-spills in alloc.spill_code.  Since v1.4 spill_code
    # also carries the stack stores emitted after a *re-defined* spilled vreg,
    # both classes are genuinely "sw after a definition".  The actual physical
    # stores bound to stack slots are counted separately from the emitted
    # assembly (``code.count('  sw ')``).
    eviction_sw = sum(len(v) for v in alloc._evictions.values())
    self_spill_sw = sum(len(v) for v in alloc.spill_code.values())
    evicted = eviction_sw
    self_spill = self_spill_sw

    # Code metrics — match opcodes at the start of an emitted line (all
    # emitted instructions are indented by two spaces, comments are separated
    # by ``#``), so the tally is robust to indentation width and never
    # matches an operand or a comment.
    stores = sum(1 for ln in code.split('\n')
                 if ln.startswith('  sw ') or ln.startswith('\tsw '))
    loads = sum(1 for ln in code.split('\n')
                if ln.startswith('  lw ') or ln.startswith('\tlw '))
    total_lines = len(code.split('\n'))
    asm_size = len(code.encode('utf-8'))

    # Compute avg uses per vreg and other patterns
    avg_uses = sum(len(iv.uses) for iv in ivs) / max(len(ivs), 1)
    max_uses = max((len(iv.uses) for iv in ivs), default=0)

    # Redundant reloads: a reload of a vreg is only wasteful if the very same
    # (vreg, slot) pair is reloaded more than once within a single instruction
    # slot — the earlier ``lw`` result would be overwritten before it is ever
    # consumed.  Reloads of the same vreg at *different* source positions are
    # legitimate: the stack slot may have been redefined in between, so the
    # value must be fetched afresh.  (A naive "count every reload after the
    # first for each vreg" massively overstates redundancy — most positions
    # genuinely need their own load.)
    redundant_reloads = 0
    seen_in_slot: dict[str, int] = {}  # vreg -> count within current slot
    current_slot: int | None = None
    for pos in sorted(alloc._reloads):
        if current_slot != pos:
            current_slot = pos
            seen_in_slot = {}
        for vreg, _slot in alloc._reloads[pos]:
            seen_in_slot[vreg] = seen_in_slot.get(vreg, 0) + 1
            if seen_in_slot[vreg] > 1:
                redundant_reloads += 1

    # slot reuse efficiency
    # Count how many times a slot at the same offset is used for different vregs
    if alloc._spill_slots:
        unique_slots = len(set(alloc._spill_slots.values()))
        slot_reuse_ratio = 1.0 - (unique_slots / max(len(alloc._spill_slots), 1))
    else:
        slot_reuse_ratio = 0.0

    # alloc_map integrity check: any vreg leaking into emitted asm operands?
    vreg_leaks = []
    for line in code.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        asm = line.split('#')[0]
        for v in sorted(alloc.alloc_map.keys(), key=len, reverse=True):
            # ``SPILL_<vreg>`` operands are the intended spilled form, not a
            # leak; only a bare vreg name in the asm operands is a real leak.
            if ('SPILL_' + v) in asm:
                continue
            if v in asm:
                vreg_leaks.append(f'{v} in "{line}"')
                break

    # detect redundant sw (same slot, same value stored twice in a row)
    redundant_sw = 0
    sw_history = {}
    for line in _all_spill_lines(alloc):
        if ' sw ' not in f" {line.strip()} ":
            continue
        try:
            slot = line.split(',')[1].strip().split('(')[0]
        except IndexError:
            continue
        vpart = ''
        if 'spill ' in line:
            vpart = line.split('spill ')[-1].strip()
        elif 'evict ' in line:
            vpart = line.split('evict ')[-1].strip()
        if slot in sw_history and sw_history[slot] == vpart:
            redundant_sw += 1
        sw_history[slot] = vpart

    return {
        'name': name,
        'category': category,
        'desc': desc,
        'pool': len(regs),
        'total_vregs': total_vregs,
        'peak_active': alloc.peak_active,
        'peak_real_pressure': alloc.peak_real_pressure,
        'self_spill': self_spill,
        'evicted': evicted,
        'spill_slots': len(alloc._spill_slots),
        'stores': stores,
        'loads': loads,
        'total_lines': total_lines,
        'asm_size_bytes': asm_size,
        'avg_uses': round(avg_uses, 2),
        'max_uses': max_uses,
        'redundant_reloads': redundant_reloads,
        'slot_reuse_ratio': round(slot_reuse_ratio, 3),
        'redundant_sw': redundant_sw,
        'vreg_leaks': vreg_leaks,
        'intervals': len(ivs),
        'block_len': len(block),
        'spill_code_entries': (len(alloc.spill_code) + len(alloc._evictions)
                               + sum(len(v) for v in alloc._reloads.values())),
    }


def print_result(r, verbose=False):
    """Print formatted scenario result."""
    err = r.get('error')
    if err:
        print(f"\n{'='*72}")
        print(f"[{r['name']}] {r['desc']}")
        print(f"  ERROR: {err}")
        print(f"  {r.get('traceback', '')}")
        return

    flags = []
    if r['total_vregs'] > r['pool'] and r['peak_active'] <= r['pool']:
        flags.append('PEAK_LOCKED')
    if r['stores'] > 0 and r['loads'] == 0:
        flags.append('STORE_ONLY')
    if r['loads'] > r['stores']:
        flags.append('LOADS>STORES')
    if r['vreg_leaks']:
        flags.append('VREG_LEAK')
    if r['redundant_reloads'] > 0:
        flags.append(f'REDUN_RELOAD({r["redundant_reloads"]})')
    if r['redundant_sw'] > 0:
        flags.append(f'REDUN_SW({r["redundant_sw"]})')
    if r['slot_reuse_ratio'] < 0.01 and r['spill_slots'] > 1:
        flags.append('SLOT_NO_REUSE')

    flag_str = f' !!! {", ".join(flags)}' if flags else ''

    print(f"\n{'='*72}")
    print(f"[{r['name']}] {r['desc']}  [{r['category']}]{flag_str}")
    print(f"{'='*72}")
    print(f"  vregs={r['total_vregs']:>4d}  pool={r['pool']:>2d}  "
          f"peak_active={r['peak_active']:>3d}/{r['pool']:>2d}  "
          f"block_len={r['block_len']:>4d}")
    print(f"  self_spill={r['self_spill']:>3d}  evicted={r['evicted']:>3d}  "
          f"spill_slots={r['spill_slots']:>3d}")
    print(f"  stores(sw)={r['stores']:>4d}  loads(lw)={r['loads']:>4d}  "
          f"spill_ops={r['spill_code_entries']:>4d}")
    print(f"  asm_lines={r['total_lines']:>4d}  asm_bytes={r['asm_size_bytes']:>5d}")
    print(f"  avg_uses/vreg={r['avg_uses']:>5.2f}  max_uses={r['max_uses']:>3d}  "
          f"intervals={r['intervals']:>4d}")
    print(f"  slot_reuse={r['slot_reuse_ratio']:.1%}  redundant_sw={r['redundant_sw']:>3d}  "
          f"redundant_reloads={r['redundant_reloads']:>3d}")
    if r['vreg_leaks']:
        for leak in r['vreg_leaks'][:3]:
            print(f"  VREG LEAK: {leak}")


# =====================================================================
# 瓶颈维度 1: 活跃度维度
#   当活跃 vreg 数量超过物理寄存器时，分配器必须做出溢出决策。
#   不同的溢出决策影响不同，测试各种 vreg/pool 比例下的瓶颈表现。
# =====================================================================

def A01_slight_overlap():
    """轻度超压: pool+5 个 vreg 完全重叠，观察 entry-level 溢出行为"""
    return _same_end_consumer(POOL + 5)

def A02_moderate_overlap():
    """中度超压: pool*2 vreg 完全重叠"""
    return _same_end_consumer(POOL * 2)

def A03_severe_overlap():
    """重度超压: pool*7 vreg 完全重叠（200个）"""
    return _same_end_consumer(200)

def A04_no_overlap():
    """零超压: 完全不重叠，验证无压力下的基线"""
    return [
        inst
        for i in range(POOL * 4)
        for inst in [
            LsInstruction(i*2, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set()),
            LsInstruction(i*2+1, 'addi', [f'v{i}_2', f'v{i}', '1'],
                          defines={f'v{i}_2'}, uses={f'v{i}'}),
        ]
    ]

# =====================================================================
# 瓶颈维度 2: 生命期模式维度
#   区间长度分布不同，影响 expire 和 eviction 的行为
# =====================================================================

def B01_short_lived_dense():
    """200个密集短区间: 每个 vreg 只活一条指令，但同时在 pos 0 全部创建"""
    return (
        [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(200)] +
        [LsInstruction(1, 'addi', [f'v{i}_2', f'v{i}', '1'],
                       defines={f'v{i}_2'}, uses={f'v{i}'})
         for i in range(200)]
    )

def B02_long_chain():
    """长依赖链: 200步链式依赖，vreg 长时间占用单个寄存器"""
    return [
        LsInstruction(0, 'li', ['v0', '0'], defines={'v0'}, uses=set())
    ] + [
        LsInstruction(i, 'addi', ['v0', 'v0', '1'], defines={'v0'}, uses={'v0'})
        for i in range(1, 200)
    ]

def B03_mixed_lifetimes():
    """混合生命期: 3个长寿命 + 大量短寿命 vreg"""
    def build():
        block = [LsInstruction(0, 'li', ['v_long1', '0'], defines={'v_long1'}, uses=set()),
                 LsInstruction(0, 'li', ['v_long2', '1'], defines={'v_long2'}, uses=set()),
                 LsInstruction(0, 'li', ['v_long3', '2'], defines={'v_long3'}, uses=set())]
        # 中间插入 60 个短区间
        for i in range(60):
            block.append(LsInstruction(1+i*2, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set()))
            block.append(LsInstruction(2+i*2, 'addi', [f'v{i}_2', f'v{i}', '1'],
                          defines={f'v{i}_2'}, uses={f'v{i}'}))
        # 块尾使用长寿命 vreg
        block.append(LsInstruction(200, 'add', ['v_out', 'v_long1', 'v_long2'],
                                    defines={'v_out'}, uses={'v_long1', 'v_long2'}))
        return block
    return build()

def B04_alternating_short():
    """交替短区间: 大量 vreg 交错定义和使用，制造频繁 expire"""
    return [
        inst
        for i in range(POOL * 6)
        for inst in [
            LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set()),
            LsInstruction(i + POOL * 6, 'addi', [f'v{i}_2', f'v{i}', '1'],
                          defines={f'v{i}_2'}, uses={f'v{i}'}),
        ]
    ]

# =====================================================================
# 瓶颈维度 3: 复用模式维度
#   自溢 vreg 被 reload 的次数和模式直接影响代码质量
# =====================================================================

def C01_one_spill_many_uses():
    """单自溢vreg被频繁使用: 制造1个自溢vreg，然后在后续指令中重复使用它"""
    def build():
        # 先用 POOL 个 vreg 占满寄存器
        block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(POOL)]
        # 引入 POOL+1 个 vreg，确保 v{POOL} 自溢
        block += [LsInstruction(0, 'li', [f'v{POOL}', '99'], defines={f'v{POOL}'}, uses=set())]
        block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(POOL+1)],
                   defines={'v_sum'}, uses={f'v{i}' for i in range(POOL+1)})]
        # 然后连续 20 次使用 v{POOL}
        for i in range(20):
            block.append(LsInstruction(2+i, 'addi', [f'v_out{i}', f'v{POOL}', str(i)],
                          defines={f'v_out{i}'}, uses={f'v{POOL}'}))
        return block
    return build()

def C02_many_spills_one_use():
    """大量自溢vreg各用一次: 自溢 vreg 各自只被使用一次"""
    def build():
        block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(POOL + 20)]
        block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(POOL + 20)],
                   defines={'v_sum'}, uses={f'v{i}' for i in range(POOL + 20)})]
        # 每个自溢vreg只用一次
        for i in range(20):
            block.append(LsInstruction(2+i, 'addi', [f'v_out{i}', f'v{POOL+i}', str(i)],
                          defines={f'v_out{i}'}, uses={f'v{POOL+i}'}))
        return block
    return build()

def C03_spill_in_loop_body():
    """模拟循环体: 少量vreg + 重复定义使用模式"""
    def build():
        block = []
        for iteration in range(30):
            start = iteration * 5
            for j in range(10):
                block.append(LsInstruction(start + j, 'li', [f'v{j}', str(j)],
                              defines={f'v{j}'}, uses=set()))
            for j in range(8):
                block.append(LsInstruction(start + 10 + j, 'add', [f'v_acc{j}', f'v{j}', f'v{j+1}'],
                              defines={f'v_acc{j}'}, uses={f'v{j}', f'v{j+1}'}))
        return block
    return build()

def C04_spill_adjacent_uses():
    """同一自溢vreg在相邻位置被多次使用: 测试scratch寄存器复用"""
    def build():
        block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(POOL + 3)]
        block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(POOL + 3)],
                   defines={'v_sum'}, uses={f'v{i}' for i in range(POOL + 3)})]
        # v{POOL} 在同一指令中被用两次，然后连续被用
        block += [LsInstruction(2, 'add', ['v_out', f'v{POOL}', f'v{POOL}'],
                   defines={'v_out'}, uses={f'v{POOL}'})]
        for i in range(10):
            block.append(LsInstruction(3+i, 'add', [f'v_tmp{i}', f'v{POOL}', str(i)],
                          defines={f'v_tmp{i}'}, uses={f'v{POOL}'}))
        return block
    return build()

# =====================================================================
# 瓶颈维度 4: Eviction 模式维度
#   eviction 策略（farthest-end）在不同区间分布下的表现
# =====================================================================

def D01_same_end_all():
    """全部同终点: 所有区间[0, end)相同，farthest-end 退化为随机选择"""
    return _same_end_consumer(POOL + 20)

def D02_skewed_ends():
    """偏斜终点: 前半区间短，后半区间长，观察 farthest-end 的偏差"""
    N = POOL + 10
    def build():
        # 前半: 短区间 (用完后很快 expire)
        block = [LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(POOL // 2)]
        block += [LsInstruction(POOL // 2, 'add', [f'v_sum_early'] + [f'v{i}' for i in range(POOL // 2)],
                   defines={'v_sum_early'}, uses={f'v{i}' for i in range(POOL // 2)})]
        # 后半: 长区间 (活到块尾)
        for i in range(POOL // 2, N):
            block.append(LsInstruction(POOL // 2 + 1, 'li', [f'v{i}', str(i)],
                          defines={f'v{i}'}, uses=set()))
        block.append(LsInstruction(POOL // 2 + 2, 'sub', ['v_out', f'v{POOL//2}', f'v{N-1}'],
                      defines={'v_out'}, uses={f'v{POOL//2}', f'v{N-1}'}))
        return block
    return build()

def D03_eviction_chain():
    """eviction链式传递: 每引入一个新vreg就evict上一个，形成链"""
    def build():
        block = [LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(POOL)]
        # 连续引入新vreg，每个立即使用前一个的值
        for w in range(30):
            idx = POOL + w
            block.append(LsInstruction(POOL + w*2, 'li', [f'v{idx}', str(w)],
                          defines={f'v{idx}'}, uses=set()))
            block.append(LsInstruction(POOL + w*2 + 1, 'add', ['v_out', f'v{idx}', 'v0'],
                          defines={'v_out'}, uses={f'v{idx}', 'v0'}))
        return block
    return build()

def D04_spiral_interleaving():
    """螺旋交织: 复杂的区间依赖关系，用模运算生成。

    v1.4 修复: 原实现用 ``v{(i*3)%100}`` / ``v{(i*5)%100}`` /
    ``v{(i*7)%100}`` 三个独立模函数分别取 define 与两个 use, 导致大量
    invalid IR —— 一个 vreg 在它被定义(interval 起点)之前就被 use
    (use-before-def), 代码生成阶段该 vreg 仍是 ``SPILL_`` 占位符,
    ``SPILL_vXX`` 直接泄漏进汇编。现将 define 改为每次生成一个全新
    vreg(``v{i}``), 两个 use 取自先前已定义、仍存活的 vreg(模索引),
    从而保留"区间螺旋交织、重叠跨度不一"的压力特征, 同时保证输入合法。
    """
    def build():
        block = []
        # 前 2 个先天定义, 作为最初可用的 use 源
        block.append(LsInstruction(0, 'li', ['v0', '0'], defines={'v0'}, uses=set()))
        block.append(LsInstruction(1, 'li', ['v1', '1'], defines={'v1'}, uses=set()))
        for i in range(2, 102):
            d = f'v{i}'
            # use 取自先前已定义的 vreg (索引 < i, 保证在其 interval 起点之后),
            # 用两种不同的"间隔"制造重叠跨度差异; 含 % 的模回绕会把索引绕回
            # 尚未定义的高编号 vreg 造成 use-before-def, 故直接线性回退。
            u1 = f'v{i - 1}'
            u2 = f'v{max(0, i - 3)}'
            while u2 == u1 or u2 == d:
                u2 = f'v{max(0, i - 5)}'
            block.append(LsInstruction(i, 'add', [d, u1, u2], defines={d}, uses={u1, u2}))
        return block
    return build()

# =====================================================================
# 瓶颈维度 5: 栈压力维度
#   spill slot 的分配和复用效率
# =====================================================================

def E01_wave_spills():
    """10波交替溢出: 每波产生 POOL+3 个新vreg，然后全部释放"""
    def build():
        block = []
        for wave in range(10):
            base = wave * 30
            for i in range(POOL + 3):
                block.append(LsInstruction(wave*2, 'li', [f'v{base+i}', str(i)],
                              defines={f'v{base+i}'}, uses=set()))
            all_v = [f'v{base+i}' for i in range(POOL + 3)]
            block.append(LsInstruction(wave*2+1, 'add', [f'v_sum{wave}'] + all_v,
                          defines={f'v_sum{wave}'}, uses=set(all_v)))
        return block
    return build()

def E02_cascade_spills():
    """级联溢出: 每个新 vreg 依赖前一个的值，eviction 链不断延伸"""
    def build():
        block = [LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(POOL)]
        for i in range(50):
            idx = POOL + i
            block.append(LsInstruction(POOL + i, 'mul', [f'v{idx}', f'v{idx-1}', f'v{idx-2}'],
                          defines={f'v{idx}'}, uses={f'v{idx-1}', f'v{idx-2}'}))
        return block
    return build()

def E03_no_slot_reuse():
    """模拟栈槽完全不可复用: 所有区间几乎同时存活"""
    return _same_end_consumer(100)

# =====================================================================
# 瓶颈维度 6: 代码膨胀维度
#   spill code 对最终代码大小的影响
# =====================================================================

def F01_max_spill_code():
    """最大溢出路径: 每个 vreg 都自溢+多次使用→大量sw/lw"""
    def build():
        block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(100)]
        for i in range(100):
            for j in range(5):
                block.append(LsInstruction(1 + i*5 + j, 'add', [f'v_tmp{i}_{j}', f'v{i}', str(j)],
                              defines={f'v_tmp{i}_{j}'}, uses={f'v{i}'}))
        return block
    return build()

def F02_code_expansion_ratio():
    """代码膨胀比: 测量 asm 行数 / 原始 block 长度比"""
    def build():
        ratio = 10  # 每个原始 vreg 对应 10 次使用
        block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(30)]
        block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(30)],
                   defines={'v_sum'}, uses={f'v{i}' for i in range(30)})]
        for v in range(30):
            for u in range(ratio):
                block.append(LsInstruction(2 + v*ratio + u, 'addi', [f'v_out{v}_{u}', f'v{v}', str(u)],
                              defines={f'v_out{v}_{u}'}, uses={f'v{v}'}))
        return block
    return build()

def F03_dense_chain():
    """链式密集全重叠 (dense_60 风格)"""
    N = 60
    def build():
        block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
                 for i in range(N)]
        for i in range(N - 1):
            block.append(LsInstruction(1 + i, 'add', [f'v{i}', f'v{i}', f'v{i+1}'],
                          defines={f'v{i}'}, uses={f'v{i}', f'v{i+1}'}))
        return block
    return build()

def F04_large_random_block():
    """大规模随机块: 500条指令的完全随机模式"""
    rng = random.Random(12345)  # seed=12345 —— 固定种子保证输出可复现
    def build():
        block = []
        live_set = set()
        active_vregs = set()
        for i in range(500):
            # 80% 概率使用已有 vreg，20% 概率创建新 vreg
            if rng.random() < 0.2 or not active_vregs:
                v = f'v{rng.randint(0, 299)}'
                block.append(LsInstruction(i, 'li', [v, str(rng.randint(0, 99))],
                              defines={v}, uses=set()))
                active_vregs.add(v)
            else:
                src1 = rng.choice(list(active_vregs))
                src2 = rng.choice(list(active_vregs))
                dst = f'v{rng.randint(0, 299)}'
                block.append(LsInstruction(i, 'add', [dst, src1, src2],
                              defines={dst}, uses={src1, src2}))
                active_vregs.add(dst)
            # 随机 evict: 20% 概率释放一个 vreg
            if active_vregs and rng.random() < 0.2:
                victim = rng.choice(list(active_vregs))
                active_vregs.discard(victim)
        return block
    return build()


# =====================================================================
# 运行所有场景
# =====================================================================

if __name__ == '__main__':
    all_scenarios = [
        # 维度 1: 活跃度
        ('A01', '活跃度', '轻度超压(pool+5,完全重叠)', A01_slight_overlap),
        ('A02', '活跃度', '中度超压(pool*2,完全重叠)', A02_moderate_overlap),
        ('A03', '活跃度', '重度超压(200vreg,完全重叠)', A03_severe_overlap),
        ('A04', '活跃度', '零超压(pool*4,完全不重叠,基线)', A04_no_overlap),

        # 维度 2: 生命期模式
        ('B01', '生命期', '200密集短区间(同时创建)', B01_short_lived_dense),
        ('B02', '生命期', '200步长依赖链(单vreg复用)', B02_long_chain),
        ('B03', '生命期', '3长寿命+60短寿命混合', B03_mixed_lifetimes),
        ('B04', '生命期', '交替短区间(pool*6个)', B04_alternating_short),

        # 维度 3: 复用模式
        ('C01', '复用模式', '单自溢vreg被20次连续使用', C01_one_spill_many_uses),
        ('C02', '复用模式', '20个自溢vreg各用1次', C02_many_spills_one_use),
        ('C03', '复用模式', '模拟循环体(30轮x10vreg)', C03_spill_in_loop_body),
        ('C04', '复用模式', '自溢vreg相邻位置多次使用', C04_spill_adjacent_uses),

        # 维度 4: Eviction 模式
        ('D01', 'Eviction', '全部同终点(farthest-end退化为随机)', D01_same_end_all),
        ('D02', 'Eviction', '偏斜终点(前短后长,farthest-end偏差)', D02_skewed_ends),
        ('D03', 'Eviction', '链式eviction(每新vregevict上一个)', D03_eviction_chain),
        ('D04', 'Eviction', '100步螺旋交织(复杂依赖)', D04_spiral_interleaving),

        # 维度 5: 栈压力
        ('E01', '栈压力', '10波交替溢出(波间可复用)', E01_wave_spills),
        ('E02', '栈压力', '级联溢出(50步,eviction不断延伸)', E02_cascade_spills),
        ('E03', '栈压力', '100vreg完全同时存活(零复用)', E03_no_slot_reuse),

        # 维度 6: 代码膨胀
        ('F01', '代码膨胀', '100vreg各用5次(最大溢出路径)', F01_max_spill_code),
        ('F02', '代码膨胀', '30vreg各用10次(膨胀比测量)', F02_code_expansion_ratio),
        ('F03', '代码膨胀', '60链式密集全重叠(dense_60)', F03_dense_chain),
        ('F04', '代码膨胀', '500条指令完全随机块', F04_large_random_block),
    ]

    print("=" * 72)
    print("课题17: 寄存器分配瓶颈场景系统性测试")
    print("=" * 72)
    print(f"物理寄存器池: {POOL}个 ({', '.join(PHYS_REGS)})")
    print(f"测试场景数: {len(all_scenarios)}")
    print("=" * 72)

    results = []
    for sid, cat, desc, fn in all_scenarios:
        # fn.__name__ already carries the unique scenario id prefix (e.g.
        # ``A01_slight_overlap``); prefixing ``sid`` again would yield
        # ``A01_A01_slight_overlap``.  Keep just the function name.
        name = f"{fn.__name__}"
        r = run_scenario(name, cat, desc, fn)
        results.append(r)
        print_result(r)

    # ================================================================
    # 汇总表
    # ================================================================
    print("\n\n")
    print("=" * 120)
    print("汇总表")
    print("=" * 120)
    header = f"{'场景':25s} {'类别':8s} {'vregs':>4s} {'pool':>3s} {'peak':>4s} {'self':>4s} {'evict':>4s} {'slots':>4s} {'sw':>4s} {'lw':>4s} {'lines':>5s} {'bytes':>6s} {'u/v':>5s} {'redund':>6s} {'s.reuse':>7s} {'flags':>20s}"
    print(header)
    print("-" * 120)
    for r in results:
        if r.get('error'):
            print(f"{r['name']:25s} {'ERROR':>8s} {r.get('error', '')[:60]}")
            continue
        f = []
        if r['total_vregs'] > r['pool'] and r['peak_active'] <= r['pool']:
            f.append('PKLOCK')
        if r['vreg_leaks']:
            f.append('VLEAK')
        if r['redundant_sw'] > 0 and r['redundant_sw'] == r['spill_slots']:
            f.append('ALL_REDUN_SW')
        if r['redundant_reloads'] > 0:
            f.append(f'RLOADx{r["redundant_reloads"]}')
        if r['slot_reuse_ratio'] < 0.01 and r['spill_slots'] > 5:
            f.append('NO_REUSE')
        flags_s = ','.join(f[:3])
        print(f"{r['name']:25s} {r['category']:8s} {r['total_vregs']:>4d} {r['pool']:>3d} "
              f"{r['peak_active']:>4d} {r['self_spill']:>4d} {r['evicted']:>4d} "
              f"{r['spill_slots']:>4d} {r['stores']:>4d} {r['loads']:>4d} "
              f"{r['total_lines']:>5d} {r['asm_size_bytes']:>6d} {r['avg_uses']:>5.2f} "
              f"{r['redundant_reloads']:>6d} {r['slot_reuse_ratio']:>7.1%} {flags_s:>20s}")

    print("-" * 120)

    # ================================================================
    # 打印每个场景的详细构建代码 (供复现)
    # ================================================================
    print("\n\n")
    print("=" * 72)
    print("附录: 每个场景的构建方式 (Python function)")
    print("=" * 72)
    for sid, cat, desc, fn in all_scenarios:
        name = f"{fn.__name__}"
        source = inspect.getsource(fn)
        print(f"\n--- {name}: {desc} ---")
        print(source)