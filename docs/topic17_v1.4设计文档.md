# ScratchV 寄存器分配（基本块内线性扫描）技术设计文档

> 文档版本：v1.4  \
> 编写日期：2026-07-13  \
> 最后更新：2026-08-06（v1.4：在 v1.3.1 基础上补充修复——① 重定义写回路径补全：spilled vreg 纯重定义/多次重定义时新值未写回栈，redefine 判据改为 `d in self._spilled`；② 场景 D04 非法 use-before-def 输入修复，杜绝 `SPILL_vXX` 泄漏进汇编；③ `_pick_scratch` 冲突、`_evict_for_reload` 回退、多源 add 非法场景重构；模块重命名为 `regalloc_linear_v1_4.py`）  \
> 涉及模块：`regalloc_linear_v1_4.py`（寄存器分配器）、`instruction_select.py`（指令选择器）、`topic17_bottleneck_scenarios_v1.4.py`（回归场景）  \
> 功能范围：基本块内虚拟寄存器到物理寄存器的映射、线性扫描分配、溢出处理

---

## 一、功能介绍

### 1.1 功能概述

寄存器分配是编译器后端的关键环节。指令选择阶段为了方便，假设有无穷多个虚拟寄存器（vreg），但 RISC-V CPU 只有 32 个物理寄存器，排除 x0（恒零）、sp（栈指针）、gp（全局指针）、tp（线程指针）、ra（返回地址）后，仅 **27 个可分配**。

寄存器分配器负责：

1. 分析每个基本块内虚拟寄存器的活跃区间（live interval）
2. 用线性扫描算法将活跃区间映射到物理寄存器
3. 当物理寄存器不够时，选择最不紧急的变量溢出到栈上（spill），并自动插入 sw/lw 指令

### 1.2 设计目标

- **正确性优先**：任两个同时活跃的虚拟寄存器不能分配同一物理寄存器（寄存器冲突 = 死程序）
- **算法简洁**：线性扫描 O(n log n) 适用于基本块内的快速分配，避免图着色算法的复杂度
- **溢出策略合理**：溢出 end 最晚的区间，最大化单次溢出的收益
- **可集成**：分配器输出可直接替换指令序列中的 vreg，下游汇编发射器无需感知

---

## 二、设计思路

### 2.1 核心数据结构

#### LiveInterval

每个虚拟寄存器对应一个活跃区间：

| 字段 | 类型 | 说明 |
|------|------|------|
| `vreg` | str | 虚拟寄存器名（如 `"v1"`, `"v2"`） |
| `start` | int | 定义时的指令序号（包含） |
| `end` | int | 最后一次使用后 +1（半开区间 `[start, end)`） |
| `uses` | set[int] | 使用该寄存器的所有指令序号集合 |

**活跃区间含义**：一个 vreg 在 `[start, end)` 区间内是活跃的，即它已被定义且还可能被后续指令使用。一旦执行到指令 `end`，该 vreg 不再需要，其占用的物理寄存器可以被回收。

#### LsInstruction

包装基本块内的一条指令，供分配器分析：

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | int | 块内指令序号（从 0 开始） |
| `opcode` | str | 操作码 |
| `operands` | list[str] | 完整操作数列表 |
| `defines` | set[str] | 本指令定义的 vreg 集合 |
| `uses` | set[str] | 本指令使用的 vreg 集合 |

### 2.2 线性扫描算法

#### 算法流程

```
输入：排序后的活跃区间列表 intervals (按 start 升序)
输出：vreg → preg 映射表 mapping

active = []               // (end, preg, interval)，按 end 升序维护
free_regs = [x1...x31]    // 可用物理寄存器池

for each current in intervals:
    // 1. 过期检查
    for each (end, preg, iv) in active:
        if end <= current.start:
            从 active 中移除
            将 preg 归还到 free_regs
    
    // 2. 分配或溢出（核心逻辑）
    if free_regs 非空:
        preg = free_regs.pop()
        mapping[current.vreg] = preg
        active.append((current.end, preg, current))
    else:
        // 3. 溢出：选择 active 中 end 最晚的区间
        victim = active 中 end 最大的条目
        从 active 中移除 victim
        mapping[current.vreg] = victim.preg
        mapping[victim.vreg] = SPILLED   // 标记为溢出
        active.append((current.end, victim.preg, current))
    
    // 4. 跟踪峰值活跃数（分配/溢出之后）
    peak_active = max(peak_active, len(active))
```

#### 溢出策略说明

选择 active 中 **end 最大**（结束最晚）的区间溢出。理由是：

- 溢出一个很快结束的变量，刚溢出就要加载回来，得不偿失
- 溢出一个还要用很久的变量，省出的寄存器可以服务多个后续区间

这种贪心策略在基本块场景下效果良好，实现简单。

### 2.3 Spill Code 生成

被标记为溢出的 vreg，需要在：

- **每次定义后**插入 `sw preg, offset(sp)` — 将结果存到栈上
- **每次使用前**插入 `lw preg, offset(sp)` — 从栈加载回寄存器

栈槽分配采用简单顺序分配：每个溢出 vreg 获取一个独立槽位，偏移从 0 开始向下增长（RISC-V 栈向下生长）。

```
offset = next_stack_offset
stack_slot[vreg] = offset
next_stack_offset += 4
```

### 2.5 分配报告（report）

分配器提供 `report()` 方法输出分配结果摘要，包括：

- **Virtual registers allocated**：参与分配的虚拟寄存器总数
- **Stack spill slots used**：使用的栈槽数（溢出变量数）
- **Physical registers in pool**：可用的物理寄存器池大小
- **Peak simultaneously active**：扫描过程中 active 列表最大长度，即同一时刻最多有多少个区间同时活跃。这是衡量寄存器压力的核心指标——如果该值接近或超过池大小，说明寄存器压力大，可能触发溢出。
- **Physical regs actually assigned**：实际被分配出去的不同物理寄存器数量

示例输出：

```
Linear Scan Register Allocation Report
  Virtual registers allocated: 30
  Stack spill slots used: 0
  Physical registers in pool: 27
  Peak simultaneously active: 17
  Physical regs actually assigned: 27
```

### 2.6 物理寄存器池

可分配的 27 个物理寄存器（RISC-V x1-x31，排除特殊寄存器）：

| 组 | 寄存器 | 数量 | 说明 |
|----|--------|------|------|
| 参数/临时 | a0-a7（x10-x17）, t0-t6（x5-x7, x28-x31） | 15 | 调用者保存 |
| 保留 | s0-s11（x8-x9, x18-x27） | 12 | 被调用者保存 |
| **合计** | | **27** | 排除 x0, sp, gp, tp, ra |

当前版本仅实现基本块内分配，暂不区分 caller/callee save，所有 27 个寄存器统一入池。调用约定处理放在后续迭代。

---

## 三、修改模块与实现步骤

### 3.1 涉及文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `scratchv/backend/regalloc_linear.py` | **新增** | 线性扫描分配器主模块（基线版本） |
| `scratchv/backend/regalloc_linear_v1_4.py` | **新增**（继承自 `regalloc_linear_v1_3.py`，v1.4 重命名） | 分配器主模块：v1.3 新增 `peak_real_pressure`、`_scratch_cache`，修复 eviction 后 alloc_map 未更新、self-spill 寄存器污染、eviction 后 vreg 泄漏；v1.4 补全重定义写回路径（`d in self._spilled` 判据）、`_pick_scratch` 冲突、`_evict_for_reload` 回退 |
| `scratchv/backend/topic17_bottleneck_scenarios_v1.4.py` | **新增** | 23 个瓶颈场景回归脚本（v1.3.1 修复导入与指标解析；v1.4 重构 A01/A02/A03/D01/E03 多源 add 为合法累加链、重写 D04 消除 use-before-def） |
| `scratchv/backend/instruction_select.py` | 修改 | 确保每条 MachineInstr 带 defines/uses 标注 |
| `scratchv/standalone/onnx_to_riscv_standalone.py` | 修改 | 管线中插入寄存器分配步骤，添加 `--regalloc=linear` 选项 |
| `tests/test_regalloc.py` | 新增 | 寄存器分配单元测试 |

### 3.2 分步实现

#### Step 1：数据结构定义（LiveInterval, LsInstruction, LinearScanAllocator 骨架）

```python
@dataclass
class LiveInterval:
    vreg: str
    start: int
    end: int
    uses: set[int]

@dataclass
class LsInstruction:
    index: int
    opcode: str
    operands: list[str]
    defines: set[str]
    uses: set[str]
```

#### Step 2：活跃区间计算（O(N+V) 单遍扫描）

```python
def compute_live_intervals(self, block: list[LsInstruction]) -> list[LiveInterval]:
    # 单遍扫描：跟踪每个 vreg 的首定义和末次使用
    first_def: dict[str, int] = {}
    last_use: dict[str, int] = {}
    all_uses: dict[str, set[int]] = {}

    for inst in block:
        i = inst.id
        for d in inst.defines:
            if d not in first_def:
                first_def[d] = i
            if d in inst.uses:
                # 同一指令内定义并使用的 vreg
                all_uses.setdefault(d, set()).add(i)
                last_use[d] = max(last_use.get(d, -1), i + 1)
        for u in inst.uses:
            all_uses.setdefault(u, set()).add(i)
            last_use[u] = max(last_use.get(u, -1), i + 1)

    intervals = []
    for vreg in set(first_def) | set(last_use):
        start = first_def.get(vreg, 0)       # 块内无定义即为 live-in
        end = last_use.get(vreg, start + 1)  # 无使用则区间长度为 1
        intervals.append(LiveInterval(vreg, start, end, all_uses.get(vreg, set())))

    return sorted(intervals, key=lambda iv: iv.start)
```

**优化说明**：v1.1 将原 O(V*N) 的两重循环改为 O(N+V) 单遍扫描，同时消除了 define+use 同指令的冗余处理逻辑。

#### Step 3：核心分配循环

```python
def allocate(self, intervals: list[LiveInterval]) -> dict[str, str]:
    # Active list: (interval, phys_reg) 按 end 升序维护
    active: list[tuple[LiveInterval, str]] = []
    free_regs: list[str] = list(self.phys_regs)

    for interval in intervals:
        self._expire_old_intervals(active, interval.start, free_regs)
        if free_regs:
            reg = free_regs.pop(0)
            self.alloc_map[interval.vreg] = reg
            active.append((interval, reg))
        else:
            spill = self.spill(interval, active, free_regs)
            if spill is not None:
                # 踢出活跃区间中 end 最晚的，将其寄存器给当前区间
                self.alloc_map[interval.vreg] = spill
                active.append((interval, spill))
            else:
                # 当前区间自身被溢出（它是所有活跃区间中 end 最晚的）
                # 不分配物理寄存器，标记为 SPILL_，每次使用前从栈加载
                self.alloc_map[interval.vreg] = f"SPILL_{interval.vreg}"
    return dict(self.alloc_map)
```

**溢出策略**：选择 active 中 **end 最大**（结束最晚）的区间溢出。理由是：
- 溢出一个很快结束的变量，刚溢出就要加载回来，得不偿失
- 溢出一个还要用很久的变量，省出的寄存器可以服务多个后续区间

**溢出自身的情形**：当当前区间比所有活跃区间都更晚结束时，将其自身标记为 SPILL_ 而不分配任何物理寄存器。该变量的所有使用点都会插入 lw 加载到一个临时寄存器（通过 `_pick_scratch` 选择一个不与当前指令操作数冲突的寄存器）。

#### Step 4：Spill Code 生成

溢出变量需要生成额外的加载/存储指令：

- **被踢出的溢出**（spill 踢出活跃区间）：在该区间的定义点后插入 `sw reg, offset(sp)`，free_regs 回收该寄存器
- **自身溢出**（当前区间被标记为 SPILL_）：不分配寄存器，每次使用前插入 `lw scratch, offset(sp)`，每次定义后插入 `sw scratch, offset(sp)`

vreg → preg 替换和 spill code 插入在 `get_allocated_code` 中统一完成：

```python
def get_allocated_code(self, block: list[LsInstruction]) -> str:
    lines: list[str] = []
    # 构建位置 → 指令覆盖表
    spill_loads: dict[int, list[str]] = {}
    spill_stores: dict[int, list[str]] = {}
    for pos, op, operand in self.spill_code:
        target = spill_loads if "lw" in op else spill_stores
        target.setdefault(pos, []).append(f"  {op} {operand}")

    for inst in block:
        # 使用前插入 lw
        if inst.id in spill_loads:
            for line in spill_loads[inst.id]:
                lines.append(line)
        # SPILL_标记的变量：按需加载到临时寄存器
        rename = dict(self.alloc_map)
        for u in inst.uses:
            if str(rename.get(u, "")).startswith("SPILL_"):
                slot = self._spill_slots[u]
                scratch = self._pick_scratch(inst, rename)
                lines.append(f"  lw {scratch}, {slot}(sp)  # reload {u}")
                rename[u] = scratch
        lines.append(inst.to_asm(rename))
        # 定义后插入 sw（被溢出变量的持久化）
        if inst.id in spill_stores:
            for line in spill_stores[inst.id]:
                lines.append(line)
        for d in inst.defines:
            if d in self._spill_slots and str(rename.get(d, "")) in self.phys_regs:
                lines.append(f"  sw {rename[d]}, {self._spill_slots[d]}(sp)  # spill {d}")
    return "\n".join(lines)
```

**v1.1 修复说明**：原实现将 sw 全部追加到输出末尾而不是定义指令之后，导致溢出值不能及时写回栈。新实现使用 `spill_stores` 表按位置插入 sw，同时处理了 SPILL_ 标记变量的临时寄存器加载。

#### Step 5：集成到编译器管线

在 `onnx_to_riscv_standalone.py` 中，指令选择之后、汇编发射之前插入：

```python
if args.regalloc == "linear":
    allocator = LinearScanAllocator()
    intervals = allocator.compute_live_intervals(block)
    mapping = allocator.allocate(intervals)
    asm_code = allocator.get_allocated_code(block, mapping)
```

添加命令行参数 `--regalloc=linear`。

### 3.3 边界条件处理

1. **空基本块**：无指令的块，`compute_live_intervals` 返回空列表，`allocate` 直接返回空映射
2. **单个指令**：指令定义并使用了 vreg，区间 `[i, i+1)`，分配正确
3. **未使用的定义**：某个 vreg 被定义但后续从未使用（uses 为空），区间为 `[i, i+1)`，无溢出需求
4. **所有寄存器耗尽**：当活跃区间数超过 27 时触发溢出，确保不发生寄存器冲突

---

## 四、测试设计

### 4.1 单元测试覆盖场景

| 测试 | 描述 | 预期 |
|------|------|------|
| 基础分配 | 3 个 vreg 在 2 个 preg 上，无溢出 | vreg → preg 映射正确，无冲突 |
| 溢出触发 | 5 个 vreg 在 3 个 preg 上，频繁溢出 | 正确选择 end 最晚的溢出 |
| 空基本块 | 空指令列表 | 返回空映射 |
| 单指令块 | 定义+使用在一条指令内 | 区间正确，分配正常 |
| 长活跃区间 | 一个 vreg 贯穿整个块 | 物理寄存器被长期占用 |

### 4.2 基准用例

1. **简单算术**：3-5 个 vreg 的基本算术运算，验证无溢出时的分配正确性
2. **高密度变量**：30 个 vreg 在 5 个物理寄存器上运行，验证溢出逻辑
3. **CNN Conv2D 集成**：将分配器接入 CNN 编译管线，验证生成的汇编能被 Spike 仿真正确执行

### 4.3 验收标准

- [ ] 所有单元测试通过
- [ ] `make test` 全部通过（不引入回归）
- [ ] 生成的汇编中不存在 vreg 引用
- [ ] 溢出场景下正确插入 sw/lw，栈偏移无冲突
- [ ] 在 Spike 仿真器上，分配前后的二进制输出一致

---

## 五、附录

### 5.1 示例：活跃区间计算

**输入指令序列**（3 条指令，3 个 vreg）：

```
0: ADD v1, v2, v3     defines={v1} uses={v2, v3}
1: MUL v4, v1, v5     defines={v4} uses={v1, v5}
2: ADD v6, v4, v1     defines={v6} uses={v4, v1}
```

**输出活跃区间**：

| vreg | start | end | uses | 说明 |
|------|-------|-----|------|------|
| v2 | 0 | 1 | {0} | 只在指令 0 被使用 |
| v3 | 0 | 1 | {0} | 只在指令 0 被使用 |
| v1 | 0 | 3 | {1, 2} | 从指令 0 活跃到指令 2 |
| v5 | 1 | 2 | {1} | 只在指令 1 被使用 |
| v4 | 1 | 3 | {2} | 从指令 1 活跃到指令 2 |
| v6 | 2 | 3 | {} | 只有定义无使用 |

### 5.2 代码优化记录（v1.1）

v1.1 对核心实现进行了以下优化和修复：

| 编号 | 问题 | 修复方式 | 影响 |
|------|------|----------|------|
| 1 | `get_allocated_code` 中 sw 全部追加到末尾 | 改为按定义指令的位置插入 sw | 修复溢出值写回时序错误 |
| 2 | 当前区间被自身溢出时使用了 `phys_regs[0]`（可能被占用） | 改为返回 None，allocate 中标记为 SPILL_ 且不分配寄存器 | 消除非法寄存器使用 |
| 3 | `compute_live_intervals` 复杂度 O(V*N) | 改为单遍扫描 O(N+V) | 对大型基本块提升 ~2x |
| 4 | define+use 同指令的冗余处理 | 合并到单遍扫描逻辑中 | 代码更简洁 |

验证：优化后 344 个全量测试全部通过，4 个跳过（依赖 onnxruntime），无回归。

### 5.3 v1.2 新增指标：peak_active

在 `allocate()` 中新增 `self.peak_active` 字段，在每次过期检查后、分配之前记录 `len(active)` 的最大值。该值直接反映整个扫描过程中同时活跃区间数的峰值，即**寄存器压力的上界**。

`report()` 同步新增两行输出：
- `Peak simultaneously active: N` — 峰值活跃数
- `Physical regs actually assigned: N` — 实际被分配的不同物理寄存器数

Conv2D 模拟 workload 实测结果：30 个 vreg，27 个物理寄存器池，峰值同时活跃仅 17，物理寄存器实际分配 27，零溢出。

### 5.4 v1.3 Bug 修复与优化

v1.3 修复了 3 个 bug，新增 2 个优化：

| 编号 | 类型 | 问题 | 修复方式 | 影响 |
|------|------|------|----------|------|
| 1 | Bug A（严重·正确性） | `spill()` eviction 后 `alloc_map` 未更新为 `SPILL_` | 在 eviction 分支中追加 `self.alloc_map[spill_interval.vreg] = f"SPILL_{spill_interval.vreg}"` | 修复被 evict 的 vreg 后续使用时不生成 lw reload，直接使用被污染寄存器的问题 |
| 2 | Bug B（中·诊断） | `peak_active` 被 pool size 锁定，不包含 self-spill | 新增 `peak_real_pressure` 字段，统计时包含 self-spill 的区间 | 23 场景测试中部分场景真实压力比之前显示高 100%+ |
| 3 | Bug C（低·代码质量） | `_pick_scratch` 每次选不同 scratch 寄存器 | 新增 `_scratch_cache` 缓存，同一 vreg 尝试复用上次选择的 scratch 寄存器 | 自溢 vreg 连续使用时可减少冗余 lw |
| 4 | 优化 | 测试框架中 eviction/self-spill 计数逻辑 | Bug A 修复后 evicted 和 self-spill 都用 `SPILL_` 前缀，改为从 `spill_code` 条目数区分 | eviction 计数准确 |

#### Bug A 详细说明

**场景复现**（2 寄存器 pool）：
```
v0=[0,11) → a0         # 分配到 a0
v1=[1,3)  → a1         # 分配到 a1
v2=[3,5)  → evict v0   # a0 被 v2 占用
  sw a0, -4(sp)         # 溢出 v0
  v2 → a0
  但 alloc_map[v0] 仍是 a0 !    ← Bug

后续使用 v0 时：不生成 lw，直接读取 a0
但 a0 存的是 v2 的值 → 错误
```

**修复**：在 `spill()` 的 eviction 分支中（第 366-374 行），active.pop 之后追加 `self.alloc_map[spill_interval.vreg] = f"SPILL_{spill_interval.vreg}"`。

验证：修复前 5 个场景标记 `STORE_ONLY`（有 sw 无 lw），修复后全部消失，stores == loads。

#### Bug B 详细说明

`peak_active` 的计算基于 `len(active)`，但 self-spill 的区间不会进入 active 列表（`spill()` 返回 None 时，allocate 直接标记 `SPILL_` 而不做 `active.append()`）。因此当所有活跃区间都 self-spill 时，`peak_active` 被 pool size 锁定。

**修复**：新增 `self.peak_real_pressure`，在统计时额外加上 self-spill 的区间。

### 5.5 v1.3.1 代码审查修复（PR #37 → 审查反馈落实）

v1.3.1 基于 PR #37 的 AI 代码审查反馈，经逐条核对确认并修复了 6 个问题（4 个在分配器、1 个模块命名、1 个场景脚本）。核心修复如下：

| # | 类型 | 问题（v1.3 实现） | 修复（v1.3.1） |
|---|------|--------------------|----------------|
| 1 | 可导入性（严重） | 文件名 `regalloc_linear_v1.4.py` 含 `.`，Python 无法 `import` | 重命名为 `regalloc_linear_v1_4.py` |
| 2 | 正确性（严重） | self-spill 写死 `temp_reg = phys_regs[0]`，在所有寄存器被占时污染仍存活的 vreg | self-spill 分支改为统一 evict 结束最晚的活跃区间，将寄存器让给 `current` |
| 3 | 正确性（严重） | `_evict_for_reload` 用 `del rename[vreg]` 永久删除映射，导致 vreg 泄漏进汇编 | 改为 `rename[vreg] = "SPILL_" + vreg` 降级，并让 reload/evict 跳过非物理寄存器条目 |
| 4 | 代码质量（低） | `compute_live_intervals` 冗余 define+use 分支 | 删除重复分支 |
| 5 | 代码质量（低） | `import machine_types` 在函数内重复执行 | 上移到模块级 |
| 6 | 可运行性（严重） | 场景脚本 import 错误、按旧 `spill_code` 列表接口解析、重复指令 id | 改 import v1_3、重写指标解析适配 dict 结构、`_renumber()` 统一 id |

#### Fix 2 详细说明：self-spill 寄存器污染

**场景**（寄存器全被占用，且 `current` 活得比所有活跃区间都久）：
```
v_status=[0,11) → a0   # 1号活跃区间
v_other=[1,5)  → a1
v_new=[3,100)          # 定义时间为 3，活到 100，远超所有活跃区间
# 无空闲寄存器。v1.3 self-spill 分支执行：
temp_reg = phys_regs[0]  # = a0
alloc_map[v_new] = a0    # v_new 永久占用 a0
spill_code[3] += "sw a0, slot(sp)"
# 但 a0 此刻仍存 v_status 的值（v_status 活跃到 10）！
# → 污染：v_status 后续读 a0 得到的是 v_new（或 undefined）
```

**修复**：当 `free_regs` 为空时，不再尝试 self-spill（避免写死寄存器），而是统一 evict「结束最晚」的活跃区间，把其寄存器让给 `current`。由于 `current` 活得比所有活跃区间都久，让出寄存器后 `current` 会把它保留到块尾，被 evict 的 victim 仅在需要时按需 reload。这既消除了正确性缺陷，又比反复 reload 长生命 `current` 更高效。

#### Fix 3 详细说明：eviction 后 rename 泄漏

**根因**：`get_allocated_code()` 用一份向前遍历、随指令修改的 `rename`（初值 = `dict(alloc_map)`）。`_evict_for_reload` 通过 `del rename[victim]` 把 victim 从映射中移除。若 victim 之后出现定义/使用且没有对应的 reload 路径，`inst.to_asm(rename)` 中 `rename.get(vreg, vreg)` 回退为原始 `vXX` 名字，把虚拟寄存器泄漏进最终汇编（B01/E01/F04 实测复现）。

**修复**：victim 不删除而降级为 `SPILL_<victim>` 标记，使其后续定义走 scratch 重命名路径、后续使用触发 reload。由于该标记不代表物理寄存器，在 `_pick_reload_reg`/`_evict_for_reload` 中需跳过非物理寄存器条目。验证：泄漏场景消失，23/23 场景无泄漏、无寄存器冲突。

#### v1.3.1 验证结果（23 个瓶颈场景回归）

独立运行 `python scratchv/backend/topic17_bottleneck_scenarios_v1.4.py`，23 个场景全部通过：
- **无 vreg 泄漏**：生成汇编中不存在未映射的虚拟寄存器名
- **无寄存器冲突**：无两个同时存活的区间被分配同一物理寄存器
- 单元测试 `tests/test_regalloc_linear.py` 全部通过（18 passed，无回归）

> 设计边界说明：跨基本块 liveness（live-in/live-out）与多定义 vreg 属设计边界而非本次 bug，列入后续优化（见开发文档 7.8 Opt 5）。

### 5.6 v1.4 设计补充

v1.4 对分配器与场景脚本的补充修复（对应开发文档 7.7 Fix 7-11）。

#### Fix 7 详细说明：重定义写回路径补全（spill redefine lost）

**根因**：v1.3.1 的 Bug A 修复只覆盖 define+use 重叠（`v = v op v`）这一条路径——该路径中 v 的 use reload 已把 `rename[v]` 更新为物理寄存器（如 `a0`），`get_allocated_code` 的 redefine 分支条件 `rename[d].startswith("SPILL_")` 因此不触发（`rename[v]` 非 `SPILL_` 前缀），但 `to_asm` 需要 `rename[v]` 作为读写位置，新值写入 `a0` 后未写回栈。当 vreg 在 redefine 后仍 live 并再被使用，reload 会从栈读到旧值。

更隐蔽的第二条路径（本次确认）：spilled vreg 被**纯重定义**（`li v0, 999`，src 不含 v0）。首次重定义通过 redefine 分支选中 scratch 并把 `rename[v0]` 更新为物理寄存器；同一 vreg 若**再次**被重定义，此时 `rename[v0]` 已是物理寄存器、不再是 `SPILL_` 前缀，redefine 分支漏检 → 新值再次不写回栈、丢失。

**修复设计**：redefine 判据由 `rename[d].startswith("SPILL_")` 改为 `d in self._spilled`（成员判定，不受 `rename` 降级状态影响）。对新定义的每一条 spilled vreg，若 `rename[d]` 尚未落到物理寄存器则先用 `_pick_scratch`（避让 `live_regs`）选定，随后在 `inst.to_asm` **之后**（此刻 `rename[d]` 持有新计算值、无中间 clobber）追加 `sw rename[d], slot(sp)` 写回。

#### Fix 11 详细说明：D04 use-before-def 非法输入

**根因**：D04 原实现以 3 个独立模函数 `v{(i*3)%100}`/`v{(i*5)%100}`/`v{(i*7)%100}` 分别选 define 与两个 use。模回绕使某个 vreg 在它被定义（interval 起点）**之前**就被当作源操作数使用（实测 99 处 use-before-def）。这些 use 的 vreg 在代码生成时仍未从 `SPILL_` 降级解除（无对应 reload），`to_asm` 把 `SPILL_vXX` 字面量泄漏进汇编（如 `add t4, SPILL_v60, t6`）。

**修复设计**：重写 D04 builder——define 每次生成全新 vreg `v{i}`，两个 use 取严格前向（索引 `< i`）的已定义 vreg（`v{i-1}`、`v{max(0,i-3)}`），消除 use-before-def 的同时保留"螺旋交织、重叠跨度不一"的寄存器压力特征。

#### v1.4 验证结果（23 个瓶颈场景回归）

独立运行 `python scratchv/backend/topic17_bottleneck_scenarios_v1.4.py`，23 个场景全部通过：
- **无泄漏**：D04 修复后生成的汇编不再出现 `SPILL_` 占位符泄漏（原 9 处，现 0 处）；其余 22 场景本就干净。
- **无寄存器冲突**：无两个同时存活的区间被分配同一物理寄存器。
- **语义仿真**：针对 Fix 7 的 redefine-after-spill、同 vreg 双重重定义用例在单/双寄存器池下 0 错误；A-F 全部场景 0 错误。
- 单元测试 `tests/test_regalloc_linear.py` 18 passed；全量 `pytest tests/` 342 passed（2 个失败均为 `test_simulator.py` 的 tinyfive 环境问题，与本次 PR 无关）。

### 5.7 参考资料

- ScratchV 项目文档：`docs/topics/17-寄存器分配.md`
- Poletto & Sarkar (1999): *Linear Scan Register Allocation*, ACM TOPLAS
- 龙书第 8.8 节：Register Allocation
- RISC-V psABI: https://github.com/riscv-non-isa/riscv-elf-psabi-doc