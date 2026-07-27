# ScratchV 功能开发文档 — 课题17：寄存器分配（基本块内线性扫描）

> **文档版本**：v1.3  \
> **创建日期**：2026-07-13  \
> **最后更新**：2026-07-27（v1.3：修复 3 个 bug——alloc_map eviction 未更新、peak_active 被 pool 锁定、_pick_scratch 记忆缺失；新增 peak_real_pressure 指标；新增 _scratch_cache 优化 lw 复用）  \
> **作者**：Xi  \
> **关联 Issue**：无  \
> **涉及模块**：backend/（寄存器分配后端）

---

## 1. 功能概述与目标

### 1.1 背景与动机
- **现状问题**：ScratchV 当前代码生成阶段直接使用虚拟寄存器（vreg），没有经过物理寄存器的映射和分配。当指令数量增多或嵌套循环加深时，虚拟寄存器会无限增长，无法在有限物理寄存器（RISC-V x1-x31 中可用的 27 个）上执行。
- **应用场景**：所有需要将虚拟寄存器转换为实际 RISC-V 汇编的场景，尤其是 Conv2D 等 6 层嵌套循环中寄存器压力极大的算子。

### 1.2 功能描述
- **一句话定义**：为每个基本块内的虚拟寄存器分配 RISC-V 物理寄存器，寄存器不足时自动插入 spill code。
- **核心价值**：使 ScratchV 后端在真实硬件上能正确运行——虚拟寄存器不能直接编码为 RISC-V 指令的操作数，必须映射到物理寄存器。

### 1.3 目标与非目标

| 类型 | 内容 |
|------|------|
| ✅ 包含范围 | 实现基本块内的线性扫描寄存器分配：活跃区间计算、物理寄存器分配、溢出策略、spill code 生成 |
| ❌ 不包含范围 | 本次不实现跨基本块的全局寄存器分配（图着色算法）；不实现函数调用间的寄存器保存与恢复（callee/caller save 放在后续迭代） |

---

## 2. 设计与规格说明

### 2.1 用户视角（外部接口）

- **新增的 API**：

```python
from scratchv.backend.regalloc_linear import LinearScanAllocator, LsInstruction

# 构造基本块指令序列
block = [
    LsInstruction(0, "add", ["v1", "v2", "v3"], defines={"v1"}, uses={"v2", "v3"}),
    LsInstruction(1, "mul", ["v4", "v1", "v5"], defines={"v4"}, uses={"v1", "v5"}),
]

# 执行分配
allocator = LinearScanAllocator()
intervals = allocator.compute_live_intervals(block)
mapping = allocator.allocate(intervals)
code = allocator.get_allocated_code(block)

# 查看分配报告
print(allocator.report())
```

- **新增的命令行选项**：`--regalloc=linear`（集成到代码生成管线时使用）

### 2.2 内部设计（核心逻辑）

- **数据结构变更**：
  - `LiveInterval`：存储虚拟寄存器的 vreg 名、起始位置 `start`、结束位置 `end`、使用点集合 `uses`
  - `LsInstruction`：包装一条指令的序号、操作码、操作数、定义集、使用集
  - `LinearScanAllocator`：主分配器，包含活跃区间计算、线性扫描分配、spill code 生成

- **关键算法/流程**：

```
1. 遍历块内所有指令，为每个 vreg 计算 [start, end) 活跃区间
2. 按 start 从小到大对所有区间排序
3. 维护 active 有序列表（按 end 排序），表示当前活跃的区间
4. 顺序处理每个区间：
   a. 过期：从 active 中移除所有 end <= 当前 start 的区间，收回寄存器
   b. 分配：有空闲寄存器则分配
   c. 溢出：无空闲则选择 active 中 end 最晚的区间溢出，把寄存器给当前区间
5. 为被溢出 vreg 的所有定义后插入 sw、所有使用前插入 lw
6. 将所有 vreg 替换为对应 preg，输出分配后的指令序列
```

- **状态管理**：
  - `_spills`：被溢出的 vreg 集合
  - `_stack_slots`：vreg → 栈偏移的映射，用于生成 sw/lw 的地址
  - `_next_stack_offset`：下一个可用栈槽偏移量（从 0 向下增长）

### 2.3 接口定义（模块间交互）

- **上游依赖**：需要指令选择器（InstructionSelector）产出的 MachineInstr 序列，每条指令需标注 defines 和 uses 信息
- **下游影响**：寄存器分配的输出将传递给汇编发射器（AsmEmitter），替代原有的直接 vreg → 汇编模式

---

## 3. 模块修改与实现步骤

### 3.1 涉及的文件清单

| 文件路径 | 修改类型 | 修改内容概述 |
|----------|----------|--------------|
| `scratchv/backend/regalloc_linear.py` | 新增 | 实现线性扫描分配器：LiveInterval、LsInstruction、LinearScanAllocator |
| `scratchv/backend/instruction_select.py` | 修改 | 确保每条 MachineInstr 包含 defines/uses 字段 |
| `scratchv/standalone/onnx_to_riscv_standalone.py` | 修改 | 在代码生成管线中插入寄存器分配步骤，添加 `--regalloc=linear` 选项 |
| `tests/test_regalloc.py` | 新增 | 添加寄存器分配单元测试 |

### 3.2 分步实现计划

| 步骤 | 任务描述 | 预期产出 | 验证方式 |
|------|----------|----------|----------|
| 1 | 实现 LiveInterval 和 LsInstruction 数据结构 | 定义类及其方法 | 单元测试 |
| 2 | 实现 compute_live_intervals：遍历基本块计算活跃区间 | 能正确计算简单块的区间 | 手算对比验证 |
| 3 | 实现核心 allocate 方法：线性扫描 + 溢出策略 | 能分配物理寄存器并标记溢出 | 打印映射表验证 |
| 4 | 实现 spill code 生成：sw/lw 插入及栈槽管理 | 能为溢出变量生成正确加载/存储 | 检查生成的汇编 |
| 5 | 实现 vreg → preg 替换，输出最终指令序列 | 输出无虚拟寄存器的指令序列 | 检查汇编中无 vreg |
| 6 | 集成到 onnx_to_riscv_standalone.py，添加 `--regalloc=linear` | 完整管线可执行 | `make test` |
| 7 | 编写测试用例，覆盖溢出场景和边界条件 | 测试全部通过 | `python3 -m pytest tests/test_regalloc.py -v` |

### 3.3 异常处理与边界条件

- [ ] 输入为空的基本块时，分配器能否优雅跳过？
- [ ] 所有 vreg 都在一个指令内定义和使用（区间长度为 1）时，是否正常分配？
- [ ] 单个 vreg 的活跃区间跨越整个基本块时，是否会长期占用一个物理寄存器？
- [ ] 当物理寄存器数量不足以容纳所有同时活跃的 vreg 时，溢出策略是否能正确选择最合适的变量？
- [ ] x0（zero）和 ra 是否被正确保留，不会被分配出去？

---

## 4. 测试与验证方案

### 4.1 单元测试

- **测试文件位置**：`tests/test_regalloc.py`
- **至少覆盖以下场景**：
  1. 基本分配：3 个 vreg，2 个物理寄存器，无溢出
  2. 溢出场景：5 个 vreg，3 个物理寄存器，验证溢出选择正确
  3. 空基本块：无指令的块，应返回空结果
  4. 单指令块：定义+使用在同一个指令内
  5. 长活跃区间：一个 vreg 贯穿整个块

### 4.2 基准测试

- **测试位置**：`benchmarks/test_regalloc/`
- **至少准备 3 个用例**：
  1. 简单算术运算（3-5 个 vreg）
  2. 密集计算（20+ 个 vreg，触发溢出）
  3. 与现有 CNN 模型集成验证（对比分配前后的汇编正确性）

### 4.3 验收标准（Definition of Done）

- [ ] 所有新增单元测试通过
- [ ] `python3 -m pytest tests/` 全部通过（不引入回归）
- [ ] 至少 3 个基准用例正确输出分配后的汇编
- [ ] 生成的汇编中不存在任何 vreg 引用
- [ ] 溢出场景下正确生成了 sw/lw 指令且栈偏移正确
- [ ] 代码已添加必要的注释和文档字符串

---

## 5. 风险评估与依赖

| 风险项 | 影响程度 | 缓解措施 |
|--------|----------|----------|
| 溢出策略选择不当导致频繁溢出，严重影响性能 | 中 | 采用验证有效的"溢出 end 最晚"策略；提供扩展点便于后续尝试其他策略 |
| 栈槽分配冲突（多个溢出变量使用同一栈偏移） | 高 | 使用独立计数器分配栈槽，每个 vreg 独占一个槽位 |
| 现有代码未标注 defines/uses，导致活跃区间计算不准确 | 高 | 修改指令选择器，确保每条 MachineInstr 标注正确的 defines 和 uses |
| **生成本身溢出时使用了错误寄存器（v1.1 修复）** | **高** | 当前区间自溢时标记为 SPILL_ 不分配寄存器，消除非法寄存器引用 |
| **spill code 插入顺序错误（v1.1 修复）** | **中** | sw 按定义指令位置插入而非全部追加末尾 |

- **外部依赖**：仅依赖 Python 标准库，无外部依赖
- **对现有功能的兼容性**：通过 `--regalloc=linear` 开关控制，默认不启用，保证向后兼容

---

## 6. 开发进度跟踪

| 阶段 | 计划完成日期 | 状态 |
|------|--------------|------|
| 设计评审 | 2026-07-14 | ✅ v1.0 已完成 |
| 编码实现 | 2026-07-20 | ✅ v1.0 已完成 |
| 代码优化 | 2026-07-16 | ✅ v1.1：单遍扫描 O(N+V)、溢出逻辑修复、spill code 顺序修复 |
| 自测与调试 | 2026-07-22 | 进行中（344/344 通过，4 跳过） |
| 代码审查（PR） | 2026-07-25 | ⬜ |
| 合并主分支 | 2026-07-27 | ⬜ |

---

## 7. 附录

### 7.1 参考资料

- [ScratchV 课题17：寄存器分配指南](https://raw.githubusercontent.com/ScratchV-Compiler/ScratchV/main/docs/topics/17-%E5%AF%84%E5%AD%98%E5%99%A8%E5%88%86%E9%85%8D.md)
- Poletto & Sarkar (1999): Linear Scan Register Allocation (ACM TOPLAS)
- 龙书第 8.8 节：Register Allocation
- RISC-V ABI: [psABI Register Convention](https://github.com/riscv-non-isa/riscv-elf-psabi-doc)

### 7.2 测试用例详情

```
# 用例 1：简单算术运算
# 3 个 vreg，预期无需溢出
v1 = 3
v2 = 5
v3 = add(v1, v2)
return v3
# 预期输出：li + add 使用物理寄存器

# 用例 2：密集变量（触发溢出）
a1 = 1; a2 = 2; a3 = 3; a4 = 4; a5 = 5
a6 = 6; a7 = 7; a8 = 8; a9 = 9; a10 = 10
# ... 超过物理寄存器数量，触发溢出
# 预期输出：包含 sw/lw 指令

# 用例 3：与 Conv2D 算子集成
# 使用现有 CNN 模型编译，对比分配前后的汇编正确性
# 预期输出：最终二进制在 Spike 仿真器上输出正确结果
```

### 7.3 v1.1 代码优化

**优化 1：compute_live_intervals 单遍扫描**

原实现两重循环 O(V*N)，每遇到一个新的 vreg 就重新扫描整个基本块。改为单遍扫描：用一个 pass 同时记录每个 vreg 的首定义（first_def）和末次使用（last_use），复杂度降为 O(N+V)。

**优化 2：自身溢出时的寄存器占用修复**

原实现在当前区间比所有活跃区间都更晚结束时，使用 `phys_regs[0]` 作为 sw 的源寄存器。但 `phys_regs[0]` 可能正被某个活跃区间使用，导致汇编中读取了错误的值。修复为返回 None，由 allocate 标记为 SPILL_ 并从栈加载。

**优化 3：sw 插入位置修复**

原实现将所有 sw 追加到汇编末尾，导致溢出值的写回顺序与指令执行顺序不一致。修复为按指令位置插入：sw 跟在定义指令之后。同时新增 `_pick_scratch` 方法，为 SPILL_ 标记的变量选择合适的临时加载寄存器。

### 7.4 v1.2 peak_active 统计指标

**动机**：用户关心实际分配了多少个物理寄存器。`report()` 原有输出只能显示池大小和溢出数，无法回答"同一时刻最多有多少个 vreg 同时活跃"——这恰恰是寄存器压力的核心指标。

**实现**：在 `LinearScanAllocator.__init__` 中新增 `self.peak_active = 0`，在 `allocate()` 的每个区间分配/溢出之后（即 `active.append` 之后）记录 `len(active)` 的最大值。注意统计时机必须在分配决策完成后，因为过期后但分配前的 active 长度偏小。

```python
if len(active) > self.peak_active:
    self.peak_active = len(active)
```

**效果**：`report()` 新增两行输出：
- `Peak simultaneously active: N` — 扫描过程中 `active` 列表的最大长度
- `Physical regs actually assigned: N` — 实际被分配的不同物理寄存器数量

Conv2D 模拟 workload 实测：30 vreg, 27 preg 池, peak_active=17, 零溢出。说明真实 workload 的寄存器压力远低于理论上限（27）。

### 7.5 v1.3 Bug 修复与优化

**Bug A：eviction 后 alloc_map 未更新（严重·正确性）**

**根因**：`spill()` 在 eviction 分支中（第 361-374 行）将 victim 从 active 弹出、生成 sw、释放寄存器，但没有将 victim 在 `alloc_map` 中的条目改为 `SPILL_`。导致后续指令使用被 evict 的 vreg 时，`get_allocated_code()` 检查到 alloc_map 中是物理寄存器名（非 SPILL_ 开头），不生成 lw reload，直接使用该物理寄存器——但寄存器的值已被新 vreg 覆盖。

**修复**：在 `spill()` 的第 366 行追加 `self.alloc_map[spill_interval.vreg] = f"SPILL_{spill_interval.vreg}"`。

**验证**：修复前 5 个场景标记 `STORE_ONLY`（stores>0, loads=0），sw 写入栈后对应 vreg 永不 reload；修复后 stores == loads，全部消除。

---

**Bug B：peak_active 被 pool 锁定（中·诊断）**

**根因**：`peak_active` 基于 `len(active)` 计算。self-spill 的区间（`spill()` 返回 None）不执行 `active.append()`，因此 active 长度永远不超过 pool_size。但真实寄存器压力应包含这些自溢的区间。

**修复**：新增 `self.peak_real_pressure`，在峰值统计时额外计算 self-spill 的区间（`spilled_here=True` 时 `real_pressure += 1`）。

**效果**：`report()` 新增输出行：
- `Peak real pressure (incl. self-spill): N`

---

**Bug C：_pick_scratch 寄存器记忆缺失（低·代码质量）**

**根因**：`_pick_scratch` 每次为自溢的 SPILL_ vreg 选择不冲突的 scratch 寄存器，但不记住上次选了哪个。同一 vreg 在连续指令中被使用时，每次可能 reload 到不同的寄存器，浪费 lw。

**修复**：新增 `_scratch_cache: dict[str, str]` 缓存，`_pick_scratch` 接受 `vreg_hint` 参数，优先复用上次为该 vreg 选择的 scratch 寄存器（如果仍不冲突）。

---

### 7.6 未完成的优化方向

| 编号 | 方向 | 触发场景 | 预期效果 | 工作量 |
|------|------|----------|----------|--------|
| Opt 1 | 成本感知的溢出 victim 选择 | large_800x150（48 次 eviction） | sw+lw 减少 5-15% | ~20 行 |
| Opt 2 | 栈槽复用（slot reuse） | spiral_60（slot 复用率 0%） | 栈用量减少 30-50% | ~20 行 |
| Opt 3 | 自溢 sw 省略（无后续 use） | T05_all_overlap（evict 后无 use） | sw 减少最多 37% | ~10 行 |
| Opt 4 | 栈槽按区间结束回收 | 长区间锁死场景 | 栈槽数减少 30%+ | ~20 行 |