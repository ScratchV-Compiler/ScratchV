# Topic 18 指令调度器SPEC 评审

> 状态：对早期 SPEC 和旧基线的历史评审。本文的“当前”均指当时评审环境，不代表 2026-09-16 的实现。被评审的 SPEC 及文中提到的 Joska 详细设计未随仓库提供，不作为本版验收依赖。已采纳的范围调整及后续状态见 [README](README.md)；当前代码见 [代码说明](18-指令调度器代码说明.md)。

## 综述

本文对 `18-指令调度器SPEC.md`（下文称"SPEC"）进行独立技术评审。SPEC 提出在寄存器分配后、汇编发射前，对结构化 `MachineInstr` 执行基本块内列表调度。评审基于当前 ScratchV 代码仓（`/root/Lab/ScratchV` `main`）的实际状态。

| 文档 | 路径 | 版本 |
|---|---|---|
| **SPEC** | `18-指令调度器SPEC.md` | Proposed v0.1 (2026-07-25) |
| **代码基线** | `/root/Lab/ScratchV` | `main` |

---

## 课题意义：LLVM 指令调度与编译流程定位

### 为什么需要指令调度？

指令调度是编译器后端中**与微架构最接近的优化 pass**。它的核心目标是：在不改变程序语义的前提下，通过重排指令顺序来减少流水线停顿（pipeline stall），从而降低 CPI（Cycles Per Instruction）。

在现代处理器中，一条指令从取指到写回需要多个周期（典型的 RISC-V 5 级流水线：IF→ID→EX→MEM→WB）。当一条指令的结果被下一条指令立即使用时，流水线必须停顿等待结果就绪——这就是**数据冒险（data hazard）**。指令调度通过把不相关的独立指令插入到这些"等待槽"中，在不增加指令数的前提下隐藏延迟。

以经典的 load-use 场景为例：

```asm
; 调度前：4 周期，1 个 stall
lw   t0, 0(a0)    # @0 发射，结果 @2 就绪
add  t1, t0, t2   # @1 发射？不，t0 未就绪 → stall @1
                   # @2 发射 add
lw   t3, 4(a0)    # @3 发射

; 调度后：3 周期，0 个 stall
lw   t0, 0(a0)    # @0 发射
lw   t3, 4(a0)    # @1 发射（独立指令填充 load-use stall 槽）
add  t1, t0, t2   # @2 发射（t0 已就绪）
```

调度不减少指令总数（仍然是 3 条），也不改变数据流——它只是让独立指令占据原本空闲的发射槽。

### LLVM 中的指令调度

LLVM 作为工业级编译器，包含了多个不同粒度和层次的调度 pass，覆盖从 SelectionDAG 到机器指令的完整流程：

| LLVM Pass | 阶段 | 调度对象 | 目标 |
|---|---|---|---|
| `DAGCombiner` | SelectionDAG 构建中 | DAG 节点 | 简化 DAG、消除冗余节点 |
| `DAGScheduler` | SelectionDAG → MachineInstr | SelectionDAG 节点 | 确定指令选择顺序、暴露 ILP |
| `MachineScheduler` | MachineInstr 阶段（post-RA） | 机器指令（MachineInstr） | 减少流水线数据冒险 |
| `PostRAScheduler` | 寄存器分配后 | 物理寄存器指令 | 最后一轮调优 |
| `MachineBlockPlacement` | 机器指令阶段 | 基本块布局 | 改善分支预测局部性 |

其中 **MachineScheduler** 是真正意义上的"指令调度"——它在寄存器分配之后、汇编发射之前工作，此时所有指令都已绑定到物理寄存器，调度器拥有完整的 def-use 信息和精确的延迟模型。LLVM 的 MachineScheduler 支持多种调度策略（VLIW、Latency、ILP），并通过 `ScheduleDAG` + `SUnit` 数据结构构建依赖图，使用基于优先级的列表调度或回溯调度生成最终顺序。

### 调度前后的编译器在做什么

理解调度在编译流程中的位置，有助于看清它负责什么、不负责什么：

```
指令调度前（上游 pass）：
  ┌─ 指令选择（Instruction Selection）
  │   将目标无关 IR 转换为目标相关的 MachineInstr 或汇编指令。
  │   输出：指令序列（顺序接近 IR 遍历顺序，未优化）。
  │
  └─ 寄存器分配（Register Allocation）
      将虚拟寄存器映射到物理寄存器，插入 spill/reload 代码。
      输出：含物理寄存器的指令序列。
      ↑ 调度发生在此之后
      ↓
  调度器（Instruction Scheduling）
      分析依赖 → 构建 DAG → 计算优先级 → 列表调度。
      输出：重排后的指令序列（指令集合不变，顺序改变）。
      ↑ 调度发生后
      ↓
指令调度后（下游 pass）：
  ┌─ 汇编发射（Assembly Emission）
  │   将指令序列输出为汇编文本格式。
  │
  ├─ 窥孔优化（Peephole Optimization）[可选]
  │   修复调度可能暴露的局部次优模式。
  │
  ├─ 形式美化（Beautifier）[可选]
  │   调整缩进、对齐等不影响语义的格式。
  │
  └─ 周期估算 / 汇编（Cycle Estimation / Assembler）
      估算调度收益或生成最终二进制。
```

**关键理解**：
- **调度不改变指令集合**——它不删除、不添加、不替换指令。它只是重新排序。
- **调度的收益来源于流水线时序**——而不是指令数量。CPI 可以改善，指令数不变。
- **调度不能修复寄存器分配引入的 spill**——如果 RA 产生了多余的 load/store，调度只能重排它们，不能消除它们。
- **调度与架构无关**——通用的列表调度算法适用于任何单发射/多发射处理器，只需更换微架构模型参数。

### 为什么 ScratchV 需要指令调度？

ScratchV 当前的指令顺序接近 IR 遍历顺序，没有经过任何流水线感知的重排。这意味着：

1. **load-use 模式必然产生 stall**——`lw` 后紧跟使用该值的指令是最常见的可优化模式
2. **多周期指令（mul、div）后的独立指令没有被利用**——这些指令周围通常有空闲周期
3. **浮点运算的长延迟未被隐藏**——`fdiv.d` 延迟 16 周期，足够填充大量独立指令
4. **没有收益量化机制**——即使调度发生了改变，也没有周期估算来证明其收益

ScratchV 的指令调度不追求 LLVM 级别的调度复杂度（多发射、乱序、软件流水），但即使在单发射顺序流水线上，保守的列表调度也足以在典型计算密集型 kernel（如 Conv、GEMM、矩阵乘）中获得 5-15% 的 CPI 改善——这正是 Topic 18 的立项依据。

---

## 1. 总体评价

SPEC 是一份高质量的技术设计文档，架构分析深入、正确性约束严谨、对当前实现的诊断准确。但它存在显著的**工程可行性 gap**——设计的理论正确性在 ScratchV 当前代码状态下落地成本过高，且其核心主张（调度位置前移、线性分配器重构）在当前阶段不具备可执行条件。

| 维度 | 评分 | 说明 |
|---|---|---|
| **架构正确性** | 9/10 | 设计在正确位置（post-RA/pre-emission）是最优架构 |
| **代码仓吻合度** | 4/10 | 设计与代码仓默认配置（linear allocator）、pass 编排、MachineInstr 能力存在根本性 mismatch |
| **工程可执行性** | 4/10 | 需要先修改 3 个子系统才能开始实现核心调度逻辑 |
| **可测试性** | 8/10 | 11 条不变量条条可测试，验证器设计清晰 |
| **文档完整性** | 8/10 | 结构完整，但缺少与其他组件交互的具体代码级描述 |

---

## 2. 设计优点（值得肯定的部分）

### 2.1 对当前代码缺陷的诊断极其精准

SPEC 16.2 节对当前实现的分析是全文最有价值的部分。经代码仓验证：

| SPEC claim | 代码仓证据 | 严重性 |
|---|---|---|
| 生产调度发生在 AsmEmitter 之后 | `compiler.py:458-469` 确认 | 高 |
| 汇编 parser 根据 operand 位置猜测 def-use | `inst_scheduler.py:406-430`、`_asm_parser.py:245-278` | 高 |
| 没有机器基本块分区 | `inst_scheduler.py` 无任何 label/terminator 感知逻辑 | 高 |
| 构图仅实现 RAW 和 WAW | `inst_scheduler.py:186-207` 无 WAR、无 memory、无 control | 高 |
| cycle report 只累加 opcode latency | `inst_scheduler.py:319-324` `estimate_cycles` | 高 |
| 关键路径只检查 priority > 0 | `tests/test_inst_scheduler.py:157-158` | 中 |
| `machine_instrs_from_scheduled` 回退到 MV | `inst_scheduler.py:480-481` | 中 |

这 8 条诊断全部精确，直接可作为重构需求清单。

### 2.2 11 条不变量定义系统化

SPEC 第 10 节定义的 11 条正确性不变量（置换、边界、依赖、控制、结构、寄存器、确定性、禁用、报告、应用、失败原子性）是格式良好的正确性规格，可翻译为自动化测试。与详细设计的验证策略相比，SPEC 的优点是：

- 不变量与实现分离，可作为独立测试契约
- 失败原子性（第 11 条）确保部分调度结果不会污染输出
- 禁用不变量（第 8 条）保证 `--schedule=off` 时字节级不变

### 2.3 组件分解合理

SPEC 第 5 节将调度器分解为 7 个职责单一的组件，边界清晰。这比当前 `inst_scheduler.py` 的单类实现更可维护。

### 2.4 正确的调度位置分析

SPEC 4.1 节对 post-RA/pre-emission 位置的分析是正确的——这是工业编译器（LLVM、GCC）的标准做法。结构化 IR 调度优于文本正则解析，这是不容置疑的架构常识。

---

## 3. 设计与代码仓现状的 gap（核心问题）

### 3.1 默认寄存器分配器与 SPEC 的前提矛盾（致命 gap）

**前提**：SPEC 要求所有寄存器分配路径返回结构化 `MachineInstr` 列表（第 4.3 节）。

**现状**：`CompilerConfig` 默认 `reg_alloc="linear"`（`compiler.py:60`），而 linear-scan allocator 直接输出汇编文本：

```python
# compiler.py:407-413
if self.config.reg_alloc == "linear":
    from scratchv.backend.regalloc_linear import ...
    lsa = LinearScanAllocator()
    return lsa.emit(ls_insts)   # 返回 str，不是 list[MachineInstr]
```

**后果**：
- 在默认配置下，SPEC 方案无法启动
- SPEC 4.3 节的"配置验证拒绝 `--reg-alloc linear --schedule`"方案意味着：
  - 用户必须额外传递 `--reg-alloc greedy` 才能启用调度
  - 如果 `greedy` allocator 有 bug 或行为差异，用户会面临两难选择
- 要使 SPEC 完全可工作，需要重写 linear-scan 分配器的输出路径（约 200-300 行代码量级）

**严重性**：高。SPEC 应明确讨论此依赖的工程成本。

### 3.2 调度位置与 CompilerDriver 的集成 gap（架构级差异）

**SPEC 主张**：
```
IR → InstrSelect → RegAlloc → [Schedule] → AsmEmit
```

**当前代码**（`compiler.py:397-418`）：
```python
def _generate_riscv_linear(self, program) -> str:
    selector = InstructionSelector(program)
    machine_instrs = selector.run()
    # linear-scan: machine_instrs -> str
    ls_insts = block_from_machine_instrs(machine_instrs)
    lsa = LinearScanAllocator()
    return lsa.emit(ls_insts)  # 输出 str，MachineInstr 到此结束
```

`_generate_riscv_linear()` 返回 `str` 后进入 `_run_asm_passes()`。要插入 SPEC 的调度 pass，需要：
1. 让 linear-scan 返回 `list[MachineInstr]`（如 3.1 所述）
2. 在 `_generate_riscv_linear` 内部（而非 `_run_asm_passes` 中）增加调度调用
3. 修改 `_generate_riscv_dag` 做同样修改

SPEC 没有提供上述改造的具体代码级方案，只做了架构层面的描述。

### 3.3 MachineInstr 当前的语义不足以支持 SPEC 的需求（结构性 gap）

**SPEC 需要的 InstructionView**（6.2 节）：
```text
InstructionView
  ordered_operands
  explicit_defs / explicit_uses
  implicit_defs / implicit_uses
  memory_effect / memory_address
  control_effect / target
```

**当前 MachineInstr**（`machine_types.py:134-142`）：
```python
class MachineInstr:
    op: MachineOp
    dst: Optional[MachineOperand] = None
    src1: Optional[MachineOperand] = None
    src2: Optional[MachineOperand] = None
    comment: str = ""
```

差距不仅是"字段不足"，而是 MachineInstr 的语义角色本身就是错的（store 的 dst 实际上是值来源）。要适配 SPEC，需要先重构 MachineInstr（SPEC 4.1 也承认这一点），但这与 SPEC 自己的"调度器不应阻塞于 MachineInstr 重构"主张矛盾。

### 3.4 当前调度器位于 text-level 的实现债 gap

SPEC 提出的新调度位置（post-RA/pre-emission）在架构上正确，但当前代码中：

1. `_asm_parser.py:216-227` 的 `parse_asm()` 提供完整的 `ParsedAsmLine`（label、opcode、operands、comment、directive 分类）
2. `_asm_parser.py:245-278` 的 `classify_def_use()` 已经比 `inst_scheduler.py` 自己的 `parse_instructions()` 更完善

这意味着当前代码的文本解析基础已经比 SPEC 假设的更好。SPEC 说"当前汇编 parser 无法可靠处理 store/branch"，但 `_asm_parser` 的 `classify_def_use`（第 245 行）已经正确处理了 `_NON_DEF_OPCODES`（第 130 行），包括 store、branch、jump。

SPEC 对 text-level 路径的批评有道理，但低估了它的成熟度。

---

## 4. 文档内容的完整性质疑

### 4.1 线性扫描分配器的处理过于简单

SPEC 只在 4.3 节和验收标准 15.3 中提到了 linear-scan：

> "在 linear-scan 返回结构化结果之前，配置验证必须拒绝 `--reg-alloc linear --schedule`，不得静默跳过调度或回退到文本调度。"

这是"禁止使用"而非"改造方案"，但：
1. Linear-scan 是默认 allocator
2. 没有提供任何改造 linear-scan 的方案或工作量估计
3. 拒绝组合意味着 SPEC 的 V1 必须默认切换 allocator 或修改默认配置

### 4.2 `InstructionView` 的实现来源不清晰

SPEC 6.2 节提出了 `InstructionView`，但：
- 未说明 adapter 的具体实现位置（新文件还是复用 `machine_types.py`）
- 未讨论 `implicit_defs`/`implicit_uses` 如何从当前 `MachineInstr` 推断（call 的 clobber 集合需要 ABI 知识）
- 未说明 `memory_address` 的 `structured base/offset/symbol` 如何从 dst/src1 字段解析（`sw` 的地址分散在 dst 和 src1 中）

这些问题在详细设计中通过 opcode 角色模板解决，但 SPEC 将这些问题推迟到"先重构 MachineInstr"阶段。

### 4.3 验证失败处理策略过于严格

SPEC 第 11 节的失败处理策略是 **fail-stop**：

| 场景 | SPEC 行为 | 对教育编译器的适用性 |
|---|---|---|
| 依赖图有环 | 内部错误，编译失败 | ❌ 可选优化不应中断编译 |
| 验证发现依赖违规 | 编译失败，不输出部分结果 | ❌ 用户只损失到一个优化 |
| terminator 后存在同块指令 | 结构错误，编译失败 | ⚠️ 属于实现错误，可在开发期暴露 |
| 未知 opcode | barrier + 诊断 | ✅ |

教育/实验编译器的失败处理哲学应该是**安全降级**而非 **fail-stop**。SPEC 的策略复制了 LLVM/GCC 的生产质量要求，但不符合 ScratchV 的项目定位。

### 4.4 微架构模型的参数定义缺少具体化

SPEC 8 节定义了 `MicroArchitectureModel` 的接口形状和一大批 latency 表，但：

1. 未定义与 `PipelineCycleEstimator` 的共享契约
2. `result_latency` 与 `min_issue_distance` 的关系未明确定义
3. 未讨论 forwarding 如何影响模型（flush 和 stall 的区别）
4. 单发射 baseline 参数中 `beq=1`、`j=0`、`jal=0`、`ret=0` 的值为 0 表示什么语义不清晰

相反，详细设计 8.2 节的 opcode 表更完整地给出了 latency、resource、issue_gap、memory effect 四维信息，并且对伪指令约束（8.3 节）有专门讨论。

---

## 5. SPEC 与详细设计的对比定位

| 定位 | SPEC | 详细设计 |
|---|---|---|
| **文档角色** | 架构决策记录（ADR） | 工程实现说明书 |
| **更适合的阶段** | 长期架构规划（V2+） | V1 立即实现 |
| **最大贡献** | 诊断当前代码缺陷 | 提供可直接编码的数据结构和伪代码 |
| **最大问题** | 缺少集成方案细节 | 缺少架构一致性分析 |

SPEC 最适合作**项目的架构愿景文档（roadmap）**——定义"最终要建成什么样"，但不作为 V1 实现的唯一指南。详细设计更适合作为**V1 实现的技术规格**。

对比详细设计缺少什么？详细设计缺少 SPEC 第 16 节的"当前设计评议"——它对当前代码的 12 条缺陷分析是任何实现开始时都应该读的。如果详细设计的 V1 实现不读 SPEC 16 节，就可能重复当前调度器已经犯过的错误。

---

## 6. SPEC 中可从详细设计吸收的部分

如果 SPEC 要更新到 V0.2，建议吸收以下内容：

1. **具体数据类定义**（详细设计 7 节）：`DependencyKind`、`OpcodeSchedulingInfo`、`DAGNode`、`ScheduleConfig` 等
2. **集成位置妥协**：接受 V1 在 `_run_asm_passes()` 集成的现实，标注为"phase 1"而非终态
3. **局部回退策略**：接受而非禁止，作为 `--schedule-strict` 的可选强化
4. **具体测试用例**：详细设计 19.1 节的 20+ 个测试用例矩阵
5. **项目计划**：12 周实施计划 + 8 个独立提交建议

---

## 7. 结论

### 7.1 是否建议按 SPEC 实现？

**否。不建议将 SPEC 作为 V1 实现的唯一指南。**

SPEC 在**架构分析**上是卓越的——它对当前代码的 8 条缺陷诊断（16.2 节）精确无比。但在**工程可行性**上存在三个难以逾越的 gap：

1. 默认 allocator（linear-scan）不产生 MachineInstr
2. 当前 pass 编排不支持 post-RA/pre-emission 插入
3. MachineInstr 本身的语义重构是前置依赖

### 7.2 建议的实际使用方式

| 用途 | 价值 | 说明 |
|---|---|---|
| **当前代码缺陷清单** | 极高 | 16.2 节可直接作为重构 task list |
| **正确性不变量** | 高 | 第 10 节的 11 条不变量是验收标准的基础 |
| **V2 架构愿景** | 中 | 调度位置前移应进入 backlog；但不宜在 V1 实现 |
| **组件设计参考** | 中 | 第 5 节的 7 组件分解可指导代码结构 |
| **具体实现指南** | 低 | 缺少可直接编码的数据结构和伪代码 |

### 7.3 核心建议

1. **以详细设计为 V1 实现主体**，但吸收 SPEC 16.2 节的缺陷诊断作为实现前必须修复的 pre-condition
2. **SPEC 的 11 条不变量**应翻译为详细设计验证器的测试契约
3. **调度位置前移**列入 V2 backlog，依赖 MachineInstr 语义重构和 linear-scan 输出改造完成
4. **SPEC 本身**建议补充：linear-scan 改造方案、CompilerDriver 级别的集成伪代码、InstructionView 的具体 adapter 实现位置

---

*Review 生成日期：2026-07-28*
*评审范围：18-指令调度器SPEC.md v0.1*
*代码基线：ScratchV main HEAD*
*参考对比：18-指令调度器-详细设计-joska.md*
