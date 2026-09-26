# ScratchV 统一控制流图（CFG）基础设施 — 设计文档

> **版本**: 2.0 | **读者**: 模块负责人、代码审查者、IR/Machine 优化 pass 开发者
> **主实现**: `scratchv/analysis/cfg.py` | **课题**: 11 — 控制流图生成器
> **状态**: 已实现（统一 CFG 核心） | **最后更新**: 2026-09-18

---

## 1. 问题定义

### 1.1 输入与输出

**输入**：

- IR 层：`scratchv.ir.types.Program` / `Function`
- Machine IR 层：单函数扁平 `list[MachineInstr]`

**输出**：每个函数一个 `ControlFlowGraph`（兼容别名 `CFG`），包含基本块节点、控制流边，以及可供活跃变量、常量传播、校验器使用的统一图查询接口。

### 1.2 实例

DSL：

```text
if (x > 0):
    y = add(x, 1)
else:
    y = sub(x, 1)
endif
return y
```

CFG 拓扑：

```text
entry --BRANCH(true)--> L_then --JUMP--> merge --(return, exit)
  |                                        ^
  +-------BRANCH(false)--> L_else --JUMP---+
```

### 1.3 设计目标

| 目标 | 含义 |
|------|------|
| **单一事实源** | IR 与 Machine IR 共用一套 CFG 数据结构与图算法 |
| **adapter 解耦** | 不同指令表示通过 `CFGAdapter` 接入统一 builder |
| **控制流规则明确** | label、terminator、fallthrough、branch target 契约统一 |
| **分析可复用** | 活跃变量使用反向数据流，常量传播使用前向数据流，二者复用同一 worklist 框架 |
| **诊断不静默** | 悬空目标、无效 entry、重复块名、非法 fallthrough 等给出明确错误 |
| **可可视化** | DOT 输出区分入口/出口/循环头，边类型样式稳定 |

### 1.4 非目标（当前范围）

- 不实现寄存器分配的 spill/reload 插入位置选择
- 不构造或破坏 SSA；Phi 相关接口保留但 IR 当前无 Phi
- 不直接修改原始 IR/Machine IR 列表；CFG 只报告事实

---

## 2. 架构定位

### 2.1 管线位置

```text
ONNX / DSL
     │
     ▼
IR Program ────▶ IRCFGAdapter ─────┐
                                   ├──▶ build_cfg ──▶ ControlFlowGraph
MachineInstr ─▶ MachineCFGAdapter ─┘          │
                                              ├──▶ analyze_liveness
                                              ├──▶ ConstantPropagation
                                              └──▶ verify_cfg
```

### 2.2 模块布局

```text
scratchv/analysis/cfg.py             统一 CFG 数据结构 + builder + 图算法
scratchv/analysis/adapters.py        IRCFGAdapter / MachineCFGAdapter
scratchv/analysis/liveness.py        反向活跃变量分析
scratchv/analysis/dataflow.py        通用数据流求解 + 常量传播
scratchv/analysis/usedef.py          IR / Machine use-def provider
scratchv/analysis/cfg_validation.py  CFG 结构化诊断
scratchv/analysis/cfg_builder.py     旧路径兼容 shim
scratchv/ir/cfg.py                   旧路径兼容 shim
```

---

## 3. 数据结构设计

### 3.1 EdgeType

```python
class EdgeType(enum.Enum):
    FALLTHROUGH = "fallthrough"   # 顺序过渡
    BRANCH      = "branch"        # 条件分支，带 true/false 条件
    JUMP        = "jump"          # 无条件跳转
    CALL        = "call"          # 兼容保留；CALL 不是 terminator
```

### 3.2 CFGNode

```python
@dataclass
class CFGNode:
    name: str
    instructions: Any = 0          # builder 产物为 list[Instruction|MachineInstr]
    is_entry: bool = False
    is_exit: bool = False
    terminator_opcode: Optional[str] = None
    block_id: Optional[str] = None
    instruction_ids: list[str] = field(default_factory=list)
```

说明：

- `instructions` 保留旧接口的整数计数兼容；builder 产物始终为指令序列。
- `instruction_ids` 为 liveness 等分析提供稳定 `InstructionId`。
- `block_id` 当前默认等于 `name`。

### 3.3 CFGEdge

```python
@dataclass
class CFGEdge:
    source: str
    target: str
    edge_type: EdgeType = EdgeType.FALLTHROUGH
    condition: Optional[str] = None
```

### 3.4 ControlFlowGraph / CFG

```python
@dataclass
class ControlFlowGraph:
    function_name: str
    nodes: dict[str, CFGNode]
    edges: list[CFGEdge]
    entry: str
```

核心方法：

- `successors(block)` / `predecessors(block)`
- `reachable_nodes` / `unreachable_blocks`
- `to_dot(highlight_loops=False, loop_headers=None)`
- `verify_cfg(cfg)` 由 `cfg_validation.py` 提供

DOT 样式：

| 节点类型 | 颜色 |
|---------|------|
| 入口块 | `#90EE90` |
| 出口块 | `#FF6B6B` |
| 循环头 | `#87CEEB` |
| 普通块 | `lightyellow` |

| 边类型 | DOT 样式 |
|--------|---------|
| FALLTHROUGH | 黑色实线 |
| BRANCH | 蓝色虚线 + `[true]`/`[false]` |
| JUMP | 红色实线 |
| CALL | 紫色点线（兼容保留） |

---

## 4. 构建算法

### 4.1 CFGAdapter 协议

```python
class CFGAdapter(Protocol):
    @property
    def function_name(self) -> str: ...
    def blocks(self) -> Sequence[Any]: ...
    def block_name(self, block) -> str: ...
    def instructions(self, block) -> Sequence[Any]: ...
    def is_label(self, instr) -> bool: ...
    def is_terminator(self, instr) -> bool: ...
    def branch_targets(self, instr) -> Sequence[str]: ...
    def has_fallthrough(self, instr) -> bool: ...
    def opcode_name(self, instr) -> str: ...
```

### 4.2 控制流规则

| 指令/场景 | 规则 |
|-----------|------|
| LABEL | 块入口；不进入任何块的指令列表 |
| 条件分支 | target + fallthrough |
| J/JAL | 只有静态 target，不加 fallthrough |
| JALR/return | 无静态 successor |
| CALL | 非 terminator，保留 fallthrough |
| terminator 后指令 | 必须开启新基本块 |
| 空函数 | 生成唯一空 entry 块 |

### 4.3 FOR/ENDFOR 规范化

IR adapter 在构建 CFG 前将 `FOR/ENDFOR` 规范化为：

```text
load_const iv = start
br for_hdr
for_hdr:
br_if iv, end -> for_body, for_exit
for_body:
... body ...
add iv, iv, step
br for_hdr
for_exit:
```

这样 IR 与 Machine CFG 使用同一套显式 branch/label 语义，不再各自维护隐式循环规则。

### 4.4 支配集与直接支配者

```text
Dom(entry) = {entry}
Dom(n)      = {n} ∪ ⋂{Dom(p) | p ∈ predecessors(n)}  (n != entry)
```

- 支配集迭代：实际节点数 N < 100 时 3–5 轮收敛。
- idom 提取：对每个节点取支配集中“支配集合最大”的严格支配者，最坏 O(N²)。

### 4.5 自然循环检测

回边：`edge(a → b)` 且 `b ∈ Dom(a)`。

循环体：从 `source` 反向 BFS，遇 `header` 停止，访问节点加 `header` 组成 body；同一 header 的多条回边合并 body。

嵌套：若 `inner.header ∈ outer.body` 且 `inner.body ⊂ outer.body`，则 inner 为 outer 子循环，设置 `parent`、`children`、`nesting_depth`。

---

## 5. 分析接口

### 5.1 活跃变量

```python
def analyze_liveness(cfg, provider: UseDefProvider) -> LivenessResult
```

无 Phi：

```text
live_out[B] = ⋃ live_in[S]
live_in[B]  = uses[B] ∪ (live_out[B] - defs[B])
```

有 Phi：

```text
edge_live[B,S] = (live_in[S] - phi_defs[S]) ∪ phi_uses[B,S]
live_out[B]     = ⋃ edge_live[B,S]
```

结果包含：

- `blocks: Mapping[BlockId, BlockLiveness]`
- `edge_live: Mapping[(BlockId, BlockId), frozenset[ValueId]]`
- `live_before` / `live_after: Mapping[InstructionId, frozenset[ValueId]]`

### 5.2 数据流求解

`run_dataflow(cfg, analysis)` 提供 forward/backward 统一 worklist，`ConstantPropagation` 是基于该框架的前向分析。

常量 meet：

- `undefined ⊓ c = c`
- `c ⊓ c = c`
- `c ⊓ d = overdefined`
- `overdefined ⊓ x = overdefined`

### 5.3 CFG 校验

`verify_cfg(cfg)` 检查：

- entry 是否存在
- 边 source/target 是否存在
- terminator 是否位于块尾
- 无条件跳转后是否错误存在 fallthrough
- predecessor/successor 一致性

重复块名在 `build_cfg` 中直接抛出 `ValueError`，不会静默覆盖。

---

## 6. 已知限制

| 项 | 说明 | 优先级 |
|----|------|--------|
| Phi | 接口已保留，IR 当前无 Phi 构造 | 中 |
| 常量传播 | 只输出 fact，不直接折叠指令 | 中 |
| Machine liveness | 仅追踪 vreg；物理寄存器与 CALL clobber 分开暴露 | 中 |
| 调用图 | 当前每函数 CFG 独立构建 | 低 |

---

## 7. 相关文件索引

### 核心文件

| 文件 | 关系 |
|------|------|
| `scratchv/analysis/cfg.py` | 统一 CFG 核心 |
| `scratchv/analysis/adapters.py` | IR/Machine adapter |
| `scratchv/analysis/liveness.py` | 反向活跃变量 |
| `scratchv/analysis/dataflow.py` | 数据流框架 + 常量传播 |
| `scratchv/analysis/usedef.py` | use/def provider |
| `scratchv/analysis/cfg_validation.py` | CFG 校验 |
| `scratchv/analysis/cfg_builder.py` | 旧路径 shim |
| `scratchv/ir/cfg.py` | 旧路径 shim |

### 测试与示例

| 文件 | 关系 |
|------|------|
| `tests/test_cfg.py` | Topic 11 回归测试 |
| `tests/test_cfg_builder.py` | 旧 builder 回归测试 |
| `tests/test_unified_cfg.py` | 统一 CFG 基础设施测试 |
| `scripts/visualize_cfg.py` | DOT/PNG 可视化 |
| `examples/cfg/*.dsl` | if/else、while、nested、unreachable 示例 |

---

## 8. 验收标准

- 项目只有一个正式 CFG 核心实现：`scratchv/analysis/cfg.py`
- IR 与 Machine IR 共用数据结构与图算法
- `IRCFGAdapter` / `MachineCFGAdapter` 提供稳定控制流语义
- liveness 提供块级、边级、指令级结果
- `verify_cfg` 对非法结构返回明确诊断
- 精确节点/边/活跃集合测试通过
- 全量测试通过
