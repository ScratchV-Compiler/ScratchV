# ScratchV 控制流图（CFG）模块 — 设计文档

> **版本**: 1.0 | **读者**: 模块负责人、代码审查者、下游优化 pass 开发者
> **源文件**: `scratchv/ir/cfg.py`（待实现） | **课题**: 11 — 控制流图生成器
> **状态**: 设计中 | **最后更新**: 2026-08-02

---

## 1. 问题定义

### 1.1 输入与输出

**输入**: ScratchV IR 的 `Program` 对象（经过前端解析的 ONNX 或 DSL）。

**输出**: 每个 `Function` 对应一个 `CFG` 对象，包含基本块节点和控制流边。

### 1.2 实例

给定 DSL 源码：

```
if (x > 0):
    y = add(x, 1)
else:
    y = sub(x, 1)
endif
return y
```

经前端解析为 IR 指令后，CFG 模块应产出 4 个基本块和 4 条边：

```
entry --BRANCH(true)--> L_then --JUMP--> merge --(return, 出口)
  |                                       ^
  +-------BRANCH(false)--> L_else --JUMP--+
```

### 1.3 设计目标

| 目标 | 含义 |
|------|------|
| **正确的基本块划分** | 以 label/br/br_if/return 为边界，单趟线性扫描完成 |
| **准确的边类型** | FALLTHROUGH / BRANCH / JUMP / CALL（预留）四类 |
| **基础分析能力** | 不可达消除、支配树、自然循环检测 |
| **可可视化** | 输出 Graphviz DOT，节点着色区分入口/出口/循环头 |
| **零外部依赖** | 仅依赖 Python 标准库 + `scratchv.ir.types` |

### 1.4 非目标（当前版本不做）

- **控制依赖图** — 程序切片的前置需求
- **SSA 构造 / 破坏** — ScratchV IR 已有 SSA 形式
- **数据流分析**（到达定义、活跃变量）— 留给后续优化 pass
- **跨函数分析**（调用图）— 每个函数的 CFG 独立构建

---

## 2. 架构定位

### 2.1 在 ScratchV 管线中的位置

```
 ONNX / DSL
     │
     ▼
┌─────────┐     ┌───────────┐     ┌───────────┐
│ Frontend│────▶│ IR Program│────▶│ Optimizer │────▶ Backend
│ Parser  │     │ types.py  │     │ 5 passes  │
└─────────┘     └─────┬─────┘     └─────┬─────┘
                      │                 │
                      ▼                 │
                ┌──────────┐            │
                │   CFG    │◀───────────┘
                │ Builder  │
                └────┬─────┘
                     │
                     ▼
           ┌────────────────┐
           │  下游消费者:    │
           │ · LICM         │
           │ · 死代码消除    │
           │ · 寄存器分配    │
           │ · IR 验证器     │
           └────────────────┘
```

**核心定位**: CFG 是分析基础设施，位于 IR 层之上、优化 pass 之下。

### 2.2 模块依赖

```
scratchv/ir/cfg.py
  ├── 依赖: scratchv/ir/types.py
  ├── 被依赖: scratchv/optimizer/
  ├── 被依赖: scratchv/analysis/ir_verifier.py
  └── 外部依赖: 无（仅 Python 标准库）
```

---

## 3. 数据结构设计

### 3.1 EdgeType

```python
class EdgeType(enum.Enum):
    FALLTHROUGH = "fallthrough"   # 顺序执行过渡
    BRANCH      = "branch"        # 条件分支（带 condition 标签）
    JUMP        = "jump"          # 无条件跳转
    CALL        = "call"          # 函数调用（预留，IR 当前无此指令）
```

**设计决策**: 包含 CALL 作为预留类型，与课题 user guide 保持一致。

### 3.2 CFGNode

```python
@dataclass
class CFGNode:
    name: str                       # 块名（唯一标识）
    instructions: list[Instruction] # 对原列表的引用（非拷贝），外部修改原 IR 会反映到此列表
    is_entry: bool                  # 函数入口
    is_exit: bool                   # 以 return 结尾
    terminator_opcode: str | None   # 终止指令类型
```

### 3.3 CFGEdge

```python
@dataclass
class CFGEdge:
    source: str
    target: str
    edge_type: EdgeType
    condition: str | None  # "true"/"false"，仅 BRANCH 类型
```

### 3.4 CFG

```python
@dataclass
class CFG:
    function_name: str
    nodes: dict[str, CFGNode]  # dict 保证 O(1) 按名查找
    edges: list[CFGEdge]       # list 保证遍历效率
    entry: str                 # 入口块名
```

**DOT 可视化样式规范**（与课题 user guide 一致）：

节点样式:

| 节点类型 | 颜色 |
|---------|------|
| 入口块 | 绿色 (`#90EE90`) |
| 出口块 (return) | 红色 (`#FF6B6B`) |
| 循环头 | 蓝色 (`#87CEEB`) |
| 普通块 | 浅黄色 (`lightyellow`) |

边样式:

| 边类型 | DOT 样式 |
|--------|---------|
| FALLTHROUGH | 黑色实线 |
| BRANCH | 蓝色虚线 + `[true]`/`[false]` 标签 |
| JUMP | 红色实线 |
| CALL（预留） | 紫色点线 |

### 3.5 NaturalLoop

```python
@dataclass
class NaturalLoop:
    header: str                        # 循环头块名
    body: set[str]                     # 循环体中所有块名（含 header）
    back_edges: list[tuple[str, str]]  # (source, header) 回边列表
    parent: str | None                 # 外层循环 header
    children: list[str]                # 内层循环 headers
    nesting_depth: int                 # 嵌套深度（0 = 最外层）
```

**嵌套关系构建**: 在所有循环检测完毕后，对每对循环 (outer, inner)，若 `inner.header in outer.body` 且 `inner.body` 是 `outer.body` 的真子集，则 inner 嵌套于 outer。递归计算 nesting_depth = outer.depth + 1。

---

## 4. 算法设计

### 4.1 基本块划分

**输入**: `list[Instruction]`  **输出**: `list[tuple[name, list[Instruction]]]`

**算法**: 单趟线性扫描 O(n)

```
遍历指令:
  遇到 LABEL → 结束当前块；标签名 = 新块名
               LABEL 指令本身不加入任何块的指令列表，仅用作块名
  加入指令到当前块
  遇到 BR / BR_IF / RETURN → 结束当前块
```

**边界情况**: 空函数返回空 CFG；连续 label 产生空块（保留）；BR 后无 label 则自动命名 `b0`, `b1`...。

### 4.2 支配集计算

**算法**: 迭代不动点 (Iterative Fixed-Point)

**初始化**:
```
Dom(entry) = {entry}
Dom(n) = {所有节点}  (n != entry)
```

**迭代** (重复直到不再变化):
```
Dom(n) = {n} ∪ 交集{ Dom(p) | p ∈ predecessors(n) }
```

**复杂度**: 
- 支配集迭代：理论最坏 O(N²) 轮，实际 3-5 轮收敛（ScratchV CFG 节点数 N < 100）
- idom 提取：对每个节点遍历其严格支配者，最坏 O(N²)
- 总体在实际规模下 < 1ms

**选型理由**: 工业编译器用 Lengauer-Tarjan（近似 O(N log N)），此处选迭代不动点——代码量约 30 行 vs LT 约 150 行，CFG 节点数小，可读性优先。

### 4.3 直接支配树

从支配集筛选: `idom(n)` = 最接近 n 的严格支配者（在 `Dom(n) - {n}` 中，不被任何其他严格支配者支配的那个）。

### 4.4 自然循环检测

**回边识别**:
```
edge(a → b) 且 b ∈ Dom(a) → 回边
```

**循环体收集**:
```
对每条回边 (source, header):
    从 source 反向 BFS，遇到 header 即停止
    所有被访问到的节点 + header = body
    多条回边指向同一 header 时，合并所有回边对应的 body 取并集
```

**嵌套检测**:
```
对每对循环 (outer, inner):
    若 inner.header ∈ outer.body 且 inner.body ⊂ outer.body:
        inner.parent = outer.header
        inner.nesting_depth = outer.nesting_depth + 1
        outer.children.append(inner.header)
```

### 4.5 边构建规则

| 块终止指令 | 产生的边 |
|-----------|---------|
| RETURN | 无出边 |
| BR target | 1 条 JUMP → target（不加 FALLTHROUGH） |
| BR_IF → t1, t2 | 2 条 BRANCH: true→t1, false→t2 |
| 无终止符（纯计算块） | 1 条 FALLTHROUGH → 下一个块 |
| BR_IF 只有 1 个 target | true→target + FALLTHROUGH→下一个块 作为 false 路径 |

---

## 5. 接口契约

| 函数 | 输入 | 输出 | 不变量 |
|------|------|------|--------|
| `partition_basic_blocks_with_names` | 指令列表 | (名,指令) 列表 | 纯函数；LABEL 不进任何指令列表 |
| `build_cfg_from_instructions` | 指令列表 | CFG | 纯函数；entry 始终是第一个块的名称 |
| `cfg.successors(name)` | 块名 | 后继列表 | 不存在的块返回 [] |
| `eliminate_unreachable` | CFG | 被删块名列表 | 原地修改 CFG（非纯函数）；entry 始终保留 |

---

## 6. 技术债与已知限制

| 项 | 说明 | 优先级 |
|----|------|--------|
| idom 提取复杂度 | O(N²) 理论最坏；实际 N<100 可接受 | 低 |
| 回边检测去重 | 多条回边指向同一 header 时合并 body 取并集——算法设计中已规划，实现时需验证 | 中 |
| 空块处理 | 连续 label 间的空块保留但无意义 | 低 |
| 与旧版 `analysis/cfg_builder.py` 的关系 | 本模块是重写版，旧文件应标注 deprecated | 中 |
| CALL 边未实现 | CALL 已定义，构建逻辑待 IR 支持函数调用后实现 | 低 |
| instructions 是引用非拷贝 | 外部修改原 IR 会影响 CFG 节点；CFG 构建期间应冻结 IR | 低 |

---

## 7. 相关文件索引

### 现有文件
| 文件 | 关系 |
|------|------|
| `scratchv/ir/types.py` | 依赖 — Instruction、OpCode、BasicBlock 等 IR 类型 |
| `scratchv/analysis/cfg_builder.py` | 旧版实现（本模块重写后标注 deprecated） |
| `docs/topics/11-控制流图生成器.md` | 课题原始说明 |
| `docs/CFG_Design.md` | 本文档 |
| `docs/CFG_Dev.md` | 配套开发文档 |

### 待创建文件
| 文件 | 对应周 | 说明 |
|------|--------|------|
| `scratchv/ir/cfg.py` | W2-W9 | 主实现 |
| `tests/test_cfg.py` | W2-W12 | 单元测试 |
| `examples/cfg/if_else.dsl` | W4 | if/else 示例 |
| `examples/cfg/while_loop.dsl` | W9 | while 循环示例 |
| `examples/cfg/nested_loop.dsl` | W9 | 嵌套循环示例 |
| `examples/cfg/unreachable.dsl` | W7 | 不可达代码示例 |
| `scripts/visualize_cfg.py` | W10 | 可视化脚本 |

---

## 8. 审查清单

| 检查项 | 验证方法 |
|--------|---------|
| 基本块划分正确处理 LABEL/BR/BR_IF/RETURN | 提供 if_else.dsl，检查块数和块名 |
| LABEL 不进任何指令列表 | 检查每个块的 instructions 中无 LABEL |
| 边类型与 IR 语义一致 | 对 if_else.dsl 验证 JUMP>=2, BRANCH>=2 |
| DOT 样式与 user guide 一致 | 生成 DOT 后用 Graphviz 渲染，检查颜色 |
| 不可达消除不破坏 entry 可达性 | 构造含 return 后死代码的 DSL，验证 dead 块被移除 |
| 支配集在 10 轮内收敛 | 对所有示例 DSL 运行，断言迭代次数 < 10 |
| 自然循环检测识别回边和嵌套 | while_loop.dsl 检测到循环，nested_loop.dsl 检测到外层 depth=0 内层 depth=1 |
| 所有测试通过 | pytest tests/test_cfg.py -v |

---

## 9. 参考资料

- Aho, Lam, Sethi, Ullman — *Compilers: Principles, Techniques, and Tools* (龙书), 第 8.4 / 9.6 节
- Cooper, Torczon — *Engineering a Compiler*, 第 5 / 9 章
- Graphviz — [DOT Language Specification](https://graphviz.org/doc/info/lang.html)
- 课题 11 User Guide — `topic11_cfg_builder_guide.md`

