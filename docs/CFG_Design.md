# ScratchV 控制流图（CFG）模块 — 设计文档

> **版本**: 2.0 | **读者**: 模块负责人、代码审查者、下游优化 pass 开发者
> **源文件**: `scratchv/ir/cfg.py`（待实现） | **课题**: 11 — 控制流图生成器
> **状态**: 设计中 | **最后更新**: 2026-07-31

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
entry ──BRANCH(true)──▶ L_then ──JUMP──▶ merge ──(return, 出口)
  │                                      ▲
  └──BRANCH(false)──▶ L_else ──JUMP──────┘
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
    instructions: list[Instruction] # 引用而非拷贝
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

| 节点 | 颜色 | 边类型 | DOT 样式 |
|------|------|--------|---------|
| 入口块 | 绿色 | FALLTHROUGH | 实线 |
| 出口块 (return) | 红色 | BRANCH | 蓝色虚线 + 条件标签 |
| 循环头 | 蓝色 | JUMP | 红色实线 |
| 普通块 | 浅黄色 | CALL（预留） | 紫色点线 |

### 3.5 NaturalLoop

```python
@dataclass
class NaturalLoop:
    header: str
    body: set[str]
    back_edges: list[tuple[str, str]]
    parent: str | None
    children: list[str]
    nesting_depth: int
```

---

## 4. 算法设计

### 4.1 基本块划分

**输入**: `list[Instruction]`  **输出**: `list[tuple[name, list[Instruction]]]`

**算法**: 单趟线性扫描 O(n)

```
遍历指令:
  遇到 LABEL → 结束当前块，标签名 = 新块名
  加入指令到当前块
  遇到 BR/BR_IF/RETURN → 结束当前块
```

### 4.2 支配集计算

**算法**: 迭代不动点 (Iterative Fixed-Point)

```
Dom(entry) = {entry}
Dom(n) = 全集 (n != entry)

重复直到不变:
  Dom(n) = {n} U 交集{Dom(p) | p in preds(n)}
```

**复杂度**: 理论最坏 O(N^2) 轮，实际 3-5 轮收敛（ScratchV CFG 节点 < 100）。

**选型理由**: 工业编译器用 Lengauer-Tarjan（O(N log N)），此处选迭代不动点——代码量 30 行 vs LT 150 行，CFG 规模小。

### 4.3 直接支配树

从支配集筛选: `idom(n)` = 最接近 n 的严格支配者。

### 4.4 自然循环检测

```
回边: edge(a->b) 且 b in Dom(a)
循环体: 从 source 反向 BFS，遇 header 即止
嵌套: inner.body 是 outer.body 的真子集
```

---

## 5. 接口契约

| 函数 | 输入 | 输出 | 不变量 |
|------|------|------|--------|
| `partition_basic_blocks_with_names` | 指令列表 | (名,指令) 列表 | 纯函数；LABEL 不进指令列表 |
| `build_cfg_from_instructions` | 指令列表 | CFG | 纯函数；entry 始终是第一个块 |
| `cfg.successors(name)` | 块名 | 后继列表 | 不存在的块返回 [] |
| `eliminate_unreachable` | CFG | 被删块名列表 | 原地修改；保留 entry |

---

## 6. 已知限制

| 项 | 说明 | 优先级 |
|----|------|--------|
| 支配树复杂度 | idom 提取 O(N^3) 理论最坏；实际 N<100 可接受 | 低 |
| 回边检测去重 | 多条回边指向同一 header 时，仅第一个被完整分析 | 中 |
| 空块处理 | 连续两个 label 之间的空块保留但无意义 | 低 |
| CALL 边未实现 | CALL 类型已定义但构建逻辑未实现（等 IR 支持函数调用） | 低 |

---

## 7. 相关文件索引

### 现有文件
| 文件 | 关系 |
|------|------|
| `scratchv/ir/types.py` | 依赖 — IR 类型定义（Instruction, OpCode, BasicBlock 等） |
| `scratchv/analysis/cfg_builder.py` | 旧版 CFG 实现
| `docs/topics/11-控制流图生成器.md` | 课题原始说明文档 |
| `docs/CFG_Design.md` | 本文档 |
| `docs/CFG_Dev.md` | 配套开发文档 |

### 待创建文件（按开发路线依次产出）
| 文件 | 对应周 | 说明 |
|------|--------|------|
| `scratchv/ir/cfg.py` | W2-W9 | 主实现文件 |
| `tests/test_cfg.py` | W2-W12 | 单元测试（随功能逐步扩充） |
| `examples/cfg/if_else.dsl` | W4 | if/else 示例 |
| `examples/cfg/while_loop.dsl` | W9 | while 循环示例 |
| `examples/cfg/nested_loop.dsl` | W9 | 嵌套循环示例 |
| `examples/cfg/unreachable.dsl` | W7 | 不可达代码示例 |
| `scripts/visualize_cfg.py` | W10 | 可视化脚本 |

---

## 8. 审查清单

- [ ] 基本块划分正确处理 LABEL / BR / BR_IF / RETURN 边界
- [ ] 边类型与 IR 语义一致（FALLTHROUGH / BRANCH / JUMP / CALL）
- [ ] DOT 样式与课题 user guide 规范一致
- [ ] 不可达消除不破坏 entry 可达性
- [ ] 支配集计算在 10 轮内收敛
- [ ] 自然循环检测识别回边和嵌套
- [ ] DOT 输出可被 Graphviz 正确渲染
- [ ] 所有测试通过

---

## 9. 参考资料

- Aho, Lam, Sethi, Ullman — *Compilers: Principles, Techniques, and Tools* (龙书), 第 8.4 / 9.6 节
- Cooper, Torczon — *Engineering a Compiler*, 第 5 / 9 章
- Graphviz — [DOT Language Specification](https://graphviz.org/doc/info/lang.html)
- 课题 11 User Guide — `topic11_cfg_builder_guide.md`

