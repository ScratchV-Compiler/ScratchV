# ScratchV 控制流图（CFG）模块 — 开发文档

> **配套设计文档**: `CFG_Design.md` | **目标文件**: `scratchv/ir/cfg.py`
> **状态**: 待开发 | **最后更新**: 2026-08-02

---

## 1. 开发环境

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | >= 3.12 | 运行时 |
| ScratchV | main 分支 | IR 类型 (`scratchv/ir/types.py`)、DSL 解析器 |
| pytest | >= 8.0 | 测试 |
| Graphviz | 任意 | DOT -> PNG 渲染（可选） |

```bash
cd ScratchV
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/ -v --tb=short
```

---

## 2. Module Map（符号 -> 职责）

| 符号 | 职责 |
|------|------|
| `EdgeType` | 边类型枚举（FALLTHROUGH / BRANCH / JUMP / CALL） |
| `CFGNode` | 基本块节点数据类；`instructions` 是对原 IR 列表的引用（非拷贝） |
| `CFGEdge` | 控制流边数据类 |
| `CFG` | 完整控制流图 + successors / predecessors / to_dot |
| `NaturalLoop` | 自然循环结构数据类 |
| `partition_basic_blocks_with_names` | 指令列表 -> 带名称的基本块列表（纯函数） |
| `build_cfg_from_instructions` | 指令列表 -> 完整 CFG（纯函数；串联块划分 + 节点 + 边） |
| `eliminate_unreachable` | 从 entry DFS，删除不可达块及关联边（**原地修改**，非纯函数） |
| `compute_dominators` | 迭代不动点计算支配者集合（纯函数） |
| `compute_dominator_tree` | 提取每个节点的直接支配者（纯函数） |
| `detect_loops` | 基于回边检测自然循环；同 header 多条回边合并 body 取并集（纯函数） |
| `detect_nested_loops` | 计算循环嵌套关系与深度（纯函数） |

> **Instruction 类型定义**: 参见 `scratchv/ir/types.py`。核心字段：`opcode`（OpCode 枚举）、`dest`（目标 Value）、`operands`（操作数列表）、`target`（跳转目标字符串）。

---

## 3. 12 周开发路线

| 周 | 目标 | 关键产出 |
|----|------|---------|
| W1 | 理论：龙书 8.4/9.6 + 手绘 3 张 CFG | 手绘草图 |
| W2 | `partition_basic_blocks_with_names` | 块划分函数 |
| W3 | `EdgeType`, `CFGNode`, `CFGEdge`, `CFG` | 数据结构 |
| W4 | `build_cfg_from_instructions` | 完整 CFG 构建 |
| W5 | 学习 Graphviz DOT 语法 | 手写 DOT |
| W6 | `CFG.to_dot()` — 样式与 user guide 一致 | DOT 输出 |
| W7 | `eliminate_unreachable` | 不可达消除 |
| W8 | `compute_dominators` + `compute_dominator_tree` | 支配树 |
| W9 | `detect_loops` + `detect_nested_loops` | 循环检测 |
| W10 | `visualize_cfg.py` 独立脚本 | 可视化工具 |
| W11 | 可选：CLI 集成 --cfg | 管线集成 |
| W12 | 测试完善 + 文档 | 35+ 用例 + 交付 |

---

## 4. API 契约与不变量

### 4.1 纯函数（只读分析）

以下函数不修改输入，多次调用结果一致：

- `partition_basic_blocks_with_names`
- `build_cfg_from_instructions`
- `compute_dominators`
- `compute_dominator_tree`
- `detect_loops`
- `detect_nested_loops`

### 4.2 原地修改函数

- `eliminate_unreachable` — 修改传入的 `CFG` 对象，删除不可达节点和边。调用后应丢弃旧 CFG 引用。

### 4.3 build_cfg_from_instructions

```python
def build_cfg_from_instructions(
    instructions: list[Instruction],
    function_name: str = "main",
    entry_name: str = "entry",
) -> CFG:
```

**不变量**:
1. 纯函数
2. `cfg.entry` 始终是第一个块的名称
3. 空指令列表返回含空 entry 块的 CFG（`CFGNode(name=entry_name, instructions=[])`），避免调用方依赖 entry 时出错
4. 每个块的 `instructions` 是对原列表子序列的**引用**（非拷贝）。CFG 构建期间不应修改原 IR

### 4.4 自动命名策略

当 BR/BR_IF/RETURN 后紧跟的指令无 LABEL 时，新块自动命名为 `b0`, `b1`, `b2`...（自增计数器，从 0 开始）。

### 4.5 边构建规则

| 块类型 | 边 | 说明 |
|--------|-----|------|
| 以 `BR target` 结尾 | 1 条 JUMP -> target | **不加** FALLTHROUGH |
| 以 `BR_IF -> t1,t2` 结尾 | 2 条 BRANCH: true->t1, false->t2 | |
| 以 `RETURN` 结尾 | 无出边 | |
| 其他（无终止指令） | 1 条 FALLTHROUGH -> 下一个块 | 仅当不是最后一个块时 |

### 4.6 DOT 样式（与 user guide 一致）

| 节点 | 颜色 | 边类型 | 样式 |
|------|------|--------|------|
| 入口 | 绿色 | FALLTHROUGH | 黑色实线 |
| 出口 | 红色 | BRANCH | 蓝色虚线 + 标签 |
| 循环头 | 蓝色 | JUMP | 红色实线 |
| 普通 | 浅黄色 | CALL（预留） | 紫色点线 |

---

## 5. 常见陷阱

| 陷阱 | 表现 | 修复 |
|------|------|------|
| LABEL 被当成普通指令加入块 | 块内出现 `.L1:` | `continue` 跳过 LABEL |
| 最后一个块忘记追加 | CFG 缺最后指令 | 循环后加收尾逻辑 |
| BR 后无 LABEL 导致块名冲突 | 多个块同名 | 自动命名 `b0`, `b1`... |
| 支配集不收敛 | 迭代超过 100 轮 | 检查 entry 是否在 nodes 中 |
| BR 块误加 FALLTHROUGH | 无条件跳转后还有额外边 | BR 块只加 JUMP，不加 FALLTHROUGH |
| DOT 引号未转义 | Graphviz 渲染报错 | 标签中的 `"` 替换为 `\"` |
| visited 用 list 而非 set | 无限循环 | 必须用 set |
| CFG 构建后修改原 IR 列表 | CFG 节点内容异常变化 | `instructions` 是引用非拷贝，构建后应冻结原 IR |

---

## 6. 测试覆盖矩阵

| 测试类 | 关键用例 | 断言 |
|--------|---------|------|
| TestBasicBlockPartitioning | 纯计算 / if-else / for | 块数正确 |
| TestEdgeTypes | if-else / while / 线形 | JUMP>=2; BRANCH>=2; 线形无 JUMP |
| TestDOTOutput | 基础 / 高亮循环 | 含 `digraph`、`CFG_main` |
| TestUnreachableElimination | 无线形不可达 / 含孤立节点 | removed 正确 |
| TestDominators | entry 自支配/支配全部/idom=None | 支配集正确 |
| TestLoopDetection | 无线形 / while / 嵌套for | 循环数正确；嵌套 depth 正确 |
| TestEdgeCases | 空函数 / 多函数 / 空指令列表 | 不崩溃；entry 节点存在 |

---

## 7. 修改检查清单

**改代码之前**:
- [ ] 新增函数在 Module Map 中有条目？
- [ ] 纯函数确认不修改输入；原地修改函数在 Module Map 中已标注？
- [ ] 覆盖了边界情况（空输入、单节点、无出边）？
- [ ] DOT 样式符合 user guide？

**改代码之后**:
- [ ] `pytest tests/test_cfg.py -v` 全通过
- [ ] `visualize_cfg.py` 对 if_else/while_loop 生成 DOT 正确
- [ ] Graphviz 渲染图中节点颜色和边样式正确

---

## 8. 端到端示例：新增一个分析函数

以添加 `count_back_edges(cfg) -> int` 为例：

1. `cfg.py` 中添加函数 -> Module Map 加一行，标注纯函数
2. 写测试 -> 测试覆盖矩阵加一条
3. 对 while_loop.dsl 手工验证（回边数应为 1）
4. `pytest tests/test_cfg.py -v`
5. 更新本文档

