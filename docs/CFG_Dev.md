# ScratchV 统一控制流图（CFG）基础设施 — 开发文档

> **配套设计文档**: `CFG_Design.md` | **主实现**: `scratchv/analysis/cfg.py`
> **状态**: 已实现 | **最后更新**: 2026-09-18

---

## 1. 开发环境

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | >= 3.12 | 运行时 |
| ScratchV | 当前分支 | IR 类型、Machine IR 类型、DSL 解析器 |
| pytest | >= 8.0 | 测试 |
| Graphviz | 任意 | DOT -> PNG 渲染（可选） |

```bash
cd ScratchV
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/ -v --tb=short
```

---

## 2. Module Map

### 2.1 `scratchv/analysis/cfg.py`

| 符号 | 职责 |
|------|------|
| `EdgeType` | 边类型枚举 |
| `CFGNode` | 基本块节点；builder 产物 `instructions` 为指令列表 |
| `CFGEdge` | 控制流边 |
| `ControlFlowGraph` / `CFG` | 图查询、可达性、DOT |
| `CFGAdapter` | 指令表示抽象协议 |
| `build_cfg` | 单函数统一构建入口 |
| `build_cfg_from_instructions` | 旧 Topic 11 兼容入口 |
| `partition_basic_blocks_with_names` | 旧 Topic 11 兼容入口 |
| `compute_dominators` / `compute_dominator_tree` | 支配分析 |
| `detect_loops` / `detect_nested_loops` | 自然循环检测 |
| `to_dot` | DOT 输出便捷函数 |
| `verify_cfg` | `cfg_validation` 的薄封装 |

### 2.2 `scratchv/analysis/adapters.py`

| 符号 | 职责 |
|------|------|
| `IRCFGAdapter` | IR `Function` 接入统一 CFG，规范化 FOR/ENDFOR |
| `MachineCFGAdapter` | 单函数扁平 `MachineInstr` 接入统一 CFG |

### 2.3 分析模块

| 模块 | 关键符号 |
|------|---------|
| `scratchv/analysis/liveness.py` | `analyze_liveness`, `BlockLiveness`, `LivenessResult`, `UseDefProvider` |
| `scratchv/analysis/dataflow.py` | `run_dataflow`, `DataflowAnalysis`, `DataflowResult`, `ConstantPropagation` |
| `scratchv/analysis/usedef.py` | `IRUseDefProvider`, `MachineUseDefProvider` |
| `scratchv/analysis/cfg_validation.py` | `verify_cfg`, `CFGDiagnostic` |

---

## 3. API 契约与不变量

### 3.1 `build_cfg(adapter) -> ControlFlowGraph`

不变量：

1. 每个函数始终有唯一 entry 块。
2. 空函数生成空 entry 块。
3. 重复块名抛出 `ValueError`。
4. `instructions` 是 builder 新建列表，元素为原指令对象；不直接引用输入块列表。
5. `instruction_ids` 与 `instructions` 等长，格式为 `"<block>:<index>"`。

### 3.2 边构建规则

| 场景 | 边 |
|------|----|
| 空块 | FALLTHROUGH 到下一块 |
| 普通非 terminator 结尾 | FALLTHROUGH 到下一块 |
| 单 target 条件分支 + fallthrough | BRANCH(true) + FALLTHROUGH(false) |
| 双 target 条件分支 | BRANCH(true) + BRANCH(false) |
| 无条件跳转 | JUMP，不加 FALLTHROUGH |
| return/JALR | 无出边 |

### 3.3 `analyze_liveness(cfg, provider)`

- 反向 worklist 迭代至不动点。
- `uses[B]` 为首次定义前被读取的值；`defs[B]` 为块内全部定义。
- 先定义后使用不加入 `uses[B]`；先使用后定义同时属于 `uses[B]` 和 `defs[B]`。
- label、立即数、jump target 不进入活跃集合。
- CALL clobber 不混入 vreg defs。

### 3.4 `run_dataflow(cfg, analysis)`

- `Direction.FORWARD`：从 entry 向后继传播。
- `Direction.BACKWARD`：从无后继块向前驱传播。
- `meet` 使用排序后的邻接值，保证确定性。

### 3.5 `verify_cfg(cfg)`

返回 `list[CFGDiagnostic]`，不修改 CFG。诊断代码：

- `CFG_NO_ENTRY`
- `CFG_INVALID_ENTRY`
- `CFG_DANGLING_SOURCE`
- `CFG_DANGLING_TARGET`
- `CFG_TERMINATOR_NOT_LAST`
- `CFG_JUMP_WITH_FALLTHROUGH`
- `CFG_BAD_PREDECESSOR`
- `CFG_BAD_SUCCESSOR`

---

## 4. 常见陷阱

| 陷阱 | 表现 | 修复 |
|------|------|------|
| `CFGNode.instructions` 误当 int 使用 | 类型假设错误 | builder 产物按序列处理；仅旧测试保留 int 兼容 |
| FOR/ENDFOR 隐式循环规则分叉 | IR 与 Machine CFG 拓扑不一致 | adapter 统一规范化为 BR/BR_IF + label |
| BNEZ 忘记 fallthrough | 条件分支拓扑错误 | `has_fallthrough=True` 并生成 FALLTHROUGH(false) |
| J/JAL 后生成 fallthrough | 无条件跳转拓扑错误 | `has_fallthrough=False` |
| CALL 被当 terminator | 错误切分基本块 | CALL 保持 fallthrough |
| 空函数没有 entry | 下游依赖 entry 崩溃 | `build_cfg` 生成空 entry 块 |
| 分析缓存未失效 | 修改 CFG 后使用旧结果 | 本模块不缓存；调用方自行重建 |

---

## 5. 测试覆盖矩阵

| 测试文件 | 覆盖内容 |
|---------|---------|
| `tests/test_cfg.py` | Topic 11 兼容：partition、edge、DOT、支配、循环 |
| `tests/test_cfg_builder.py` | 旧 `CFGBuilder` 回归 |
| `tests/test_unified_cfg.py` | 统一 builder、Machine adapter、liveness、dataflow、validation |

关键断言：

- IR/Machine CFG 节点与 successor/predecessor 精确集合
- 条件分支 BRANCH + FALLTHROUGH 边类型
- liveness `live_in`/`live_out`/`live_before`/`live_after` 精确集合
- 常量传播 out 环境精确常量
- 非法 CFG 诊断码精确
- FOR 循环规范化后至少检测到 1 个自然循环

---

## 6. 修改检查清单

**改代码之前**：

- [ ] 是否修改 `CFGAdapter` 契约？若是，两个 adapter 是否同步更新？
- [ ] 是否会影响旧 `ir/cfg.py` 兼容 shim？
- [ ] 是否影响 `InstructionId` / `ValueId` 稳定性？

**改代码之后**：

- [ ] 更新 Module Map
- [ ] 更新设计/开发文档
- [ ] 补充或更新精确断言测试
- [ ] 最终运行 `pytest tests/ -v --tb=short`
- [ ] 用 `visualize_cfg.py` 对 if/else、while 示例目视检查 DOT
