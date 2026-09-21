# Topic: IR 优化器框架 (Optimizer Framework)

> **归档说明：** 本文保留为课题历史摘要；当前公共接口以
> [`docs/topics/04-IR优化器框架.md`](../04-IR优化器框架.md) 和实际代码为准。

> **源文件**: `scratchv/optimizer/`
> **层级**: IR → IR (源和目标都是 IR Program)
> **策略**: 所有 pass 原地修改 Program, 返回变更计数

---

## 概述

优化器在 IR 层面执行平台无关的优化。所有 pass 共享相同接口: 输入 `Program`, 原地修改, 返回 `int` (变更次数)。

## 优化级别

| 级别 | Pass 组合 | 适用场景 |
|------|----------|---------|
| `"none"` | 无 | 调试, 验证未优化 IR 的正确性 |
| `"basic"` | ConstantFolder + DeadCodeEliminator | 日常使用, 安全无害 |
| `"all"` | basic + Peephole + MulAddFusion + LICM | 性能优先 |

## 五个优化 Pass

### 1. ConstantFolder (`constant_folding.py`)

**功能**: 编译时计算常量表达式

```
优化前:  %1 = load_const 2.0       优化后:  %3 = load_const 6.0
         %2 = load_const 3.0
         %3 = add %1, %2
         %4 = mul %3, %1            %4 = mul %3, %1  (仍可继续折叠)
```

- 支持 ADD, SUB, MUL, DIV
- 要求所有操作数为编译时常量 (`Value.constant == True`)
- 将结果替换为 `LOAD_CONST` 指令

### 2. DeadCodeEliminator (`dead_code.py`)

**功能**: 删除无用的指令 (结果不被任何其他指令使用)

```
优化前:  %1 = add %a, %b           优化后:  (整条指令删除,
         %2 = mul %c, %d                   因为 %1 从未被使用)
         store %2, %ptr            store %2, %ptr
```

- 保留有副作用的指令: STORE, RETURN, BR, BR_IF, FOR, ENDFOR, ALLOCA
- 迭代执行直到不动点

### 3. IRPeepholeOptimizer (`peephole.py`)

**功能**: 局部指令模式替换

| 模式 | 优化 |
|------|------|
| `add rd, rs, 0` | 删除 (恒等) |
| `mul rd, rs, 1` | 替换为 `mv rd, rs` |
| `mul rd, rs, 0` | 替换为 `li rd, 0` |
| `j L` 后紧跟 `L:` | 删除冗余跳转 |

### 4. MulAddFusion (`muladd_fusion.py`)

**功能**: 合并连续的 mul→add 为单个注释 add (为后端 FMA 融合提供标记)

```
优化前:  %1 = mul %a, %b           优化后:  %2 = add %1, %c   ; fused_mul_add=True
         %2 = add %1, %c
```

- 后端看到 `fused_mul_add=True` 属性可发射 `fmadd` 指令 (如果 ISA 支持)

### 5. LICM (`licm.py`)

**功能**: 循环不变量外提 (Loop Invariant Code Motion)

```
优化前:                           优化后:
  for i in range(0, 100):          %1 = load_const 3.14
    %1 = load_const 3.14            for i in range(0, 100):
    %2 = mul %x, %1                  %2 = mul %x, %1
  endfor                           endfor
```

- 检测在 FOR/ENDFOR 循环内但操作数不依赖于循环变量的指令
- 将其移到循环之前

## Pass 接口规范

```python
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.ir.types import Program

program: Program = ...
folder = ConstantFolder()
changes = folder.optimize(program)  # 返回变更计数
if changes > 0:
    print(f"Folded {changes} constants")
```

所有 pass 遵循相同接口。当前实现的默认三级管线通过唯一工厂函数创建：

```python
from scratchv.compiler import create_optimization_pass_manager

manager = create_optimization_pass_manager("all")
report = manager.run(program)
print(report.total_changes)
```

编译器入口和 benchmark 共用该工厂，当前映射为：

- `none`：空管线
- `basic`：`constant-folding` → `dead-code-elim`
- `all`：`constant-folding` → `dead-code-elim` → `ir-peephole` →
  `muladd-fusion` → `licm`

`manager.run()` 返回不可变的
`OptimizationReport`；其 `executions` 按执行顺序记录 pass 名称、变更数和耗时，
`total_changes` 与 `elapsed_seconds` 提供总计。

主 CLI 使用 `--opt-level none|basic|all`。`--optimize` 只是参数名的兼容别名，
仍须显式提供 `none`、`basic` 或 `all`；裸 `--optimize` 不合法。

```bash
scratchv model.onnx --opt-level all
scratchv model.onnx --optimize all
```

## 相关 Topic

- Topic 12: 后端指令计数统计器 → 评估优化效果
- Topic 13: 后端窥孔优化器 → 汇编层 peephole
- Topic 21: IR 验证器 → 优化后验证 IR 正确性
- [ARCHITECTURE.md](../ARCHITECTURE.md) — 完整管线
