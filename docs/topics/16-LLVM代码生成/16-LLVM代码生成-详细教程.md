# 课题16：LLVM 代码生成后端

> **难度**：中 | **类型**：项目实战 | **源文件**：`scratchv/backend/llvm_codegen.py`, `scratchv/standalone/onnx_to_llvm_standalone.py`
> **状态**：✅ 已完成

---

## 概述

LLVM 代码生成器将 ScratchV IR / ONNX 模型翻译为 LLVM IR 文本（`.ll` 文件）。这是 ScratchV 的 float32 路径，使用硬件浮点指令（`fmul`, `fadd`）+ GEP 地址计算 + phi 节点循环结构，每条 MAC 约 2 条指令（vs Q16.16 的 ~12 条）。生成的 LLVM IR 可接入 LLVM 全套工具链（`opt`, `llc`, `lli`）进行优化和多目标编译。

---

## 理解背景

### 是什么？

LLVM 代码生成器将 ScratchV IR / ONNX 模型翻译为 **LLVM IR 文本**（`.ll` 文件）。这是 ScratchV 的 **float32 路径**，与 Q16.16 定点路径互补。

```
ONNX 模型
    │
    ├─→ 原生路径: RV32IM 定点机器码 (.bin)
    │
    └─→ LLVM 路径: LLVM IR (.ll) → llc → RISC-V 64 浮点汇编 (.s)
                    这条就是你正在看的
```

### 为什么？

1. **基准对比**: LLVM 路径使用 float32 + 硬件浮点指令（`fmul`, `fadd`），每条 MAC ~2 条指令；原生路径使用 Q16.16，每条 MAC ~12 条。对比两边的差距，找到优化方向
2. **LLVM 全套优化**: 生成的 `.ll` 可以接 `opt -O3` 享受 LLVM 的循环展开、向量化、指令合并等
3. **多目标**: LLVM IR 可以编译到 RV64、x86-64、ARM 等多个平台

### 核心概念

#### 两条 LLVM 路径

| | 路径 A (库路径) | 路径 B (Standalone) |
|------|---------------|-------------------|
| 文件 | `backend/llvm_codegen.py` | `standalone/onnx_to_llvm_standalone.py` |
| 输入 | IR Program | ONNXModel (手工解析) |
| 状态 | 框架完整，NN 算子占位 | 完整的 float32 循环生成 |
| 依赖 | 可选 llvmlite | Python stdlib only |

#### LLVM IR 关键特性

| 特性 | LLVM IR | ScratchV 原生 |
|------|---------|-------------|
| 地址计算 | `getelementptr` 单指令 | 手动 `MUL + ADD` 链 (~5条) |
| 浮点运算 | `fmul`, `fadd` 原生 | `MUL + SRAI 16` (Q16.16) |
| 循环结构 | `phi` 节点 + `icmp` + `br` | 手动 label + jump |
| 类型系统 | LLVM 类型系统 | 原始 32-bit 整数 |
| 优化 | `opt -O3` 全套 pass | 无后优化 |

---

## 详细任务

1. 学习 LLVM IR 基础语法：类型系统、指令格式、phi 节点、基本块结构。
2. 实现 LLVMIRBuilder 辅助类：new_register(), emit_fmul(), emit_fadd(), emit_gep(), emit_header()。
3. 实现简单算术算子的 LLVM IR 生成：ADD, SUB, MUL, DIV 映射为 fadd/fsub/fmul/fdiv。
4. 实现激活函数的 LLVM IR 生成：ReLU（fcmp ogt + select）、Sigmoid（call @expf）。
5. 实现 Conv2D 的 LLVM IR 生成：6 层 phi 嵌套循环 + GEP 地址计算 + fmul/fadd MAC。
6. 实现 Gemm 的 LLVM IR 生成：3 层循环 + transB 支持。
7. 实现 MaxPool 的 LLVM IR 生成：5 层循环 + 边界检查 + fcmp ogt。
8. 实现 Reshape 和其他算子的生成。
9. 实现 --compare 模式：估算 LLVM 和 ScratchV 的动态操作数差距。
10. 用 LLVM 工具链（opt -O3, llc, lli）验证生成的 IR 的正确性和优化效果。

---

## 交付产物

- `scratchv/backend/llvm_codegen.py` — 库路径 LLVM IR 生成器
- `scratchv/standalone/onnx_to_llvm_standalone.py` — Standalone 路径 LLVM 编译器
- 生成的 `.ll` 示例文件（CNN 模型全量）
- LLVM vs ScratchV 对比报告（--compare 模式）

---

## 代码走读

### LLVMIRBuilder 辅助类

```python
class LLVMIRBuilder:
    def __init__(self):
        self._reg_counter = 0
        self._output = []

    def new_register(self) -> str:
        self._reg_counter += 1
        return f"%{self._reg_counter}"

    def emit_fmul(self, a: str, b: str) -> str:
        r = self.new_register()
        self._output.append(f"  {r} = fmul float {a}, {b}")
        return r

    def emit_gep(self, base: str, idx: str) -> str:
        r = self.new_register()
        self._output.append(
            f"  {r} = getelementptr float, float* {base}, i32 {idx}"
        )
        return r
```

### Conv2D 的 LLVM IR 生成

与原生路径的差异：
- 用 `phi` 节点实现循环（无需手动 `br` + label）
- 用 `getelementptr` 做地址计算（替代 MUL+ADD 链）
- 用 `fmul`/`fadd` 做浮点乘加（替代 Q16.16 的 MUL+SRAI）

---

## 动手练习

### 练习 1: 阅读 LLVM IR

生成 CNN 模型的 LLVM IR，找到 Conv2D 的最内层循环代码，和 Q16.16 版本对比指令数。

### 练习 2: 体验 opt -O3 的效果

对一个 LLVM IR 文件分别用 `-O0`、`-O2`、`-O3` 优化，用 `diff` 对比优化后的 IR 有什么不同。

### 练习 3: 多目标编译

用 `llc -march=x86-64` 把 LLVM IR 编译为 x86 汇编，看看和 RISC-V 版本有什么不同。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **llvmlite 安装** | 路径 A 需要 `llvmlite`，在某些平台可能安装失败。路径 B 不需要 |
| **Target triple** | 当前是 `riscv64-unknown-elf`，如果要编译到其他平台需要修改 |
| **float32 vs Q16.16 精度差异** | LLVM 路径用原生 float32，和 Q16.16 路径结果有微小差异（~1.5e-5 相对误差） |
| **phi 节点的复杂性** | 写 phi 节点比写 label+jump 更难，循环嵌套深时容易写错 |

---

## 进阶阅读

- [LLVM Language Reference](https://llvm.org/docs/LangRef.html) — LLVM IR 完整语法
- [ARCHITECTURE.md](../../reference/ARCHITECTURE.md) — 双路径对比详解
- [03-指标解读指南](../../guide/03-指标解读指南.md) — 如何解读 LLVM vs ScratchV 的性能差距
- 相关 topic: [课题8 — 指令选择](../08-指令选择/08-指令选择.md) | [课题19 — Standalone RISC-V 编译器](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md) | [课题22 — Standalone LLVM 编译器](../22-Standalone-LLVM编译器/22-Standalone-LLVM编译器.md)

---

## 12周每周目标

- **W1**：学习 LLVM IR 基础语法：类型系统（i32, float, pointer）、指令格式、基本块和 phi 节点。阅读 [LLVM Language Reference](https://llvm.org/docs/LangRef.html) 前 3 章。
- **W2**：阅读 `backend/llvm_codegen.py` 和 `standalone/onnx_to_llvm_standalone.py` 全文。理解两条路径的架构差异。
- **W3**：实现 LLVMIRBuilder 辅助类：new_register(), emit_fmul(), emit_fadd(), emit_load(), emit_store(), emit_gep(), emit_header(), emit_br(), emit_icmp()。
- **W4**：实现简单算术算子的 LLVM IR 生成：ADD→fadd, SUB→fsub, MUL→fmul, DIV→fdiv。编写测试：手工 IR → LLVM IR → lli 执行验证。
- **W5**：实现激活函数的 LLVM IR 生成：ReLU（fcmp ogt + select 选择 max(x, 0)）、Sigmoid（声明 @expf、call 调用）。
- **W6**：实现 Conv2D 的 LLVM IR 生成框架：6 层 phi 嵌套循环 + 循环变量 icmp + br 跳转。先实现不完整版本（只有循环骨架）。
- **W7**：完善 Conv2D：添加 GEP 地址计算（input_ptr, weight_ptr, output_ptr 的偏移计算）、fmul/fadd MAC 操作、bias add。
- **W8**：实现 Gemm 的 LLVM IR 生成：3 层 phi 循环 + transB 支持（if transB: 交换 weight 行列索引）。
- **W9**：实现 MaxPool 的 LLVM IR 生成：5 层循环 + 边界检查 + fcmp ogt 比较 + 条件 select。
- **W10**：实现 Reshape 和其他算子的 LLVM IR 生成。实现 --compare 模式：同时估算 LLVM 和 ScratchV 的动态操作数。
- **W11**：用 LLVM 工具链（opt -O3 → llc -O3 → lli）完整验证生成的 IR。对比优化前后的 IR 差异。
- **W12**：撰写文档（LLVM IR 示例、与原生路径的对比表、phi 节点使用指南），准备演示。
