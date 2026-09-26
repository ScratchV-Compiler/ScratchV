# 课题22：Standalone LLVM 编译器

> **难度**：高 | **类型**：项目实战 | **源文件**：`scratchv/standalone/onnx_to_llvm_standalone.py` | **行数**：~2100
> **状态**：✅ 已完成

---

## 概述

Standalone LLVM 编译器是 ScratchV float32 路径的完整实现。它复用 Standalone RISC-V 编译器的 ONNX 解析（零依赖 protobuf 解析），但生成 float32 LLVM IR 而非 Q16.16 机器码。通过 `--compare` 模式同时估算两条路径的动态指令数差距，是 ScratchV 的性能 baseline 和优化对标工具。

---

## 理解背景

### 是什么？

Standalone LLVM 编译器是 ScratchV **float32 路径**的完整实现。它复用 Standalone RISC-V 编译器的 ONNX 解析（零依赖 protobuf 解析），但生成 float32 LLVM IR 而非 Q16.16 机器码。

```
ONNX 模型 → ONNXModel (手工解析) → LLVMCNNGenerator → LLVM IR (.ll)
                                         ↑
                                    LLVMIRBuilder (辅助)
```

### 为什么？

- **Baseline**: LLVM float32 路径是 ScratchV 优化的"天花板"——它使用硬件浮点指令，每条 MAC ~2 条指令
- **对比分析**: `--compare` 模式同时估算两条路径的动态指令数，量化差距
- **工具链集成**: 生成的 `.ll` 可接 `opt -O3` 和 `llc` 多目标编译

### 核心概念

#### 四组件

| 组件 | 职责 |
|------|------|
| **ONNXModel** | 手工 protobuf 解析（复用自 RISC-V standalone） |
| **LLVMIRBuilder** | LLVM IR 文本生成辅助类 |
| **LLVMCNNGenerator** | 逐算子生成 float32 LLVM IR 循环 |
| **LLVMJITRunner** | （可选）libLLVM JIT 直接执行 |

#### 支持的算子和 IR 特征

| 算子 | LLVM IR 特征 |
|------|-------------|
| Conv2D | 6层循环 + GEP + `fmul`/`fadd` MAC + padding 边界 |
| Gemm | 3层循环 + transB + GEP |
| MaxPool | 5层循环 + 边界检查 + `fcmp ogt` |
| ReLU | `fcmp ogt` + `select` |
| Sigmoid | 调用外部 `@expf` 函数 |
| Reshape | 直接 memcpy |

#### 与原生路径的关键差异

| 特性 | LLVM Standalone | ScratchV Standalone |
|------|----------------|---------------------|
| 数值 | float32 (IEEE 754) | Q16.16 定点 |
| 输出 | `.ll` 文本 | `.bin` 机器码 |
| 地址计算 | `getelementptr` | 手动 MUL+ADD |
| 每条 MAC | ~2 条（fmul+fadd） | ~12 条（MUL+SRAI+ADD+...） |
| 执行 | 需 llc/lli | 可直接加载执行 |

---

## 详细任务

1. 学习 Standalone 路径的架构和与库路径的差异（零依赖 vs 使用外部包）。
2. 阅读 `onnx_to_llvm_standalone.py` 全文（~2100 行），理解四组件的职责和交互。
3. 实现 ONNXModel 手工 protobuf 解析（复用 RISC-V Standalone 的 ProtoReader）。
4. 实现 LLVMIRBuilder 辅助类：new_register(), emit_fmul/fadd/gep/load/store/br/icmp/phi。
5. 实现 LLVMCNNGenerator 框架：按算子类型分发的 gen_* 方法。
6. 实现 Conv2D 的 float32 LLVM IR 生成：6 层 phi 嵌套循环 + GEP 地址计算 + fmul/fadd MAC + padding 边界处理。
7. 实现 Gemm/MaxPool/激活函数（ReLU, Sigmoid）的 LLVM IR 生成。
8. 实现 `--compare` 模式：同时估算 LLVM 和 ScratchV 的动态操作数，生成差距报告。
9. 实现 `--report` 模式：支持 HTML/JSON/Markdown 三种输出格式。
10. 集成到 CI 流程（make bench-ci），支持自动化对比。

---

## 交付产物

- `scratchv/standalone/onnx_to_llvm_standalone.py` — 完整编译器
- 生成的 `.ll` 示例文件
- `--compare` 对比报告（JSON/HTML/Markdown）
- CI 集成（make bench-ci）

---

## 代码走读

### LLVMIRBuilder

```python
class LLVMIRBuilder:
    def new_register(self) -> str:
        self._reg_counter += 1
        return f"%{self._reg_counter}"

    def emit_fmul(self, a, b):
        r = self.new_register()
        self._output.append(f"  {r} = fmul float {a}, {b}")
        return r

    def emit_gep(self, base, idx):
        r = self.new_register()
        self._output.append(
            f"  {r} = getelementptr float, float* {base}, i32 {idx}"
        )
        return r
```

---

## 动手练习

### 练习 1: 阅读生成的 LLVM IR

生成 CNN 模型的 `.ll` 文件，找到 Conv2D 最内层循环，和 [课题16](../16-LLVM代码生成/16-LLVM代码生成.md) 的示例对比。

### 练习 2: 体验 `--compare`

运行 `--compare` 模式，理解每个算子的 LLVM vs ScratchV 指令数差异。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **llc 不可用** | 没有安装 LLVM 工具链时只能生成 `.ll`，不能编译。用 `lli` 或在线 Compiler Explorer |
| **target triple** | 固定为 `riscv64-unknown-elf`，编译到其他平台需修改 |

---

## 进阶阅读

- [ARCHITECTURE.md](../../reference/ARCHITECTURE.md) — 双路径对比详解
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
- 相关 topic: [课题16 — LLVM 代码生成](../16-LLVM代码生成/16-LLVM代码生成.md) | [课题19 — Standalone RISC-V](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md)

---

## 12周每周目标

- **W1**：学习 Standalone 路径的架构：与库路径的核心差异（零依赖 vs 使用 onnx/llvmlite）。阅读 ARCHITECTURE.md 中双路径对比部分。
- **W2**：阅读 `onnx_to_llvm_standalone.py` 全文（~2100 行）。理解 ONNXModel → LLVMIRBuilder → LLVMCNNGenerator 的数据流。画出四组件交互图。
- **W3**：实现 ONNXModel 手工 protobuf 解析（复用 RISC-V Standalone 的 ProtoReader）。提取 graph node 列表、initializer 权重、input/output 定义。
- **W4**：实现 LLVMIRBuilder 完整功能：new_register, emit_fmul/fadd/fsub/fdiv, emit_gep, emit_load/store, emit_br/br_cond, emit_icmp, emit_phi, emit_header。
- **W5**：实现 LLVMCNNGenerator 框架：按算子类型分发的 gen_* 方法。实现简单算子的生成（ADD/SUB/MUL → fadd/fsub/fmul）。
- **W6**：实现 Conv2D 的 float32 LLVM IR 生成：6 层 phi 嵌套循环 + GEP 地址计算 + fmul/fadd MAC。处理 padding 边界条件（if 判断 + 条件跳转）。
- **W7**：实现 Gemm 的 LLVM IR 生成：3 层 phi 循环 + transB 支持。实现 MaxPool：5 层循环 + 边界检查 + fcmp ogt 比较。
- **W8**：实现激活函数：ReLU（fcmp ogt + select）、Sigmoid（声明 @expf、call 调用）。实现 Reshape（直接 memcpy）。
- **W9**：实现 `--compare` 模式：同时估算 LLVM 和 ScratchV 的动态操作数。按算子拆分对比（Conv/Gemm/MaxPool/激活函数各自的差距）。
- **W10**：实现 `--report` 模式：支持 HTML（纯静态、内嵌 CSS）、JSON（结构化数据）、Markdown（可读报告）三种输出格式。
- **W11**：集成到 CI 流程：更新 Makefile（make bench-ci 调用 onnx_to_llvm_standalone.py --compare）。确保 CI yaml 中正确配置 LLVM 工具链路径。
- **W12**：撰写文档（双路径对比详解、LLVM IR 示例、--compare 输出解读），准备演示。
