# Topic: Standalone LLVM IR 编译器

> **源文件**: `scratchv/standalone/onnx_to_llvm_standalone.py` (2108 行)
> **依赖**: `onnx` (读取模型), `numpy` (权重处理)
> **输入**: ONNX `.onnx` 文件
> **输出**: LLVM IR 文本 `.ll` + 性能对比估算

---

## 概述

独立 LLVM IR 代码生成器，复用 Standalone RISC-V 编译器的 ONNX 解析，但生成 float32 LLVM IR 而非 Q16.16 机器码。

## 组件

```
onnx_to_llvm_standalone.py
├── ONNXModel (复用自 standalone RISC-V 编译器)
├── LLVMIRBuilder      ← LLVM IR 文本生成辅助
├── LLVMCNNGenerator   ← 逐算子 LLVM IR 生成
└── LLVMJITRunner      ← (可选) libLLVM JIT 执行
```

## LLVMCNNGenerator 支持的算子

| 算子 | 生成的 IR 特征 |
|------|---------------|
| Conv2D | 6 层循环 + GEP 地址计算 + fmul/fadd MAC + padding 边界检查 |
| Gemm | 3 层循环 + transB 支持 + GEP |
| MaxPool | 5 层循环 + 边界检查 + fcmp ogt 比较 |
| ReLU | `fcmp ogt` + `select` (LLVM 的 max 语义) |
| Sigmoid | 调用外部 `@expf` 函数 |
| Reshape | 直接 memcpy (无计算) |

## 关键差异 (vs ScratchV 原生)

| 特性 | LLVM Standalone | ScratchV Standalone |
|------|----------------|---------------------|
| 数值 | float32 (IEEE 754) | Q16.16 定点 |
| 输出 | `.ll` 文本 | `.bin` 机器码 |
| 地址计算 | `getelementptr` | 手动 MUL+ADD |
| MAC | `fmul + fadd` (~2 指令) | `MUL + SRAI 16 + ADD` (~15 指令) |
| 目标 | 需 llc 编译后执行 | 直接加载执行 |

## 对比功能 (--compare)

```bash
python scratchv/standalone/onnx_to_llvm_standalone.py models/graph/cnn.onnx \
    -o output.ll --compare
```

输出 LLVM IR 和 ScratchV Q16.16 的动态操作数估算对比:
- LLVM: ~5.25 亿 浮点运算
- ScratchV: ~57.1 亿 定点运算
- 预估加速: ~743x (在 x86 float32 vs RISC-V Q16.16 条件下)

## 相关 Topic

- Standalone RISC-V 编译器 (`onnx_to_riscv_standalone.py`)
- LLVM 代码生成后端 (`backend/llvm_codegen.py`)
- LLVM vs ScratchV Cache Compare (`llvm_cache_compare.py`)
- [ARCHITECTURE.md](../../reference/ARCHITECTURE.md) — 双路径详解
