# ScratchV Architecture — ONNX → RISC-V 编译全流程

> **版本**: 0.3.0 | **更新**: 2026-06-05
>
> 本文档完整说明 ScratchV 如何将 ONNX CNN 模型编译为 RISC-V 可执行代码，覆盖 **ScratchV 原生路径（Q16.16 定点）** 和 **LLVM 路径（float32）** 两条编译管线。

---

## 目录

1. [项目总览](#1-项目总览)
2. [ScratchV 原生路径：ONNX → RISC-V 机器码](#2-scratchv-原生路径onnx--risc-v-机器码)
3. [LLVM 路径：ONNX → LLVM IR → RISC-V](#3-llvm-路径onnx--llvm-ir--risc-v)
4. [双路径对比](#4-双路径对比)
5. [模块地图](#5-模块地图)
6. [数据流与文件格式](#6-数据流与文件格式)

---

## 1. 项目总览

ScratchV 是一个 Python 实现的 AI 编译器，将 ONNX 深度学习模型编译为 RISC-V 指令集。

```
                    ┌─────────────────────┐
                    │    ONNX 模型 (.onnx)  │
                    │  Conv, Gemm, ReLU,   │
                    │  MaxPool, Sigmoid... │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   ┌──────────────────────┐     ┌──────────────────────┐
   │  ScratchV 原生路径    │     │     LLVM 路径         │
   │  (Q16.16 定点)        │     │  (float32 浮点)       │
   │                      │     │                      │
   │  ONNX → IR → RISC-V  │     │  ONNX → IR → LLVM IR │
   │  原生机器码 (.bin)    │     │  → llc → RISC-V      │
   └──────────────────────┘     └──────────────────────┘
              │                               │
              ▼                               ▼
        RV32IM 机器码                   RV64FD 汇编 / LLVM IR
        (无依赖, 零 runtime)            (可接 llc/opt/lli)
```

**设计哲学**:
- Python 标准库即可运行（核心编译器零外部依赖）
- 两条路径共享同一个 IR 前端，后端分叉
- 双路径对比 → 持续优化 ScratchV 超越 LLVM

---

## 2. ScratchV 原生路径：ONNX → RISC-V 机器码

### 2.1 概览

```
ONNX 模型文件 (.onnx)
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 1: ONNX 解析 (ProtoReader)         │
│  文件: scratchv/standalone/               │
│        onnx_to_riscv_standalone.py        │
│                                          │
│  手工解析 protobuf wire format            │
│  → 提取 graph nodes, initializers,        │
│     tensor shapes, weights, biases        │
│  → 构建 ONNXModel 对象                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 2: 内存规划 (MemoryPlan)            │
│                                          │
│  为权重/偏置/输入/输出分配地址空间          │
│  将 float32 权重转为 Q16.16 定点格式       │
│  → Q16.16: 高 16 位整数, 低 16 位小数     │
│  → 值域: [-32768, 32767.99998]           │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 3: RISC-V 代码生成                   │
│  (CNNRISCVGenerator)                     │
│                                          │
│  逐算子生成内联 RISC-V 机器码:              │
│                                          │
│  Conv2D:  6 层嵌套循环                      │
│    for oh in 0..OH:                      │
│      for ow in 0..OW:                    │
│        for oc in 0..OC:                  │
│          for kh in 0..KH:                │
│            for kw in 0..KW:              │
│              for ic in 0..IC:            │
│                load input[ic][h][w]       │
│                load weight[oc][ic][kh][kw]│
│                mul (Q16.16)              │
│                add (accumulate)          │
│              srai 16 (Q16.16 定点还原)     │
│          + bias[oc]                      │
│          store output[oc][oh][ow]         │
│                                          │
│  Gemm:    3 层嵌套循环 (支持 transB)       │
│  MaxPool: 5 层嵌套循环 + 边界检查           │
│  ReLU:    max(a0, zero)                  │
│  Sigmoid: 查表法 + 线性插值                │
│  Reshape: 直接拷贝                         │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 4: 机器码编码 + 链接                  │
│  (RISCVEmitter)                          │
│                                          │
│  将 RV32IM 指令编码为 32-bit 机器码          │
│  两遍扫描:                                 │
│    第一遍: emit 所有指令, 记录 label 位置     │
│    第二遍: 回填分支/跳转偏移量 (fixup)        │
│  → 输出 flat binary (.bin), position-     │
│    independent, 可直接加载到内存执行          │
└──────────────────┬───────────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │  output.bin   │  ← RV32IM flat binary
            │  output.s     │  ← (可选) 人类可读汇编
            └──────────────┘
```

### 2.2 关键设计决策

**Q16.16 定点运算**:
- 乘法: `MUL a, b` → 64 位结果 → `SRAI 16` 截断回 32 位
- 加法: 直接 `ADD`（小数点对齐）
- 精度损失: ~1.5e-5 相对误差，CNN 推理足够
- 代价: 每个 MAC 需 ~30 条指令（含地址计算、加载、乘法、定点还原）

**无 Runtime 依赖**:
- 所有算子通过内联循环实现，不依赖外部库
- Flat binary 可直接加载到 FPGA/ASIC 的 RISC-V 核执行
- 权重嵌入二进制文件中

**Q16.16 定点转换**:
```python
def float32_to_q16(value: float) -> int:
    """将 float32 转为 Q16.16 定点整数"""
    # 乘以 2^16 = 65536，取整
    return int(value * 65536.0) & 0xFFFFFFFF  # 32-bit 截断
```

### 2.3 命令行

```bash
# ScratchV 原生编译
python scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx \
    -o output.bin          # 输出 flat binary
    --asm output.s         # 同时输出汇编
    --estimate             # 性能估算 (动态指令数)
    --report               # 生成详细报告
```

---

## 3. LLVM 路径：ONNX → LLVM IR → RISC-V

### 3.1 概览

```
ONNX 模型文件 (.onnx)
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 1: ONNX 解析                        │
│  文件: scratchv/standalone/               │
│        onnx_to_llvm_standalone.py         │
│                                          │
│  复用 ONNXModel (同原生路径)                │
│  提取节点信息、权重、tensor 形状             │
│  → 权重保持 float32 (不转 Q16.16)          │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 2: LLVM IR 生成                      │
│  (LLVMCNNGenerator + LLVMIRBuilder)       │
│                                          │
│  逐算子生成 float32 LLVM IR:               │
│                                          │
│  ; 示例: Conv2D 内层循环的 LLVM IR 片段       │
│  %idx = getelementptr float, float** %w, │
│         i32 %w_offset                     │
│  %w_val = load float, float* %idx         │
│  %in_val = load float, float* %in_ptr     │
│  %mul = fmul float %in_val, %w_val        │
│  %acc = fadd float %acc, %mul             │
│                                          │
│  特性:                                    │
│  - 真正的 float32 运算 (fmul, fadd, fdiv)  │
│  - GEP 地址计算 (替代手动 MUL+ADD)          │
│  - 嵌套循环 + padding 边界检查              │
│  - 函数签名: float* (指针传递 tensor)        │
│  - target triple: riscv64-unknown-elf     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 3: LLVM 优化 + 目标代码生成 (外部)      │
│                                          │
│  选项 A: llc (静态编译)                      │
│    $ llc output.ll -o output.s           │
│       -O3 -march=rv64fd                   │
│    → RISC-V 64-bit float 汇编              │
│                                          │
│  选项 B: opt + llc (自定义优化)              │
│    $ opt -O3 output.ll | llc -O3 -o out.s │
│    → LLVM 全套优化 pass                    │
│                                          │
│  选项 C: lli (JIT 执行)                     │
│    $ lli output.ll                        │
│    → 直接在主机上 JIT 运行 (调试/验证)        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
        ┌────────────────────┐
        │  output.ll          │  ← LLVM IR 文本
        │  output.s (via llc) │  ← RISC-V 汇编
        └────────────────────┘
```

### 3.2 关键设计决策

**Float32 原生运算**:
- 每条 MAC = `fmul + fadd` ≈ 2 条指令（vs Q16.16 的 ~30 条）
- 无需定点转换开销
- 与 ONNX 原始精度一致，数值验证更简单

**LLVM IR 作为中间产物**:
- 可接 LLVM 全套优化 pass (`-O3`: 循环展开、向量化、指令合并...)
- 可多目标编译（RV64, x86-64, ARM）
- 人类可读的 SSA 文本格式，便于分析和调试

**GEP 地址计算**:
- LLVM 的 `getelementptr` 指令单条完成多维数组地址计算
- 替代 ScratchV 原生路径中手动的 ~5 条 `MUL + ADD` 链

### 3.3 命令行

```bash
# LLVM IR 生成
python scratchv/standalone/onnx_to_llvm_standalone.py models/graph/cnn.onnx \
    -o output.ll           # 输出 LLVM IR
    --compare              # 与 ScratchV 对比估算

# 用 llc 编译为 RISC-V 汇编
llc -O3 -march=rv64fd -mattr=+d output.ll -o output.s

# LLVM vs ScratchV 缓存对比分析
python scratchv/standalone/llvm_cache_compare.py \
    --json-output output/llvm_vs_scratchv.json
```

---

## 4. 双路径对比

### 4.1 架构差异

| 维度 | ScratchV 原生 | LLVM |
|------|-------------|------|
| **数值格式** | Q16.16 定点 (int32) | IEEE 754 float32 |
| **目标 ISA** | RV32IM (整数) | RV64FD (浮点) |
| **输出格式** | Flat binary (.bin) | LLVM IR (.ll) |
| **每条 MAC 指令数** | ~30 条 | ~2 条 |
| **依赖** | Python stdlib | 需要 llvmlite / llc |
| **运行时** | 零依赖, 直接执行 | 需要 C runtime (printf 等) |
| **可移植性** | RV32IM 通用 | 依赖 LLVM 工具链 |
| **优化能力** | 手动内联循环 | LLVM 全套优化 pass |
| **数值精度** | ~1.5e-5 相对误差 | 原生 float32 |

### 4.2 性能数据 (cnn.onnx: 3×Conv + 3×MaxPool + 2×FC)

| 指标 | LLVM (RV64FD) | ScratchV (RV32IM) | 比值 |
|------|--------------|-------------------|------|
| **静态指令数** | 1,059 | 785 | 0.74x |
| **动态指令数** | **18.5 亿** | **77.7 亿** | **4.2x** |
| 动态 Load | 5.27 亿 (28.6%) | 15.6 亿 (20.1%) | 0.34x |
| 动态 Store | 0.03 亿 (0.2%) | 5.19 亿 (6.7%) | 0.01x |
| 动态 FP/ALU | 5.25 亿 (28.4%) | 46.6 亿 (60.0%) | 0.14x |
| 动态 Branch | 0.81 亿 (4.4%) | 5.20 亿 (6.7%) | 0.16x |
| **D$ 命中率** | 88.75% | 88.75% | 1:1 |
| **D$ 缺失字节** | **19.1 亿** | **74.8 亿** | **3.9x** |
| **综合加速比** | **1x (baseline)** | **0.23x** | **LLVM 4.4x 快** |

### 4.3 为什么 ScratchV 慢 4.4x？

1. **Q16.16 定点开销** (主因): 每个 MAC 需要 ~15 条整数指令（`MUL` → `SRAI 16` → `ADD` + 溢出处理），而 LLVM float32 只需 `fmul + fadd` ≈ 2 条
2. **地址计算开销**: 无 `getelementptr`，手动 `MUL + ADD` 链 ~5 条/次
3. **Store 差距**: ScratchV 每个输出元素都需要 `SW` 写回，LLVM 可将中间结果保留在浮点寄存器中
4. **循环变换**: LLVM `-O3` 自动做 loop unrolling / interchange / vectorization，ScratchV 无任何循环优化

### 4.4 超越路线

ScratchV 的优势不在于通用编译，而在于 **CNN 专用优化**:
1. **算子融合**: Conv+ReLU 端到端融合，消除中间 store/load
2. **内存布局定制**: CHW→HWC 变换提升 cache 局部性
3. **专用指令**: 如果硬件支持 SIMD/P 扩展，LLVM 不一定能自动利用
4. **精度/速度折中**: Q16.16 在推理场景足够，可尝试 Q8.8 进一步减少指令

---

## 5. 模块地图

```
ScratchV 项目 — 完整模块地图
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                        前端 (Frontend)                        │
├─────────────────────────────────────────────────────────────┤
│  scratchv/frontend/onnx_parser.py     ONNX 模型解析           │
│  scratchv/frontend/dsl_parser.py      DSL 解析 (无 ONNX 依赖) │
│  scratchv/frontend/dsl_extended.py    扩展 DSL (if/while)     │
│  scratchv/frontend/dsl_errors.py      DSL 错误提示            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      中间表示 (IR)                             │
├─────────────────────────────────────────────────────────────┤
│  scratchv/ir/types.py        OpCode, Value, DataType 定义     │
│  scratchv/ir/builder.py      IRBuilder (三地址码构造器)       │
│  scratchv/ir/printer.py      IRPrinter (文本输出)            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│   优化器 (Optimizer)        │   │   分析 (Analysis)          │
├──────────────────────────┤   ├──────────────────────────┤
│ constant_folding.py       │   │ cfg_builder.py             │
│ dead_code.py              │   │ ir_verifier.py             │
│ peephole.py               │   └──────────────────────────┘
│ muladd_fusion.py          │
│ licm.py (循环不变量外提)    │
└──────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (Backend) — 两条路径                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RISC-V 代码生成:                   LLVM 代码生成:              │
│  ├ instruction_select.py            └ llvm_codegen.py        │
│  ├ inst_select_ext.py                   (IR → LLVM IR)       │
│  ├ register_alloc.py                                          │
│  ├ regalloc_linear.py                                        │
│  ├ asm_emit.py                                               │
│  ├ asm_peephole.py    (后优化)                                 │
│  ├ const_merge.py     (常量合并)                                │
│  ├ asm_beautifier.py  (格式化)                                 │
│  ├ inst_scheduler.py  (指令调度)                                │
│  ├ inst_counter.py    (指令统计)                                │
│  ├ cycle_estimator.py (流水线模拟)                               │
│  ├ riscv_encoder.py   (汇编→机器码)                              │
│  └ machine_types.py   (类型定义)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 独立工具 (Standalone)                           │
├─────────────────────────────────────────────────────────────┤
│  onnx_to_riscv_standalone.py  ONNX → RV32IM 原生编译器         │
│  onnx_to_llvm_standalone.py   ONNX → LLVM IR 编译器           │
│  benchmark.py                 RV32IM 仿真器 + 性能计数器       │
│  cache_model.py               Set-associative cache 模型      │
│  spike_sim.py                 Spike 仿真器包装                 │
│  run_spike_bench.py           带 cache 的完整仿真器            │
│  llvm_cache_compare.py        LLVM vs ScratchV 缓存对比       │
│  tinyfive_compare.py          TinyFive 静态对比               │
│  compare_codegen.py           代码生成质量对比                  │
│  rv32_bench.py                RV32 全量 Benchmark             │
│  bench_report.py              HTML/JSON/MD 报告生成           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    CI / Dashboard                             │
├─────────────────────────────────────────────────────────────┤
│  scratchv/ci/ci_benchmark.py   CI 基准测试编排器               │
│  scratchv/ci/dashboard.py     性能对比仪表盘 (纯静态 HTML)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              模拟器 / 验证 (Simulator / Verification)          │
├─────────────────────────────────────────────────────────────┤
│  scratchv/simulator/tinyfive.py   TinyFive 适配器             │
│  scratchv/simulator/rv32_emulator.py  RV32IM 全功能仿真器     │
│  scratchv/verification/verifier.py  ONNX Runtime / numpy 验证 │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 数据流与文件格式

### 6.1 ScratchV 原生路径文件转换链

```
models/graph/cnn.onnx               ← 输入: ONNX protobuf
       │
       │  onnx_to_riscv_standalone.py
       ▼
output.bin                          ← RV32IM flat binary (position-independent)
output.s                            ← (可选) GAS 汇编
benchmark_reports/cnn_scratchv.s    ← CI 产物
```

### 6.2 LLVM 路径文件转换链

```
models/graph/cnn.onnx               ← 输入: ONNX protobuf
       │
       │  onnx_to_llvm_standalone.py
       ▼
benchmark_reports/cnn.onnx_llvm.ll  ← LLVM IR 文本
       │
       │  llc -O3 -march=rv64fd
       ▼
output.s                            ← RISC-V 64 汇编
```

### 6.3 CI 产物数据流

```
models/graph/cnn.onnx
       │
       ├──→ onnx_to_riscv_standalone.py → cnn_scratchv.s → tinyfive_compare.py
       │                                                          │
       └──→ onnx_to_llvm_standalone.py  → cnn.onnx_llvm.ll        │
                     │                                            │
                     └──→ llvm_cache_compare.py ←─────────────────┘
                              │
                              ├── benchmark_reports/llvm_vs_scratchv.json
                              ├── benchmark_reports/llvm_vs_scratchv.md
                              └──→ dashboard.py
                                       │
                                       ▼
                              benchmark_reports/dashboard.html
                                       │
                                       │  upload-pages-artifact
                                       ▼
                              https://scratchv-compiler.github.io/ScratchV/
```

### 6.4 Q16.16 定点格式

```
32-bit 整数的高 16 位 = 整数部分
32-bit 整数的低 16 位 = 小数部分

例: 3.14159 → int(3.14159 × 65536) = 205887
     205887 = 0x0003243F
     高 16 位: 0x0003 = 3 (整数部分)
     低 16 位: 0x243F ≈ 0.14159 (小数部分, 精度 1/65536 ≈ 1.5e-5)

乘法: MUL a, b → 64-bit → SRAI 16 (右移 16 位截断)
     因为 (a × 2^16) × (b × 2^16) = (a × b) × 2^32
     需要右移 16 位还原为 Q16.16 → 结果 × 2^16

加法: ADD a, b  (直接加，小数点已对齐)
```

---

## 附录: 关键命令速查

```bash
# ScratchV 原生编译 + 估算
python scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx \
    -o output.bin --asm output.s --estimate --report

# LLVM IR 生成 + 对比
python scratchv/standalone/onnx_to_llvm_standalone.py models/graph/cnn.onnx \
    -o output.ll --compare

# LLVM vs ScratchV 缓存对比
python scratchv/standalone/llvm_cache_compare.py \
    --json-output output/llvm_vs_scratchv.json

# TinyFive 静态分析
python scratchv/standalone/tinyfive_compare.py \
    --json output/tinyfive_compare.json

# Dashboard 生成
python scratchv/ci/dashboard.py \
    --llvm-json output/llvm_vs_scratchv.json \
    --tinyfive-json output/tinyfive_compare.json \
    -o dashboard.html

# 一键 CI 全部
make bench-ci

# Harness 验证
python .claude/harness/verify/run.py --level L2
```

---

## 相关文档

- 📘 **新手入门**: [00-环境搭建](../guide/00-环境搭建指南.md) → [01-概念入门](../guide/01-编译器概念入门.md) → [02-快速上手](../guide/02-快速上手教程.md)
- 📊 **指标解读**: [03-指标解读指南](../guide/03-指标解读指南.md)
- 🎓 **深入学习**: [课题索引](../topics/INDEX.md) — 14 个第一期课题
- 🌐 **在线课程**: [交互式课程站点](https://scratchv-compiler.github.io/ScratchV/docs/index.html)（本地用 `make docs` 生成）
- 📖 **全部文档**: [INDEX.md](../INDEX.md)
