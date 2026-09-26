# Topic: LLVM 代码生成后端 (LLVM IR Codegen)

> **源文件**: `scratchv/backend/llvm_codegen.py`, `scratchv/standalone/onnx_to_llvm_standalone.py`
> **输入**: IR `Program` 或 `ONNXModel`
> **输出**: LLVM IR 文本 (`.ll`)
> **Target Triple**: `riscv64-unknown-elf`

---

## 概述

LLVM 代码生成器将 ScratchV IR 翻译为人类可读的 LLVM IR 文本。这是 ScratchV 的 float32 编译路径，与 Q16.16 定点路径互补。

## 两条 LLVM 路径

### 路径 A: `llvm_codegen.py` (库路径)

```
ONNX → ONNXParser → IR Program → LLVMCodegen → .ll 文本
```

- 接收 IR `Program`, 逐指令翻译为 LLVM IR
- 当前状态: 框架完整，但 NN 算子生成占位符 (`fadd 0.0, 0.0`)
- 使用 llvmlite (可选) 或直接文本生成

### 路径 B: `onnx_to_llvm_standalone.py` (独立路径)

```
ONNX → ONNXModel (手工解析) → LLVMCNNGenerator → .ll 文本 (完整 float32 IR)
```

- ~1400 行，独立于主编译器管线
- 真正的 float32 嵌套循环 LLVM IR 生成
- 支持: Conv2D, Gemm, MaxPool, ReLU, Sigmoid, Reshape

## 路径 B: LLVM IR 生成细节

### LLVMIRBuilder

辅助类，封装 LLVM IR 文本生成:

```python
class LLVMIRBuilder:
    def emit_header()          # target triple, data layout
    def new_register() -> str  # 分配 SSA 寄存器名 (%0, %1, ...)
    def emit_load(ptr, ty)     # %x = load float, float* %ptr
    def emit_store(val, ptr)   # store float %val, float* %ptr
    def emit_gep(base, idx)    # %p = getelementptr float, float** %base, i32 %idx
    def emit_fadd(a, b)        # %r = fadd float %a, %b
    def emit_fmul(a, b)        # %r = fmul float %a, %b
    def emit_fcmp_ogt(a, b)    # %c = fcmp ogt float %a, %b (ordered greater than)
    def emit_br(label)         # br label %label
    def emit_br_cond(c, t, f)  # br i1 %c, label %t, label %f
    def emit_call(fn, args)    # %r = call float @fn(float %arg)
    def emit_ret(val)          # ret float %val
```

### LLVMCNNGenerator

逐算子生成 LLVM IR。以 Conv2D 为例:

```llvm
define void @conv2d(float* %input, float* %weight, float* %bias, float* %output) {
entry:
  br label %oh_loop

oh_loop:
  %oh = phi i32 [0, %entry], [%oh_next, %oh_inc]
  %oh_cond = icmp slt i32 %oh, 32          ; OH = 32
  br i1 %oh_cond, label %ow_loop, label %exit

ow_loop:
  ; ... 类似嵌套循环 ...

oc_loop:
  ; 最内层: 6 层循环中的 MAC 计算
  %in_ptr = getelementptr float, float* %input, i32 %in_offset
  %in_val = load float, float* %in_ptr
  %w_ptr = getelementptr float, float* %weight, i32 %w_offset
  %w_val = load float, float* %w_ptr
  %mul = fmul float %in_val, %w_val           ; MAC: 乘法
  %acc_new = fadd float %acc, %mul            ; MAC: 累加
  br label %ic_inc

exit:
  ret void
}
```

## LLVM IR 特性

与 ScratchV 原生路径的关键差异:

| 特性 | LLVM IR | ScratchV 原生 |
|------|---------|-------------|
| 地址计算 | `getelementptr` 单指令 | 手动 `MUL + ADD` 链 (~5 条) |
| 浮点运算 | `fmul`, `fadd` 原生 | `MUL + SRAI 16` (Q16.16 定点) |
| 循环结构 | `phi` 节点 + `icmp` + `br` | 手动 label + jump |
| 类型安全 | LLVM 类型系统 | 原始 32-bit 整数 |
| 优化潜力 | LLVM `-O3` 全套 pass | 无后续优化 |

## 外部工具链集成

生成 `.ll` 文件后:

```bash
# 优化
opt -O3 output.ll -o optimized.ll

# 编译为 RISC-V 汇编
llc -O3 -march=rv64fd -mattr=+d,+f optimized.ll -o output.s

# JIT 执行 (调试用)
lli output.ll
```

## 相关 Topic

- [ARCHITECTURE.md](../../reference/ARCHITECTURE.md) — 双路径对比详解
- Backend Instruction Selection — RISC-V 原生代码生成
- LLVM vs ScratchV Cache Compare — `standalone/llvm_cache_compare.py`
