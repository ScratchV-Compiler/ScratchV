# 寄存器分配 Benchmark 设计文档

## 1. 概述

对 ScratchV 的**线性扫描寄存器分配器**（`LinearScanAllocator`）进行正确性与性能评估，并在 CNN 路径上与 LLVM 后端做指令数及寄存器溢出的交叉对比。

**设计目标**：

- **正确性**：验证无溢出 / 有溢出两种场景的分配结果合法（无未解析 vreg、有效 opcode）
- **性能**：测量分配耗时（均值 / 标准差）、活跃区间峰值压力
- **可对比**：每项输出统一的 `reg_spill_count` 指标，支持回归对比和筛选
- **跨后端对比**：同一 ONNX 模型经 ScratchV 和 LLVM 两条路径编译，对比静态指令数、opcode 类别分布、溢出/帧操作数量

入口：`benchmarks/test_regalloc/run_benchmark.py`  
单文件运行：直接执行 `bench_simple.py` / `bench_dense.py` / `bench_cnn.py`

---

## 2. 架构

```
benchmarks/test_regalloc/
├── __init__.py           
├── bench_utils.py        共享常量（opcode 分类、callee-saved 集合）、llvmlite IR→RISC-V
├── bench_simple.py       Benchmark 1 — 无溢出正确性
├── bench_dense.py        Benchmark 2 — 溢出正确性
├── bench_cnn.py          Benchmark 3 — CNN 集成 + LLVM 对比
└── run_benchmark.py      运行器：汇总 3 路输出 → JSON / HTML / Markdown 报告
```

### 2.1 统一接口

每个 benchmark 文件导出：

```python
def run_bench(...) -> dict:
    """返回统一结构的统计 dict，必需键见 §4。"""
```

运行器遍历三个 `run_bench()`，收集 dict 生成报告。

### 2.2 数据流

```
                  ┌──────────────────┐
                  │ run_benchmark.py │
                  └──────┬───────────┘
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    bench_simple   bench_dense    bench_cnn
    run_bench()    run_bench()    run_bench()
         │              │              │
         ▼              ▼              ▼
   LinearScanAllocator  ...    LinearScanAllocator
   .allocate()               .allocate()
   .report() / to_dict()     .report() / to_dict()
         │              │         │         │
         ▼              ▼         ▼         ▼
      {dict}         {dict}    {dict}   {LLVM dict}
         │              │         │         │
         └──────────────┴─────────┴─────────┘
                        │
                        ▼
              JSON / HTML / MD 报告
```

---

## 3. 三项 Benchmark

### 3.1 Benchmark 1 — 简单算术（无溢出）

**文件**：`bench_simple.py`  
**目的**：验证物理寄存器充足时分配器产生零溢出。

| 参数 | 值 |
|------|-----|
| 虚拟寄存器 | 5 个（`v0`–`v4`） |
| 物理寄存器池 | 8 个（`r0`–`r7`） |
| 指令数 | 10 条（随机 add/sub/mul/and/or） |
| 块生成 | `_gen_block(num_insts=10, num_vregs=5)` |
| 断言 | `reg_spill_count == 0` → `valid = True` |



### 3.2 Benchmark 2 — 密集计算（触发溢出）

**文件**：`bench_dense.py`  
**目的**：人为制造高寄存器压力，迫使分配器溢出，验证溢出代码生成正确。

| 参数 | 值 |
|------|-----|
| 虚拟寄存器 | 30 个（`v0`–`v29`） |
| 物理寄存器池 | 5 个（`r0`–`r4`） |
| 指令数 | 80 条（30 条 define + 50 条交叉引用） |
| 块生成 | 阶段 1: 逐个定义 vreg → 创建长 live range；阶段 2: 随机交叉引用保持活跃 |
| 断言 | `reg_spill_count > 0` → `valid = True` |



### 3.3 Benchmark 3 — CNN 集成 + LLVM 对比

**文件**：`bench_cnn.py`  
**目的**：通过真实 ONNX 模型编译流水线验证分配器，并与 LLVM 后端做指令数和溢出对比。

**ScratchV 编译流水线**：

```
ONNXParser → IR Program (17 ops)
  → ConstantFolder → DeadCodeEliminator
  → InstructionSelector → MachineInstr (57, with vregs)
  → block_from_machine_instrs → [LsInstruction]
  → LinearScanAllocator.allocate()
  → get_allocated_code() → RISC-V 伪指令
  → _validate_asm()         # 检查无未解析 vreg、合法 opcode
```

**LLVM 对比流水线**：

```
convert_onnx_to_llvm(model)     → LLVM IR (866K lines, 183MB)
  → llvmlite IR → RISC-V asm    → 静态指令数 + opcode 分布
  → 汇编启发式                   → 溢出 slot / 帧操作 近似统计
```

| 参数 | 值 |
|------|-----|
| 模型 | `models/graph/cnn.onnx`（可 CLI 覆盖） |
| IR 指令 | 17 条（3×conv + 3×relu + 3×maxpool + 2×gemm + sigmoid + 2×reshape） |
| 物理寄存器 | `_INT_REGS`（28 个） |
| ScratchV 输出 | ~57 条伪指令（mv/mul/add/slt/bnez…） |
| LLVM 输出 | ~1099 条（RV64FD O2，真实循环展开） |
| 断言 | `asm_valid == True` |

> **注意**：ScratchV 侧输出 57 条**伪指令**（conv/maxpool 等语义级操作由仿真器实现），LLVM 侧输出 1099 条**自包含机器指令**（每个 conv 展开为 5 重嵌套循环的完整 RISC-V 指令序列）。`instr_ratio_fd ≈ 23.89x` 反映的是抽象层级差异而非优化能力差距，因此还提供 opcode **类别分布**作为跨层级可比指标。

---

## 4. 指标规范

### 4.1 所有 benchmark 通用键

| 键 | 类型 | 来源 | 说明 |
|----|------|------|------|
| `mean_s` | `float` | `perf_counter` 均值 | 单次分配耗时（秒） |
| `stdev_s` | `float` | `stdev` | 耗时标准差 |
| `vreg_count` | `int` | `len(alloc.alloc_map)` | 已分配的虚拟寄存器数 |
| `spills` | `int` | `len(alloc._spill_slots)` | 溢出 slot 数（别名） |
| `reg_spill_count` | `int` | 同上 | **统一溢出指标键**（接口规范） |
| `peak_active` | `int` | `alloc.peak_active` | 峰值同时活跃的物理寄存器数 |
| `asm_lines` | `int` | `len(code.splitlines())` | 汇编输出行数 |
| `valid` | `bool` | 由 `run_bench()` 设置 | 该项是否通过断言 |

### 4.2 Benchmark 特有键

**bench_dense**：

| 键 | 说明 |
|----|------|
| `reloads` | `lw ... # reload` 注释行数 |

**bench_cnn (ScratchV 侧)**：

| 键 | 来源 | 说明 |
|----|------|------|
| `vreg_total` | ONNXParser→isel | 编译流水线中出现的 vreg 总数 |
| `ir_inst_count` | Program 指令计数 | IR 层操作数（=17） |
| `machine_instrs` | `len(machine)` | MachineInstr 数量（=57） |
| `sv_static_instrs` | `count_riscv_instrs()` | 汇编指令数（~46，不含标签/注释） |
| `sv_cats` | `count_riscv_instrs()` | opcode 原始计数 |
| `sv_cat_buckets` | `_op_categories()` | 6 类汇总（ALU/Load/Store/Branch/Mul/Other） |
| `asm_errors` | `_validate_asm()` | 未解析 vreg / 未知 opcode 列表 |
| `asm_valid` | `len(asm_errors) == 0` | 汇编合法性 |
| `greedy_time_s` | Greedy allocator | Greedy分配器耗时（baseline） |
| `greedy_out_instrs` | — | Greedy分配器输出指令数 |

**bench_cnn (LLVM 对比侧)**：

| 键 | 说明 |
|----|------|
| `llvm_im_instrs` | LLVM RV64IM 静态指令数 |
| `llvm_fd_instrs` | LLVM RV64FD 静态指令数 |
| `instr_ratio_fd` | `llvm_fd_instrs / max(sv_static_instrs, 1)` |
| `llvm_fd_cats` | opcode 原始计数 |
| `llvm_fd_cat_buckets` | 7 类汇总（+ Stack） |
| `llvm_spill_slots` | 4 字节 sp 访问 ÷ 2（**近似**） |
| `llvm_frame_save` | `sd` 到 callee-saved 的计数 |
| `llvm_frame_restore` | `ld` 到 callee-saved 的计数 |

### 4.3 `reg_spill_count` 规范

- **经过 regalloc 的路径**：直接取自 `alloc._spill_slots` 长度 → 精确值
- **不经过 regalloc 的路径**：LLVM 侧 `reg_spill_count` 是本路径的 ScratchV 精确值（0）；LLVM 近似溢出独立为 `llvm_spill_slots`，不污染统一键
- **降级路径**：当 libLLVM 不可用时，`llvm_fd_instrs`/`llvm_spill_slots` 等键不存在于 dict 中，报告渲染 fallback 到 `"-"`

---

## 5. LLVM 溢出统计

由于 libLLVM 缺少 regalloc pass 统计入口，LLVM 侧无法获取精确的寄存器溢出计数，通过汇编层面识别特定模式统计溢出：

| 形态 | 正则 / 判定 | 含义 |
|------|------------|------|
| 帧保存 | `sd <callee-saved>, N(sp)` | prologue 保存 callee-saved 寄存器 |
| 帧恢复 | `ld <callee-saved>, N(sp)` | epilogue 恢复 callee-saved 寄存器 |
| 溢出 | `sw/lw/fsw/flw <reg>, N(sp)` | 寄存器值被 spill→reload（4 字节） |

**callee-saved 集合**（RV64 ABI）：`ra, fp, s0–s11, fs0–fs11`

---

## 6. 输出格式

### 6.1 终端

```
============================================================
  ScratchV — Register Allocation Benchmark Suite
============================================================
  1. Simple:  reg_spill_count=0, mean=0.019ms  ✓
  2. Dense:   reg_spill_count=15, mean=0.084ms  ✓
  3. CNN:     reg_spill_count=0, mean=0.052ms  ✓
     LLVM:   RV64FD=1099 instrs (23.89x vs ScratchV 46)

  Total: 25173.6ms  PASS

  JSON report: /tmp/rg_spill.json
  HTML report: /tmp/rg_spill.html
  Markdown:    /tmp/rg_spill.md
```

### 6.2 JSON

标准化结构，`_` 前缀字段（如 asm 文本、allocator 实例）被排除：

```json
{
  "timestamp": "2026-08-09T...",
  "total_time_s": 25.17,
  "repeats": 3,
  "results": {
    "1. Simple Arithmetic": {
      "mean_s": 1.9e-05, "reg_spill_count": 0, "peak_active": 5,
      "asm_lines": 10, "valid": true
    },
    "2. Dense Computation": {
      "mean_s": 8.4e-05, "reg_spill_count": 15, "peak_active": 14,
      "asm_lines": 179, "reloads": 61, "valid": true
    },
    "3. CNN Integration": {
      "mean_s": 5.2e-05, "reg_spill_count": 0,
      "sv_static_instrs": 46, "llvm_fd_instrs": 1099,
      "instr_ratio_fd": 23.89, "llvm_spill_slots": 87,
      "llvm_frame_save": 70, "llvm_frame_restore": 70,
      "llvm_fd_cat_buckets": {"ALU": 346, "Load": 98, ...},
      "asm_valid": true, "valid": true
    }
  }
}
```

### 6.3 HTML / Markdown 汇总表

| Benchmark | Mean(ms) | Std(ms) | Vregs | Spills | Peak | Reloads | Asm | LLVM-FD | Ratio | LLVM-Spill | Valid |
|-----------|----------|---------|-------|--------|------|---------|-----|---------|-------|------------|-------|
| 1. Simple Arithmetic | 0.019 | 0.011 | 5 | 0 | 5 | - | 10 | - | - | - | ✓ |
| 2. Dense Computation | 0.084 | 0.016 | 30 | 15 | 14 | 61 | 179 | - | - | - | ✓ |
| 3. CNN Integration | 0.052 | 0.015 | 30 | 0 | 11 | - | 57 | 1099 | 23.89 | 87 | ✓ |

---

## 7. 使用方式

```bash
# 运行全部 3 项 benchmark，生成三格式报告
python benchmarks/test_regalloc/run_benchmark.py \
    --repeats 30 \
    --output-json report.json \
    --output-html report.html \
    --output-md report.md

# 单独运行某项
python benchmarks/test_regalloc/bench_simple.py --repeats 100
python benchmarks/test_regalloc/bench_dense.py --repeats 50
python benchmarks/test_regalloc/bench_cnn.py \
    --cnn-path models/graph/cnn.onnx --repeats 30
```

---

## 8. TODO

1. **优化报告输出格式**
2. **将通用的Benchmark组件进一步抽象到`bench_utils.py`当中**
3. **目前ONNX模型的编译路径指令选择方面无法完全正确生成算子的汇编指令，与LLVM后端对比不公平**
