# 课题26：TinyFive 对比工具

> **难度**：高 | **类型**：参考分析 | **源文件**：`scratchv/standalone/tinyfive_compare.py`
> **状态**：✅ 已完成

---

## 概述

TinyFive 对比工具将 ScratchV 和 LLVM 生成的 RISC-V 代码同时送入 TinyFive 仿真器，逐条执行并收集精确的指令统计（无需估算）。聚焦最内层循环体（per-MAC 级别）——这里占 >99% 执行时间。因为 TinyFive 只支持 RV32IM，对比在统一的 RV32IM 基准上进行。

---

## 理解背景

### 是什么？

TinyFive 对比工具将 ScratchV 和 LLVM 生成的 RISC-V 代码**同时送入 TinyFive 仿真器**，逐条执行并收集精确的指令统计（无需估算）。因为 TinyFive 只支持 RV32IM，它将对比聚焦在**最内层循环体**（per-MAC 级别）——这里占 >99% 执行时间。

### 为什么？

- **估算不可靠**：`--estimate` 用 `CONV_INSNS_PER_MAC=12` 估算，但实际指令数取决于具体优化
- **Spike 太重**：编译安装复杂，且 RV64FD 和 RV32IM 对比不在同一基准
- **TinyFive 轻量**：`pip install tinyfive` 即可，纯 Python，聚焦 RV32IM 最内层循环

### 核心概念

#### 聚焦最内层循环

不做全模型仿真（太慢），只提取 Conv2D 最内层循环体（~10-30 条指令），在 TinyFive 中跑一轮 MAC 操作：

```
ScratchV 最内层 (~12 instr/MAC):
    lw t0, 0(a0)        # load input
    lw t1, 0(a1)        # load weight
    mul t2, t0, t1      # Q16.16 乘法
    srai t2, t2, 16     # 定点截断
    add a2, a2, t2      # 累加

LLVM 等效 RV32IMF 最内层 (~6 instr/MAC):
    flw f0, 0(a0)       # load input (float)
    flw f1, 0(a1)       # load weight (float)
    fmadd.s f2, f0, f1, f2  # 融合乘加 (1条!)
```

#### 收集的指标

| 指标 | TinyFive 来源 |
|------|-------------|
| 总指令数 | `ProfiledMachine.instr_count` |
| Load/Store 次数 | `machine.print_perf()` |
| 乘加/分支次数 | `machine.print_perf()` |
| 寄存器使用 | x0-x31, f0-f31 使用位图 |
| 代码大小 | 汇编文本字节数 |

---

## 理解要点

1. 理解"聚焦最内层循环"策略——为什么只仿真 per-MAC 级别就能准确反映差距
2. 掌握 TinyFive ProfiledMachine 的 API（load_asm, run, print_perf）
3. 能够从汇编中提取最内层循环体并送入 TinyFive 执行
4. 理解 Q16.16 vs float32 在 per-MAC 指令数上的根本差异（12 vs 2-6 instr/MAC）
5. 了解 TinyFive 的局限（速度慢、只支持 RV32IM）

---

## 交付产物

- ScratchV 和 LLVM 最内层循环体的逐条指令对比
- TinyFive 仿真统计报告（per-MAC 指令数、Load/Store 次数、寄存器使用）
- Q16.16 vs float32 精度对比（相同输入下的输出差异）

---

## 代码走读

### 内层循环提取

```python
def extract_inner_loop_asm(asm_text, loop_label):
    """提取指定标签的循环体汇编"""
    lines = asm_text.split("\n")
    in_loop = False
    loop_body = []
    for line in lines:
        if loop_label in line:
            in_loop = True
            continue
        if in_loop:
            if line.strip().startswith("j ") or "bne" in line:
                break
            if line.strip() and not line.strip().startswith("#"):
                loop_body.append(line.strip())
    return loop_body
```

### ProfiledMachine 适配

```python
# TinyFive 有两种导入方式
try:
    from scratchv.simulator.tinyfive import ProfiledMachine
except ImportError:
    from tinyfive.machine import Machine
    # Fallback: 手动包装带计数器的 Machine
    class ProfiledMachine:
        def __init__(self, mem_size=4096):
            self._m = Machine(mem_size=mem_size)
            self.instr_count = 0
        def load_asm(self, lines):
            for line in lines:
                self._m.asm_str(line.strip())
        def run(self, n=None):
            # 包装 exe 方法, 每条指令计数器 +1
            ...
```

---

## 动手练习

### 练习 1: 提取并对比最内层循环

分别提取 ScratchV 和 LLVM 的 Conv2D 最内层循环汇编，手工对比每条指令的类型和数量。

### 练习 2: 在 TinyFive 中验证 Q16.16 精度

加载 ScratchV 的 Q16.16 循环体到 TinyFive，喂入已知的 input/weight 值，读取输出寄存器，和 float32 结果对比精度。

### 练习 3: 对比 per-MAC 指令数趋势

记录每次优化后的 per-MAC 指令数（从最初的 ~30 到现在的 ~12），画趋势图。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **TinyFive 速度** | ProfiledMachine 约 1000 instr/s，全量 CNN (32 亿) 不可行。只跑内层循环体 |
| **RV64FD 不兼容** | LLVM 输出是 RV64FD，TinyFive 只支持 RV32IM。需要将 LLVM 的 float32 循环转换为等效 RV32IMF 再比较 |
| **伪指令展开** | ScratchV 的 `li`, `mv`, `max` 等伪指令需要展开为 RV32IM 标准指令才能被 TinyFive 识别 |

---

## 进阶阅读

- [TinyFive 文档](https://pypi.org/project/tinyfive/)
- RISC-V M-extension: 乘法/除法指令规范
- 相关 topic: [课题24 — Spike 仿真](../24-Spike仿真/24-Spike仿真.md) | [课题19 — Standalone RISC-V](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md)

---

## 自学路线

- **第 1 周**：安装 TinyFive（`pip install tinyfive`），学习 ProfiledMachine API。手动加载几条 RISC-V 指令到 TinyFive 并执行，理解 `load_asm` / `run` / `print_perf` 的用法。
- **第 2 周**：分别提取 ScratchV 和 LLVM 的 Conv2D 最内层循环汇编。手工逐条对比指令类型和数量，计算 per-MAC 指令数。在 TinyFive 中验证。
- **第 3 周**：在 TinyFive 中执行 Q16.16 循环体，喂入已知 float32 值，对比输出精度。计算 Q16.16 定点运算的误差范围。
- **第 4 周**：研究 ScratchV per-MAC 指令数的优化历程（~30 → ~12），撰写优化技术总结。尝试提出进一步降低 per-MAC 指令数的方案（如融合 load + mul）。
