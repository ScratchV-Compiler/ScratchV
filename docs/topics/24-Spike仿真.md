# 课题24：Spike 仿真集成

> **难度**：高 | **类型**：参考分析 | **源文件**：`scratchv/standalone/spike_sim.py`
> **状态**：✅ 已完成

---

## 概述

Spike 仿真器集成将 ScratchV 编译出的 RV32IM flat binary 包装为最小 ELF32 格式，交给 Spike（RISC-V 官方黄金参考仿真器）执行，收集精确的动态指令数、I$/D$ 缓存命中率和指令热点。Spike 比 TinyFive 快 100-500 倍，是验证 ScratchV 代码正确性和性能的权威工具。

---

## 理解背景

### 是什么？

Spike 仿真器集成将 ScratchV 编译出的 RV32IM flat binary 包装为最小 ELF32 格式，交给 **Spike**（RISC-V 官方黄金参考仿真器）执行，收集精确的动态指令数、I$/D$ 缓存命中率和指令热点。

```
output.bin (flat binary)
    │
    ▼
spike_sim.py  ← 包装为 ELF32 + 设置栈/堆/输入输出buffer
    │
    ▼
Spike (外部 C++ 仿真器, ~100-500 KIPS)
    │
    ▼
spike-log-parser  ← 解析提交指令日志
    │
    ▼
动态指令数 + I$/D$ stats + PC 热点 + 指令trace
```

### 为什么？

ScratchV 内置的估算器（`--estimate`）是**分析估算**——用 `CONV_INSNS_PER_MAC=12` 乘以 MAC 数，速度快但不精确。TinyFive 仿真器是**逐条仿真**但很慢（~1 KIPS）。

Spike 是**工业级 RISC-V 黄金参考模型**——结果权威、速度快 100-500 倍、自带 cache 模拟和指令日志。

### 核心概念

#### ELF32 包装

ScratchV 输出是 flat binary（position-independent，无 header）。Spike 需要 ELF 格式。`spike_sim.py` 手工构造最小 ELF32：

```python
ELF_BASE = 0x80000000       # RISC-V DRAM 基地址
STACK_TOP = 0x85000000      # 栈顶（80MB）
INPUT_BUF  = 0x86000000     # a0 = 输入 buffer
OUTPUT_BUF = 0x87000000     # a1 = 输出 buffer
```

构造过程：手工编码 5 条启动指令（LUI + JAL）→ 拼接 ScratchV 代码 → 写 ELF header。

#### Spike Cache 配置

```bash
# I-cache: 64 组, 2 路, 32B/块
# D-cache: 128 组, 4 路, 32B/块
--ic 64:2:32 --dc 128:4:32
```

与 `cache_model.py` 使用相同的参数，确保结果可比。

#### 收集的指标

| 指标 | 来源 | 精度 |
|------|------|------|
| 动态指令数（提交） | Spike commit log | 精确 |
| 指令分类统计 | spike-dasm + 分类器 | 精确 |
| I$ 命中率 | Spike cache stats | 精确 |
| D$ 命中率 | Spike cache stats | 精确 |
| PC 热点 | 指令 trace 聚合 | 采样 |
| 执行时间 | Spike wall time | 参考 |

---

## 理解要点

1. 理解 ELF32 最小包装的原理：启动代码（5 条指令）、地址空间布局（ELF_BASE, STACK_TOP, INPUT/OUTPUT_BUF）
2. 掌握 Spike 的 cache 配置参数（--ic, --dc）与 cache_model.py 的对应关系
3. 能够独立运行 Spike 仿真并解读 commit log 输出
4. 理解 Spike vs TinyFive vs 分析估算三种方式的精度/速度权衡
5. 了解 PC 热点分析的基本方法

---

## 交付产物

- Spike 仿真运行日志（包含动态指令数、I$/D$ 命中率）
- Spike vs 分析估算的对比表（同一模型，两种方式）
- PC 热点分布图（确认 >99% 执行时间在 Conv2D 内层循环）

---

## 代码走读

### 最小 ELF32 构造

```python
def build_minimal_elf32(code: bytes) -> bytes:
    """手工构造包含启动代码 + ScratchV 代码的最小 ELF32"""
    # 启动代码（5 条指令）:
    startup = [
        rv_lui(2, 0x85000),   # sp = 0x85000000（栈顶）
        rv_lui(10, 0x86000),  # a0 = input buffer
        rv_lui(11, 0x87000),  # a1 = output buffer
        rv_jal(0, 5 * 4),     # 跳转到 ScratchV 代码（跳过启动）
        rv_ecall(),           # 程序退出
    ]
    payload = encode_instructions(startup) + code

    # 构造 ELF header + program header + section headers
    return build_elf(payload, entry=ELF_BASE, load_addr=ELF_BASE)
```

### Spike 调用

```python
def run_spike(elf_path, max_instr, ic_config, dc_config):
    cmd = [
        SPIKE,
        f"--ic={ic_config}",
        f"--dc={dc_config}",
        f"-m{max_instr}",
        "--log-commits",
        "--log-cache",
        elf_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return parse_spike_output(result.stdout, result.stderr)
```

---

## 运行方式与降级行为

```bash
# 1. 显式指定（优先级最高，CI 推荐固定版本）
python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140 \
    --spike-bin /opt/riscv/bin/spike --json

# 2. 环境变量（整机安装）
export SCRATCHV_SPIKE_HOME=/opt/coralnpu-spike-rv32   # 约定 $HOME/bin/spike
python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140
```

无 Spike 机器上的行为：

- 未找到 spike：默认打印 `SKIP: spike binary not found.`（含搜索位置与修复提示），
  退出码 0，不生成 ELF；加 `--require-spike` 则打印 `ERROR:` 并返回退出码 2。
- CLI 显式路径无效（不存在 / 不可执行 / 是目录）：`ERROR:` + 退出码 2；
  环境变量指向无效路径：`WARNING:` 后继续向后一层解析。
- `spike-dasm` / `spike-log-parser` 缺失只产生 `WARNING:`，不阻断仿真。
- `--json` 报告新增 `status` / `skip_reason` / `spike_binary` / `spike_tools`
  / `parse_warnings` / `tool_warnings` 字段，既有字段保持不变。

---

## 动手练习

### 练习 1: 对比 Spike vs 估算

对同一个 CNN 模型，分别用 `--estimate`（分析估算）和 Spike 仿真（真实执行）统计动态指令数，对比差异。

### 练习 2: 调整 cache 参数

把 D-cache 从 `128:4:32` 改为 `256:8:64`，重新仿真，观察命中率变化。

### 练习 3: 分析 PC 热点

从 Spike 的指令 trace 中提取 PC 热点，确认 >99% 执行时间确实在 Conv2D 最内层循环。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **Spike 二进制路径** | 不再硬编码：按 `--spike-bin` > `SCRATCHV_SPIKE_BIN` > `SCRATCHV_SPIKE_HOME` > `PATH` > 常见目录 > legacy 常量解析；`spike-dasm` / `spike-log-parser` 同规则 |
| **内存限制** | Spike 默认内存模型可能不够大，CNN 模型需要 `-m512`（512MB） |
| **执行时间** | 全量 CNN（32 亿指令）在 Spike 上可能跑数小时，用 `--max-instr` 限制 |
| **ELF 兼容性** | 最小 ELF32 只包含必要 header，某些 Spike 版本可能要求更完整的 ELF |

---

## 进阶阅读

- [Spike RISC-V ISA Simulator](https://github.com/riscv-software-src/riscv-isa-sim)
- ELF 格式规范: `man 5 elf`
- 相关 topic: [课题23 — Cache 模型](23-Cache模型.md) | [课题19 — Standalone RISC-V](19-Standalone-RISC-V编译器.md)

---

## 自学路线

- **第 1 周**：安装并编译 Spike RV32 版本。阅读 `spike_sim.py` 源码，理解 ELF32 包装流程（启动代码 → 拼接 payload → ELF header）。画出地址空间布局图。
- **第 2 周**：编译一个简单的 CNN 模型，分别用分析估算和 Spike 仿真统计动态指令数。对比差异并分析原因（估算公式的假设是否准确？）。
- **第 3 周**：使用 Spike 的 `--log-commits` 和 `--log-cache` 功能，提取指令 trace 和 cache 统计。用 PC 热点分析确认执行时间分布。
- **第 4 周**：尝试将 Spike 仿真集成到 CI 流程中（替代或补充 TinyFive）。设计 timeout 和 fallback 策略（Spike 可能不可用）。
