# 课题23：Cache 模型（组相联 + LRU）

> **难度**：高 | **类型**：参考分析 | **源文件**：`scratchv/standalone/cache_model.py`
> **状态**：✅ 已完成

---

## 概述

Cache 模型模拟 L1 指令缓存（I$）和数据缓存（D$）的行为。它是 RV32IM 仿真器的附件——每条访存指令都经过 cache 模型，统计命中/缺失次数。理解 Cache 行为是解释"为什么相同指令数下性能不同"的关键。

---

## 理解背景

### 是什么？

Cache 模型（`cache_model.py`）模拟 **L1 指令缓存（I$）和数据缓存（D$）** 的行为。

```
CPU 执行指令
    ↓ 每次 fetch         ↓ 每次 load/store
┌──────────┐       ┌──────────┐
│  I$ Cache │       │  D$ Cache │
│ 64×2×32B │       │128×4×32B │
└──────────┘       └──────────┘
    ↓ miss             ↓ miss
    主内存 (无限容量，固定延迟)
```

### 为什么？

只统计动态指令数不够——两条指令序列可能指令数相同但 cache 行为截然不同。Cache 模型让你看到：
- 内存访问模式是否友好（空间局部性、时间局部性）
- ScratchV 和 LLVM 的缓存效率差距
- 为什么相同指令数下某个版本更快

### 核心概念

#### 组相联 Cache 参数

| 参数 | I$ | D$ | 含义 |
|------|-----|-----|------|
| sets | 64 | 128 | 组数 |
| ways | 2 | 4 | 每组的路数（相联度） |
| block_size | 32B | 32B | 每块 8 个 32-bit 字 |
| **总容量** | **4KB** | **16KB** | sets × ways × block_size |

#### 地址分解（32 位）

```
|     tag (剩余位)    | index (log2 sets) | offset (log2 block_size) |
|     20 位 (I$)      |      6 位 (64)    |        5 位 (32B)       |
```

#### LRU 替换策略

每组有 `ways` 个位置。当所有位置都满了又有新数据要进来时，淘汰**最久没被访问**的那个（Least Recently Used）。

#### Miss 分类

| 类型 | 含义 | 能优化吗？ |
|------|------|----------|
| **Compulsory miss** | 第一次访问某地址 | ❌ 不可避免 |
| **Conflict miss** | 不同地址映射到同一组，互相踢出 | ✅ 改数据布局 |

---

## 理解要点

1. 理解组相联 Cache 的三段地址分解（tag / index / offset）
2. 掌握 LRU 替换策略的实现（时间戳方式）
3. 区分 Compulsory miss 和 Conflict miss，知道哪种可优化
4. 能够独立运行 Cache 仿真并解读 hit_rate、MPKI 等指标
5. 理解 I$ 和 D$ 参数差异的原因（指令访问 vs 数据访问模式不同）

---

## 交付产物

- Cache 地址分解示意图（标注 tag/index/offset 位宽）
- CNN 模型的 Cache 分析报告（I$ 和 D$ 命中率、miss 分类）
- ScratchV vs LLVM 的 Cache 对比分析

---

## 代码走读

### CacheLine 结构

```python
class CacheLine:
    __slots__ = ("tag", "valid", "lru")
    def __init__(self):
        self.tag: int = 0        # 地址高位
        self.valid: bool = False # 有效位（初始无效）
        self.lru: int = 0        # 最后访问时间戳
```

### access 核心逻辑

```python
def access(self, addr, is_read):
    index = (addr >> self.offset_bits) & (self.sets - 1)
    tag = addr >> (self.offset_bits + self.index_bits)

    set_lines = self.sets_lines[index]  # 该组的所有路

    # 1. 先检查是否命中
    for line in set_lines:
        if line.valid and line.tag == tag:
            line.lru = self._timestamp
            self.stats.hits += 1
            return True

    # 2. Miss — 需要替换
    self.stats.misses += 1
    # 找空位或淘汰 LRU 最老的
    victim = min(set_lines, key=lambda l: l.lru if l.valid else -1)
    if not victim.valid:
        self.stats.compulsory_misses += 1
    else:
        self.stats.conflict_misses += 1
    victim.tag = tag
    victim.valid = True
    victim.lru = self._timestamp
    return False
```

---

## 动手练习

### 练习 1: 分析 CNN 模型的 cache 行为

运行 `llvm_cache_compare.py`，查看 ScratchV 和 LLVM 的 I$/D$ 命中率差异。

### 练习 2: 调整 cache 参数

把 D$ 的 ways 从 4 改为 8，重新运行对比。命中率提升多少？

### 练习 3: 实现 FIFO 替换策略

在 CacheSim 中添加 FIFO（先进先出）替换策略，和 LRU 对比命中率差异。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **写策略** | 当前使用 write-allocate + write-back，store miss 会先加载再写 |
| **地址对齐** | Cache 按 block 对齐，32B block 意味着低 5 位地址被忽略 |
| **LRU 精度** | 时间戳实现简单但数字可能溢出（对 32 亿次访问需要 64 位） |

---

## 进阶阅读

- Hennessy & Patterson: Computer Architecture, 附录 B（Cache 原理）
- [03-指标解读指南](../../guide/03-指标解读指南.md) — 如何解读 Cache 指标
- 相关 topic: [课题19 — Standalone RISC-V](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md) | [课题25 — LLVM vs ScratchV 对比](../25-LLVM对比工具/25-LLVM对比工具.md)

---

## 自学路线

- **第 1 周**：阅读 `cache_model.py` 全文，理解 `CacheLine` 数据结构和 `access()` 的核心逻辑。用纸笔模拟一个 4 组 × 2 路的小 cache，手动跟踪几次访问的命中/缺失。
- **第 2 周**：运行 `llvm_cache_compare.py`，收集 ScratchV 和 LLVM 的 I$/D$ 命中率数据。分析两者差异最大的访问模式（是 Conv 的 weight 访问还是 input/output 访问？）。
- **第 3 周**：尝试修改 cache 参数（ways=8, block_size=64），观察命中率变化。实验不同的替换策略（LRU vs FIFO vs Random），对比命中率差异。
- **第 4 周**：为 Cache 模型添加写策略选项（write-through vs write-back），实现并对比两种策略在 CNN 推理场景下的差异。撰写分析报告。
