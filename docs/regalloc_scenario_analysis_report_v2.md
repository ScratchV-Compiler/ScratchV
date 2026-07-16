# 寄存器分配器 (Linear Scan) 场景测试分析报告

## 概述

基于两轮系统性测试（共 64 个场景），对 ScratchV 编译器的线性扫描寄存器分配器 `regalloc_linear.py` v1.2 进行全面分析。覆盖 9 个高维场景（27/8 物理寄存器各跑一遍）、32 个通用边界场景、23 个核心分配语义场景。发现 **2 个正确性 bug**、**4 个优化方向**，以及 **多个场景级具体问题**。

测试环境：
- 标准池：27 个物理寄存器（a0-a7, t0-t6, s0-s11）
- 微缩池：3 个寄存器（a0, a1, a2）、1 个寄存器、0 个寄存器的极端环境
- 全部在 `regalloc_linear.py` v1.2 版本上运行

---

## 第一部分：场景级具体发现

### 1.1 第一轮测试（9 场景 × 27/8 物理寄存器）

#### 场景 1: dense_60 — 链式密集全重叠

构建方式：60 个 li 初始化 + 59 个链式 add，所有 vreg 几乎完全重叠 [0, ~119)

| 指标 | 值 |
|------|----|
| 总 vreg | 61 |
| 自溢数 | 34 |
| eviction | 0 |
| peak_active | 27 |
| 生成 asm sw | 92 |
| 生成 asm lw | 91 |

**问题**：
- 0 次 eviction（所有区间 end 几乎相同 → 第一个满池时 self-spill，后续全部 self-spill）
- 但生成代码中有 92 sw + 91 lw！这些全都是自溢 vreg 的 reload 操作
- 每个自溢 vreg 在每次使用前都 lw 一次，且每次 lw 到不同的 scratch 寄存器
- 寄存器复用率仅 1.0x（27 个寄存器对应 27 个 vreg，剩余 34 个全部自溢）

**场景洞察**：dense_60 暴露了"自溢 vreg 的 lw 开销没有被优化"的问题。34 个自溢 vreg 共生成 91 次 lw（每个 vreg ~2.7 次 use），如果能记忆上次使用的 scratch 寄存器，至少可省 34 次 lw。

#### 场景 2: spiral_60 — 复杂交织高压力

构建方式：60 条指令，每个指令定义和使用的 vreg 通过 mod 计算，形成复杂依赖网

| 指标 | v1.0 | v1.2 | 变化 |
|------|------|------|------|
| peak_active | 27 | 36 | +33% |
| spill 数 | 20 | 11 | -45% |
| sw 数 | 20 | 10 | -50% |

**问题**：
- 11 个 spill slot 全部几乎覆盖整个块：v39[13,58), v11[0,54), v18[6,55), ..., v55[0,60)
- **栈槽复用率为 0** — 所有 slot 区间都互相重叠，无法共享
- 寄存器复用率 2.2x（59 个 vreg 轮流使用 27 个物理寄存器）

**场景洞察**：spiral_60 是"eviction 频繁但 stack slot 不可复用"的典型。10 次 eviction 产生了 10 个 slot，但这些 slot 因为区间长（几乎覆盖整个块）而被"锁死"到块尾。

#### 场景 3: large_800x150 — 大规模高密度

| 指标 | v1.0 | v1.2 | 变化 |
|------|------|------|------|
| peak_active | 27 | 74 | +174% |
| spill 数 | 120 | 73 | -39% |
| sw 数 | 120 | 48 | -60% |
| alloc_map 条目 | 71 | 150 | +111% |

**v1.0 的关键问题**：
- alloc_map 只记录了 71/150 个 vreg（只记了分配到物理寄存器的）
- 25 个自溢 vreg 完全不在 alloc_map 中 → **vreg 泄漏到汇编**（致命）
- 每次自溢都误用 `phys_regs[0]` 作为 sw 源寄存器

**v1.2 的状态**：
- 48 次 eviction，但 farthest-end 策略每次选"最晚结束"的 victim
- 48 个 victim 中，有些只有 1 次后续 use，有些有 10+ 次
- 成本差异可达 10x，但分配器完全无视

**场景洞察**：large_800x150 是"成本感知 victim 选择"优化的最佳验证场景。48 次 eviction 中，如果每次选"后续 use 最少"的 victim，可大幅减少总 lw 数。

#### 场景 4-5: staggered_60 / bursts_64 / long_lived_40 — 低压力场景

这些场景在 27 寄存器 pool 下无溢出，核心差异在 peak_active：

| 场景 | v1.0 peak | v1.2 peak | 变化原因 |
|------|-----------|-----------|----------|
| staggered_60 | 2 | 3 | 统计位置修复 |
| bursts_64 | 8 | 9 | 同上 |
| long_lived_40 | 1 | 2 | 同上 |
| conv2d_30 | 17 | 18 | 同上 |
| butterfly_75 | 15 | 16 | 同上 |

**问题**：v1.0 的 peak_active 统计位置错误（放在 expire 之后、分配之前），导致所有场景的峰值偏小 1（因为当前区间的 active.append 还没执行）。v1.2 修复了。

#### 场景 6: noise_300x60 — 随机噪声模式

| 指标 | v1.0 | v1.2 | 变化 |
|------|------|------|------|
| peak_active | 27 | 36 | +33% |
| spill 数 | 23 | 14 | -39% |
| sw 数 | 23 | 9 | -61% |

与 spiral_60 模式一致，v1.2 大幅减少了虚假 sw。

---

### 1.2 第二轮测试（32 通用场景 + 23 核心分配场景）

#### 场景 T05_all_overlap — 47 个相同区间

47 个 vreg，全部区间 [0, 2)，27 个物理寄存器：

| 指标 | 值 |
|------|----|
| peak_active | 37/27 |
| 自溢 | 0 |
| eviction | 11 |
| sw | 11 |
| lw | 0 |

- 前 27 个分配到 a0-a6, t0-t6, s0-s11
- v27~v46（20 个）依次触发 eviction：farthest-end 策略选 v0~v26，被 evict 的 vreg 分配到寄存器
- 但所有 vreg 的 use 都在 pos=1，eviction 发生在 pos=27+
- → **sw 之后块内无后续 use**，11 次 sw 全部白写
- 这是 Opt 3（无后续 use 的 eviction sw 省略）的典型场景

#### 场景 T08_short_alternating — 密集短区间交替

162 个 vreg，短区间交替排列。27 寄存器 pool：

| 指标 | 值 |
|------|----|
| peak_active | 27/27 |
| 自溢数 | 54 |
| eviction | 1 |
| sw | 54 |
| lw | 54 |

**特征**：区间模式为 v[i]=[i, i+81)，81 个区间在 pos 0~80 段大量重叠。free_regs 耗尽后：
- v27~v80（54 个）自溢，每次使用前 lw → 54 次 lw
- 仅 1 次 eviction（某个自溢 vreg 的 end 足够小被 farthest-end 选中）
- **peak_active=27 被 pool_size 锁定**，完全反映不出实际有 162 个 vreg 需要处理

| 问题 | 说明 |
|------|------|
| Bug B | peak_active(27) 无法反映超 pool 压力（实际需 162 个） |
| 自溢+eviction 共存 | 54 个自溢 + 1 个 eviction，混合溢出模式 |
| 每次 lw 到不同 scratch | 54 次 lw 共 54 次 scratch 分配，无记忆 |

#### 场景 T15_spiral_60 — 复杂螺旋交织（第二轮）

60 步螺旋，与第一轮 spiral_60 同源但使用 mod 模式（第二轮版）：

| 指标 | 值 |
|------|----|
| peak_active | 35/27 |
| 自溢 | 3 |
| eviction | 9 |
| sw | 6 |
| lw | 4 |

**与第一轮 spiral_60 的差异**：
- 第一轮 spiral_60（`build_spiral`）：60 步 × 不同随机种子 → 1 自溢 + 10 eviction
- 第二轮 T15（`build_spiral_mod`，seed=42）：3 自溢 + 9 eviction
- 两轮都确认：eviction sw 到自溢 vreg 的 lw 之间的 reload 链正确

**关键问题**：
- 同时存在自溢（3个）和 eviction（9个），混合溢出模式
- loads(4) < stores(6)，说明部分 eviction 之后没有再 reload（块尾前未使用）

#### 场景 T24_tiny_pool_dense — 3 寄存器 + 30 vreg 全重叠

极端嵌入式场景，只有 3 个物理寄存器，30 个 vreg 全部 [0, 2)：

| 指标 | 值 |
|------|----|
| peak_active | 17/3 |
| 自溢 | 0 |
| eviction | 14 |
| sw | 14 |
| lw | 0 |

**特征**：
- 3 个寄存器全部被 v0/v1/v2 占用，v3 开始 evict，每次 evict farthest-end
- 14 次 eviction 生成 14 个 sw，但所有 use 在 pos=1 → sw 之后无 use
- peak_active=17 严重失真（实际有 30 个 vreg），但 17 仍 > pool(3)，勉强反映超压
- **所有 eviction sw 都是浪费的**（块内无后续使用）

**嵌入式/小寄存器场景总结**：

| 场景 | pool | vregs | 自溢 | evict | sw | lw | 问题 |
|------|------|-------|------|-------|----|----|------|
| T24_dense | 3 | 31 | 0 | 14 | 14 | 0 | peak=17>pool(3)失真, sw浪费 |
| T25_spiral | 3 | 30 | 4 | 9 | 6 | 11 | 混合溢出, loads>stores |
| T26_chain | 3 | 1 | 0 | 0 | 0 | 0 | 单vreg链, 完美 |

#### 场景 C05 — 重叠 vreg 共享物理寄存器

7 个实例：
```
v26[0,1) 和 v15[0,1) 都 → a0
v0[0,2)  和 v22[0,1) 都 → a3
v19[0,1) 和 v4[0,1)  都 → a5
...
```

**根因**：这是 Bug A（alloc_map evict 后未更新）的另一种表现。被 evict 的 vreg 在 alloc_map 中保留旧映射，导致后续分配时认为该物理寄存器仍被占用 → 新 vreg 分配到同一寄存器。

#### 场景 C17_reuse_after_evict — Bug A 的精确复现

2 个寄存器 pool：
```
v0=[0,11) → a0
v1=[1,3)  → a1
v2=[3,5)  → evict v0 (farthest end)
  sw a0, -4(sp)
  v2 → a0
  但 alloc_map[v0] 仍是 a0！

代码生成:
  li a0, 0
  sw a0, -4(sp)    # spill v0
  li a1, 1
  addi a0, a1, 1   # v1_2
  li a0, 2         # v2 分配到 a0
  addi a1, a0, 1   # v2_2
  addi a0, a0, 1   # v_out ← 应为 reload v0, 但读到 a0 (v2的值)!
```

#### 场景 T31_reload_efficiency — 自溢 reload 到不同 scratch

```
自溢 vreg v27:
  pos 1: add 使用 → lw t4, -4(sp)   # reload 到 t4
  pos 2: addi 使用 → lw t5, -4(sp)   # reload 到 t5 (不同寄存器!)
```

**问题**：`_pick_scratch` 每次选"第一个不在操作数中的空闲寄存器"，不记忆上次用了哪个。如果两次相邻 use 之间没有寄存器冲突，完全可以保留 t4。

#### 场景 T08_short_alternating — 密集短区间交替

162 个 vreg，短区间交替排列。27 寄存器 pool：

| 指标 | 值 |
|------|----|
| peak_active | 27 |
| 自溢数 | 54 |
| eviction | 33 |

**问题**：peak_active=27 被 pool_size 锁定，但实际需要 162 个寄存器。用户无法从 peak_active 看出真实压力。

#### 边界场景验证结果

| 场景 | 描述 | 结果 |
|------|------|------|
| empty_block | 空块 | 不崩溃，返回空 |
| single_inst | 单条指令 | 正确分配 |
| no_overlap | 108 个完全不重叠 | peak_active=2，无溢出 |
| one_reg_pool | 1 个寄存器 + 10 条指令链 | 全部 SPILL_，不崩溃 |
| zero_reg_pool | 0 个寄存器 | 不崩溃 |
| expire_boundary | end == current_pos | expire 正确处理 |
| redefine_same | 同一 vreg 反复定义 | 区间计算正确 |
| redef_after_spill | 自溢后重新定义 | 旧 SPILL_ 被覆盖 |
| dense_short_lived | 200 个密集短区间 | peak_active=2 |
| duplicate_operand | 操作数中同 vreg 出现两次 | rename 正确 |

---

## 第二部分：Bug 分析

### Bug A（严重）：evict 后 alloc_map 未更新

**影响范围**：所有触发 eviction 的场景

| 场景 | eviction 次数 | 受影响 vreg |
|------|-------------|-------------|
| large_800x150 | 48 | 48 个被 evict 的 vreg |
| spiral_60 | 10 | 10 个被 evict 的 vreg |
| noise_300x60 | 9 | 9 个被 evict 的 vreg |
| T05_all_overlap | 1+ | 所有被 evict 的 vreg |

**根因**：`regalloc_linear.py` 第 365-374 行，`spill()` 的 eviction 分支中：
1. `active.pop(spill_idx)` — 从活跃列表移除
2. `self.spill_code.append(sw)` — 生成 store
3. `free_regs.append(spill_reg)` — 释放寄存器
4. **没有 `self.alloc_map[spill_interval.vreg] = "SPILL_..."`** — 漏了！

**修复方案**：
```python
# 在 spill() 的 eviction 分支加入：
self.alloc_map[spill_interval.vreg] = f"SPILL_{spill_interval.vreg}"
```

### Bug B（中）：peak_active 被 pool_size 锁定

**影响范围**：所有存在自溢的场景

| 场景 | vregs | pool | peak_active | 自溢数 | evict | 实际需要 | 失真程度 |
|------|-------|------|-------------|--------|-------|----------|----------|
| dense_60 | 61 | 27 | 27 | 34 | 0 | 61 | peak 仅为实际的 44% |
| staggered_60 | 120 | 27 | 3 | 0 | 0 | 120 | peak=3, 严重失真 |
| T08_short_alternating | 162 | 27 | 27 | 54 | 1 | 162 | peak 仅为实际的 17% |
| T15_spiral_60 | 60 | 27 | 35 | 3 | 9 | 60 | peak=35, 较好反映 |
| T24_tiny_pool_dense | 31 | 3 | 17 | 0 | 14 | 31 | peak=17>pool, 相对可用 |

**根因**：self-spill 的区间（第 304-307 行）只标记了 `SPILL_` 前缀，没有执行 `active.append()`。active 长度永远不超过 pool_size。

**修复方案**：新增 `self.peak_self_spill` 单独统计自溢峰值，或修改 peak_active 为包含自溢的 vreg。

---

## 第三部分：优化方向

### Opt 1: 成本感知的溢出受害者选择

**触发场景**：large_800x150（48 次 eviction）、spiral_60（10 次）

**现状**：`spill()` 选 farthest-end 的活跃区间作为 victim。48 个 victim 中，后续 use 数分布不均（1 次到 10+ 次），但分配器完全无视。

**方案**：在 `spill()` 中，当多个活跃区间 end 相近时，优先溢出剩余使用次数最少的。需要 `LiveInterval` 增加 `uses_remaining` 信息。

**预估效果**：large_800x150 场景总 sw+lw 减少 5-15%。

### Opt 2: 栈槽复用

**触发场景**：spiral_60（11 slot，复用率 0%）、large_800x150（73 slot）

**现状**：`_get_spill_slot()` 每次分配新槽，永不回收。spiral_60 的 11 个槽区间全部 [0,58~60)，锁死到块尾。

**方案**：维护空闲槽列表，在 `_expire_old_intervals` 中回收已死亡 vreg 的栈槽。

**预估效果**：dense_60 类场景栈用量减少 30-50%。

### Opt 3: 自溢 reload 寄存器记忆（第二次机会）

**触发场景**：dense_60（92 sw + 91 lw）、T31_reload_efficiency

**现状**：自溢 vreg 在每次使用前都 lw reload，且每次可能 reload 到不同 scratch 寄存器。

**方案**：`_pick_scratch` 记住上次为某 vreg 选择的寄存器，如果仍空闲则复用。

**预估效果**：dense_60 场景省 ~34 次 lw（约 37%）。

### Opt 4: 无后续 use 的 eviction sw 省略

**触发场景**：T05_all_overlap（evict 时 pos=27+ > use=1）

**现状**：被 evict 的 vreg 总是生成 sw 写回栈，即使块内已无后续 use。

**方案**：eviction 时检查被 evict 的 vreg 在当前块内是否还有 use。若无，则省略 sw。

**风险**：需要确认块之间是否共享栈帧。若 block-level 分配且块间不共享，可安全省略。

---

## 第四部分：优先级建议

| 优先级 | 项目 | 影响 | 工作量 |
|--------|------|------|--------|
| P0 | 修复 Bug A（alloc_map evict 未更新） | 正确性 | ~3 行 |
| P1 | 修复 Bug B（peak_active 被锁定） | 诊断 | ~3 行 |
| P2 | Opt 4: 无后续 use sw 省略 | sw 开销 | ~10 行 |
| P3 | Opt 3: reload 寄存器记忆 | lw 开销 | ~30 行 |
| P4 | Opt 1: 成本感知 victim 选择 | 总 spill 开销 | ~20 行 |
| P5 | Opt 2: 栈槽复用 | 栈用量 | ~20 行 |

Bug A 必须优先修复——它会导致生成错误的汇编。Opt 4 和 Opt 3 实现成本低且效果可测量，建议作为 v1.3 的首批优化。
