# 课题17：寄存器分配瓶颈场景测试报告

## 字段说明

| 字段 | 含义 |
|------|------|
| **pool** | 物理寄存器池大小（标准 27 个：a0-a7, t0-t6, s0-s11） |
| **peak** | 一次 allocate() 扫描中 active 列表的最大长度。反映单个时间点上同时活跃的 vreg 数量 |
| **self_spill** | 自溢数。当无空闲寄存器且所有活跃区间 end 都小于当前区间时，当前区间自己溢出。自溢的 vreg 不占用物理寄存器，每次使用前需要 lw reload |
| **evict** | eviction 次数。选择 active 中 end 最远的 vreg 踢出（farthest-end 策略），释放其物理寄存器给当前区间。被 evict 的 vreg 通过 sw 写回栈 |
| **slot** | 分配的 spill slot 总数。每个 vreg 被溢出时分配一个栈槽（spill slot），slot 可以被复用（多个 vreg 使用同一 offset），也可以不复用（slot_reuse 反映复用比例） |
| **sw** | 生成的 store 指令总数。包含 eviction 产生的 sw 和自溢 vreg 的 sw（定义后写回栈） |
| **lw** | 生成的 load 指令总数。包含自溢 vreg 每次使用前的 lw reload 和 eviction 后续使用的 lw reload |
| **vregs** | 总虚拟寄存器数。包含分配到物理寄存器的和溢出的 |
| **lines** | 最终生成的汇编代码行数（含注释和空行） |
| **bytes** | 最终生成的汇编代码字节数 |
| **asm_lines** | 同 lines |
| **asm_bytes** | 同 bytes |
| **block_len** | 原始的 LsInstruction 块长度（输入长度） |
| **avg_uses** | 平均每个 vreg 的使用次数（uses / vregs） |
| **max_uses** | 单个 vreg 的最大使用次数 |
| **slot_reuse** | spill slot 复用比例。1 - (unique slot offsets / total slots)。0% 表示所有 slot 都不复用 |
| **redundant_sw** | 冗余 store 数。同一个 slot 在连续两次 sw 之间没有被修改，第二次 sw 写入相同值 |
| **redundant_reloads** | 冗余 reload 数。同一个自溢 vreg 在相邻位置被 reload 到不同 scratch 寄存器 |
| **spill_ops** | spill_code 中的条目数（sw + lw 的总操作数） |
| **膨胀比** | asm_lines / block_len，反映 spill code 导致的代码膨胀程度 |
| **intervals** | compute_live_intervals 计算出的 LiveInterval 数 |
| **PEAK_LOCKED** | peak_active 等于 pool_size，且 vregs > pool，说明 peak_active 被池大小锁定，无法反映真实寄存器压力 |
| **STORE_ONLY** | 有 sw 但无 lw，说明 eviction 后没有生成 reload 指令（可能是 Bug A 导致） |
| **LOADS>STORES** | lw 数量超过 sw 数量，说明同一定义被多次 reload |
| **SLOT_NO_REUSE** | slot_reuse 为 0% 且 slot 数 > 5，说明 slot 完全没有被复用 |

---

## 一、活跃度维度（4 场景）

**考察焦点**：vreg 数量超过物理寄存器时的 entry-level 溢出行为

### 场景 A01：轻度超压（pool+5, 完全重叠）

**含义**：vreg 数量刚好超过物理寄存器 5 个，观察最轻微的溢出行为和 eviction 入门门槛。

**构建方式**：
```python
# 32 个 li 全在 pos=0 定义 + 1 条 add 使用全部 33 个 vreg
# 所有 interval 完全重叠 [0, 1)
block = (
    [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
     for i in range(POOL + 5)] +
    [LsInstruction(1, 'add', ['v_out'] + [f'v{i}' for i in range(POOL + 5)],
                   defines={'v_out'}, uses={f'v{i}' for i in range(POOL + 5)})]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 33 | 27 | 30 | 0 | 3 | 3 | 3 | 0 | 36 | 600 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 0.97 | 1 | 0.0% | 0 | 0 |

---

### 场景 A02：中度超压（pool×2, 完全重叠）

**含义**：vreg 数量为 pool 的 2 倍（55 个），溢出数量从轻微变为明显。

**构建方式**：
```python
# 54 个 li + 1 条 add，所有 interval 完全重叠 [0, 1)
block = (
    [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
     for i in range(POOL * 2)] +
    [LsInstruction(1, 'add', ['v_out'] + [f'v{i}' for i in range(POOL * 2)],
                   defines={'v_out'}, uses={f'v{i}' for i in range(POOL * 2)})]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 55 | 27 | 41 | 0 | 14 | 14 | 14 | 0 | 69 | 1280 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 0.98 | 1 | 0.0% | 0 | 0 |

---

### 场景 A03：重度超压（200 vreg, 完全重叠）

**含义**：200 个 vreg 争夺 27 个寄存器（约 7.5x pool），模拟极端入口函数或展开后的大基本块。

**构建方式**：
```python
# 200 个 li + 1 条 add，完全重叠区间
block = (
    [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
     for i in range(200)] +
    [LsInstruction(1, 'add', ['v_out'] + [f'v{i}' for i in range(200)],
                   defines={'v_out'}, uses={f'v{i}' for i in range(200)})]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 201 | 27 | 114 | 0 | 87 | 87 | 87 | 0 | 288 | 6046 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 1.00 | 1 | 0.0% | 0 | 0 |

---

### 场景 A04：零超压基线（pool×4, 完全不重叠）

**含义**：216 个 vreg 但完全不重叠，作为零溢出基准。验证无压力下分配器的基线行为。

**构建方式**：
```python
# 每个 vreg 在各自独立的 [i*2, i*2+2) 区间，完全不重叠
block = [
    inst
    for i in range(POOL * 4)
    for inst in [
        LsInstruction(i*2, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set()),
        LsInstruction(i*2+1, 'addi', [f'v{i}_2', f'v{i}', '1'],
                      defines={f'v{i}_2'}, uses={f'v{i}'}),
    ]
]
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 216 | 27 | 2 | 0 | 0 | 0 | 0 | 0 | 216 | 3153 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.50 | 1 | 0.0% | 0 | 0 | 1.00x |

---

## 二、生命期模式维度（4 场景）

**考察焦点**：区间长度分布不同对 expire 和 eviction 行为的影响

### 场景 B01：200 密集短区间（同时创建）

**含义**：400 个 vreg，每个只活 1 条指令，但全部同时在 pos=0 创建。测试极端瞬时压力。

**构建方式**：
```python
# 200 个 li 在 pos=0 + 200 个 addi 在 pos=1
# 形成 200 个 [0, 2) 和 200 个 [1, 3) 区间
block = (
    [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
     for i in range(200)] +
    [LsInstruction(1, 'addi', [f'v{i}_2', f'v{i}', '1'],
                   defines={f'v{i}_2'}, uses={f'v{i}'})
     for i in range(200)]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 400 | 27 | 213 | 0 | 187 | 187 | 187 | 0 | 587 | 11798 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.50 | 1 | 0.0% | 0 | 0 | 1.47x |

---

### 场景 B02：200 步长依赖链（单 vreg 复用）

**含义**：200 步链式依赖，只有 1 个 vreg，测试单个 vreg 长时间占用寄存器时的行为。

**构建方式**：
```python
# v0 从 pos=0 定义，一直活到 pos=199
block = (
    [LsInstruction(0, 'li', ['v0', '0'], defines={'v0'}, uses=set())] +
    [LsInstruction(i, 'addi', ['v0', 'v0', '1'], defines={'v0'}, uses={'v0'})
     for i in range(1, 200)]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 1 | 27 | 1 | 0 | 0 | 0 | 0 | 0 | 200 | 3393 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 199.00 | 199 | 0.0% | 0 | 0 | 1.00x |

---

### 场景 B03：3 长寿命 + 60 短寿命混合

**含义**：3 个长寿命 vreg 跨越整个块（200 条指令），60 个短寿命 vreg 在中间插入。测试长短区间混合时的 expire 效率。

**构建方式**：
```python
# 3 个 li（v_long1/2/3）
# + 60 对 (li v{i}, addi v{i}_2)
# + 1 条 add 使用 v_long1, v_long2
block = [LsInstruction(0, 'li', ['v_long1', '0'], defines={'v_long1'}, uses=set()),
         LsInstruction(0, 'li', ['v_long2', '1'], defines={'v_long2'}, uses=set()),
         LsInstruction(0, 'li', ['v_long3', '2'], defines={'v_long3'}, uses=set())]
for i in range(60):
    block.append(LsInstruction(1+i*2, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set()))
    block.append(LsInstruction(2+i*2, 'addi', [f'v{i}_2', f'v{i}', '1'],
                  defines={f'v{i}_2'}, uses={f'v{i}'}))
block.append(LsInstruction(200, 'add', ['v_out', 'v_long1', 'v_long2'],
              defines={'v_out'}, uses={'v_long1', 'v_long2'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 124 | 27 | 4 | 0 | 0 | 0 | 0 | 0 | 124 | 1791 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.50 | 1 | 0.0% | 0 | 0 | 1.00x |

---

### 场景 B04：交替短区间（pool×6 个）

**含义**：324 个 vreg 交错定义和使用，制造频繁 expire 和池耗尽。

**构建方式**：
```python
# 162 个 li 在 pos=0~161 + 162 个 addi 在 pos=162~323
block = [
    inst for i in range(POOL * 6)
    for inst in [
        LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set()),
        LsInstruction(i + POOL * 6, 'addi', [f'v{i}_2', f'v{i}', '1'],
                      defines={f'v{i}_2'}, uses={f'v{i}'}),
    ]
]
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 324 | 27 | 27 | 135 | 1 | 136 | 136 | 135 | 595 | 13383 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.50 | 1 | 0.0% | 0 | 0 | 1.84x |

---

## 三、复用模式维度（4 场景）

**考察焦点**：自溢 vreg 的 reload 频率和模式对代码效率的影响

### 场景 C01：单自溢 vreg 被 20 次连续使用

**含义**：制造 1 个自溢 vreg，在后续 20 条指令中连续使用。测试每次使用是否都生成独立的 lw reload。

**构建方式**：
```python
block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(POOL)]
block += [LsInstruction(0, 'li', [f'v{POOL}', '99'], defines={f'v{POOL}'}, uses=set())]
block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(POOL+1)],
           defines={'v_sum'}, uses={f'v{i}' for i in range(POOL+1)})]
for i in range(20):
    block.append(LsInstruction(2+i, 'addi', [f'v_out{i}', f'v{POOL}', str(i)],
                  defines={f'v_out{i}'}, uses={f'v{POOL}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 49 | 27 | 28 | 0 | 1 | 1 | 1 | 0 | 50 | 829 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 0.98 | 21 | 0.0% | 0 | 0 |

---

### 场景 C02：20 个自溢 vreg 各用 1 次

**含义**：20 个 vreg 各自被 evict 一次、使用一次。测试 eviction + reload 的 1:1 模式。

**构建方式**：
```python
block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(POOL + 20)]
block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(POOL + 20)],
           defines={'v_sum'}, uses={f'v{i}' for i in range(POOL + 20)})]
for i in range(20):
    block.append(LsInstruction(2+i, 'addi', [f'v_out{i}', f'v{POOL+i}', str(i)],
                  defines={f'v_out{i}'}, uses={f'v{POOL+i}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 68 | 27 | 37 | 0 | 11 | 11 | 11 | 0 | 79 | 1446 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 0.99 | 2 | 0.0% | 0 | 0 |

---

### 场景 C03：模拟循环体（30 轮 × 10 vreg）

**含义**：30 轮 × 10 个 vreg 的重复定义/使用模式，模拟循环体 unroll 后的基本块行为。

**构建方式**：
```python
block = []
for iteration in range(30):
    start = iteration * 5
    for j in range(10):
        block.append(LsInstruction(start + j, 'li', [f'v{j}', str(j)],
                      defines={f'v{j}'}, uses=set()))
    for j in range(8):
        block.append(LsInstruction(start + 10 + j, 'add', [f'v_acc{j}', f'v{j}', f'v{j+1}'],
                      defines={f'v_acc{j}'}, uses={f'v{j}', f'v{j+1}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 18 | 27 | 10 | 0 | 0 | 0 | 0 | 0 | 540 | 7379 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 26.67 | 60 | 0.0% | 0 | 0 | 1.00x |

---

### 场景 C04：自溢 vreg 相邻位置多次使用

**含义**：一个自溢 vreg 在同一指令中被用 2 次，然后连续 10 次被用。测试 _pick_scratch 是否在两个相邻位置复用同一个 scratch 寄存器。

**构建方式**：
```python
block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(POOL + 3)]
block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(POOL + 3)],
           defines={'v_sum'}, uses={f'v{i}' for i in range(POOL + 3)})]
block += [LsInstruction(2, 'add', ['v_out', f'v{POOL}', f'v{POOL}'],
           defines={'v_out'}, uses={f'v{POOL}'})]
for i in range(10):
    block.append(LsInstruction(3+i, 'add', [f'v_tmp{i}', f'v{POOL}', str(i)],
                  defines={f'v_tmp{i}'}, uses={f'v{POOL}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 42 | 27 | 29 | 0 | 2 | 2 | 2 | 0 | 44 | 733 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 0.98 | 12 | 0.0% | 0 | 0 |

---

## 四、Eviction 模式维度（4 场景）

**考察焦点**：farthest-end 策略在不同区间分布下的表现差异

### 场景 D01：全部同终点

**含义**：48 个 interval 全部 [0, 1)，farthest-end 策略退化为随机选择（所有 end 相同）。

**构建方式**：
```python
N = POOL + 20  # 47
block = (
    [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
     for i in range(N)] +
    [LsInstruction(1, 'add', ['v_out'] + [f'v{i}' for i in range(N)],
                   defines={'v_out'}, uses={f'v{i}' for i in range(N)})]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 48 | 27 | 37 | 0 | 11 | 11 | 11 | 0 | 59 | 1079 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads |
|----------|----------|------------|--------------|-------------------|
| 0.98 | 1 | 0.0% | 0 | 0 |

---

### 场景 D02：偏斜终点（前短后长）

**含义**：前半 13 个短区间（立即 expire）+ 后半 24 个长区间（活到块尾）。测试 farthest-end 是否选择错误的 victim。

**构建方式**：
```python
N = POOL + 10  # 37
# 前半: 短区间（用完后很快 expire）
block = [LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(POOL // 2)]
block += [LsInstruction(POOL // 2, 'add', [f'v_sum_early'] + [f'v{i}' for i in range(POOL // 2)],
           defines={'v_sum_early'}, uses={f'v{i}' for i in range(POOL // 2)})]
# 后半: 长区间（活到块尾）
for i in range(POOL // 2, N):
    block.append(LsInstruction(POOL // 2 + 1, 'li', [f'v{i}', str(i)],
                  defines={f'v{i}'}, uses=set()))
block.append(LsInstruction(POOL // 2 + 2, 'sub', ['v_out', f'v{POOL//2}', f'v{N-1}'],
              defines={'v_out'}, uses={f'v{POOL//2}', f'v{N-1}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 39 | 27 | 24 | 0 | 0 | 0 | 0 | 0 | 39 | 513 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.38 | 1 | 0.0% | 0 | 0 | 1.00x |

---

### 场景 D03：链式 eviction

**含义**：先占满 27 个寄存器，然后连续引入 30 个新 vreg，每个新 vreg 的区间覆盖前一个。测试 eviction 链式传递。

**构建方式**：
```python
block = [LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(POOL)]
for w in range(30):
    idx = POOL + w
    block.append(LsInstruction(POOL + w*2, 'li', [f'v{idx}', str(w)],
                  defines={f'v{idx}'}, uses=set()))
    block.append(LsInstruction(POOL + w*2 + 1, 'add', ['v_out', f'v{idx}', 'v0'],
                  defines={'v_out'}, uses={f'v{idx}', 'v0'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 58 | 27 | 3 | 0 | 0 | 0 | 0 | 0 | 87 | 1179 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 1.03 | 30 | 0.0% | 0 | 0 | 1.00x |

---

### 场景 D04：100 步螺旋交织

**含义**：100 步螺旋依赖（模运算选择 dst/src），制造复杂的区间交织模式和 eviction churn。

**构建方式**：
```python
rng = random.Random(42)
block = []
for i in range(100):
    d = f'v{(i * 3) % 100}'
    u1 = f'v{(i * 5) % 100}'
    u2 = f'v{(i * 7) % 100}'
    if d == u1 or d == u2:
        d = f'v{i}'
        u1 = f'v{(i + 1) % 100}'
        u2 = f'v{(i + 2) % 100}'
    block.append(LsInstruction(i, 'add', [d, u1, u2], defines={d}, uses={u1, u2}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 100 | 27 | 31 | 0 | 4 | 4 | 4 | 0 | 104 | 1840 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 2.00 | 6 | 0.0% | 0 | 0 | 1.04x |

---

## 五、栈压力维度（3 场景）

**考察焦点**：spill slot 分配和复用的空间效率

### 场景 E01：10 波交替溢出

**含义**：10 波，每波产生 POOL+3 个 vreg 全部重叠使用，波之间完全隔离。测试 slot 在波之间是否可以复用。

**构建方式**：
```python
block = []
for wave in range(10):
    base = wave * 30
    for i in range(POOL + 3):
        block.append(LsInstruction(wave*2, 'li', [f'v{base+i}', str(i)],
                      defines={f'v{base+i}'}, uses=set()))
    all_v = [f'v{base+i}' for i in range(POOL + 3)]
    block.append(LsInstruction(wave*2+1, 'add', [f'v_sum{wave}'] + all_v,
                  defines={f'v_sum{wave}'}, uses=set(all_v)))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 310 | 27 | 31 | 0 | 4 | 4 | 4 | 0 | 314 | 4946 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.97 | 1 | 0.0% | 0 | 0 | 1.01x |

---

### 场景 E02：级联溢出（50 步 mul 链）

**含义**：50 步 mul 链，每个新 vreg 依赖前两个（v{idx} = v{idx-1} * v{idx-2}）。测试长依赖链是否导致级联 eviction。

**构建方式**：
```python
block = [LsInstruction(i, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(POOL)]
for i in range(50):
    idx = POOL + i
    block.append(LsInstruction(POOL + i, 'mul', [f'v{idx}', f'v{idx-1}', f'v{idx-2}'],
                  defines={f'v{idx}'}, uses={f'v{idx-1}', f'v{idx-2}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 77 | 27 | 3 | 0 | 0 | 0 | 0 | 0 | 77 | 1174 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 1.30 | 2 | 0.0% | 0 | 0 | 1.00x |

---

### 场景 E03：100 vreg 完全同时存活（零复用）

**含义**：100 个 vreg 完全同时存活（区间全部[0,1)重叠），所有 slot 完全不可复用。测试 slot 分配在最坏情况下的极限。

**构建方式**：
```python
block = (
    [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
     for i in range(100)] +
    [LsInstruction(1, 'add', ['v_out'] + [f'v{i}' for i in range(100)],
                   defines={'v_out'}, uses={f'v{i}' for i in range(100)})]
)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 101 | 27 | 64 | 0 | 37 | 37 | 37 | 0 | 138 | 2730 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.99 | 1 | 0.0% | 0 | 0 | 1.37x |

---

## 六、代码膨胀维度（4 场景）

**考察焦点**：spill code 对最终汇编大小的影响

### 场景 F01：最大溢出路径（100 vreg 各用 5 次）

**含义**：100 个 vreg 各被使用 5 次，每个自溢 vreg 都有多次 reload。测试最坏情况下的 sw/lw 生成量。

**构建方式**：
```python
block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(100)]
for i in range(100):
    for j in range(5):
        block.append(LsInstruction(1 + i*5 + j, 'add', [f'v_tmp{i}_{j}', f'v{i}', str(j)],
                      defines={f'v_tmp{i}_{j}'}, uses={f'v{i}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 600 | 27 | 59 | 10 | 32 | 42 | 42 | 50 | 692 | 12133 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 0.83 | 5 | 0.0% | 0 | 0 | 1.15x |

---

### 场景 F02：30 vreg 各用 10 次（膨胀比测量）

**含义**：30 个 vreg 各被使用 10 次，精确测量 asm 行数对原始指令数的膨胀比。

**构建方式**：
```python
ratio = 10
block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(30)]
block += [LsInstruction(1, 'add', ['v_sum'] + [f'v{i}' for i in range(30)],
           defines={'v_sum'}, uses={f'v{i}' for i in range(30)})]
for v in range(30):
    for u in range(ratio):
        block.append(LsInstruction(2 + v*ratio + u, 'addi', [f'v_out{v}_{u}', f'v{v}', str(u)],
                      defines={f'v_out{v}_{u}'}, uses={f'v{v}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 331 | 27 | 29 | 0 | 2 | 2 | 2 | 0 | 333 | 5691 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 1.00 | 11 | 0.0% | 0 | 0 | 1.006x |

---

### 场景 F03：60 链式密集全重叠（dense_60）

**含义**：60 个 vreg，链式 add 依赖。每个 vreg 被反复定义，模拟密集运算中的寄存器分配行为。

**构建方式**：
```python
N = 60
block = [LsInstruction(0, 'li', [f'v{i}', str(i)], defines={f'v{i}'}, uses=set())
         for i in range(N)]
for i in range(N - 1):
    block.append(LsInstruction(1 + i, 'add', [f'v{i}', f'v{i}', f'v{i+1}'],
                  defines={f'v{i}'}, uses={f'v{i}', f'v{i+1}'}))
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 60 | 27 | 42 | 3 | 15 | 18 | 35 | 6 | 160 | 2963 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 1.97 | 2 | 0.0% | 0 | 0 | 1.34x |

---

### 场景 F04：500 条指令完全随机块

**含义**：500 条随机指令（80% 使用已有 vreg，20% 创建新 vreg），模拟真实复杂函数的随机行为。

**构建方式**：
```python
rng = random.Random(12345)
block = []
active_vregs = set()
for i in range(500):
    if rng.random() < 0.2 or not active_vregs:
        v = f'v{rng.randint(0, 299)}'
        block.append(LsInstruction(i, 'li', [v, str(rng.randint(0, 99))],
                      defines={v}, uses=set()))
        active_vregs.add(v)
    else:
        src1 = rng.choice(list(active_vregs))
        src2 = rng.choice(list(active_vregs))
        dst = f'v{rng.randint(0, 299)}'
        block.append(LsInstruction(i, 'add', [dst, src1, src2],
                      defines={dst}, uses={src1, src2}))
        active_vregs.add(dst)
    if active_vregs and rng.random() < 0.2:
        victim = rng.choice(list(active_vregs))
        active_vregs.discard(victim)
```

**测试数据**：

| vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes |
|-------|------|------|------|-------|------|----|----|-------|-------|
| 246 | 27 | 70 | 16 | 43 | 59 | 137 | 68 | 705 | 14398 |

| avg_uses | max_uses | slot_reuse | redundant_sw | redundant_reloads | 膨胀比 |
|----------|----------|------------|--------------|-------------------|--------|
| 3.13 | 13 | 0.0% | 0 | 0 | 1.41x |

---

## 汇总表

| 场景 | vregs | pool | peak | self | evict | slot | sw | lw | lines | bytes | avg_uses | slot_reuse | 膨胀比 |
|------|-------|------|------|------|-------|------|----|----|-------|-------|----------|------------|--------|
| **活跃度维度** | | | | | | | | | | | | | |
| A01 轻度超压 | 33 | 27 | 30 | 0 | 3 | 3 | 3 | 0 | 36 | 600 | 0.97 | 0.0% | 1.09x |
| A02 中度超压 | 55 | 27 | 41 | 0 | 14 | 14 | 14 | 0 | 69 | 1280 | 0.98 | 0.0% | 1.25x |
| A03 重度超压 | 201 | 27 | 114 | 0 | 87 | 87 | 87 | 0 | 288 | 6046 | 1.00 | 0.0% | 1.43x |
| A04 零超压基线 | 216 | 27 | 2 | 0 | 0 | 0 | 0 | 0 | 216 | 3153 | 0.50 | - | 1.00x |
| **生命期维度** | | | | | | | | | | | | | |
| B01 密集短区间 | 400 | 27 | 213 | 0 | 187 | 187 | 187 | 0 | 587 | 11798 | 0.50 | 0.0% | 1.47x |
| B02 长依赖链 | 1 | 27 | 1 | 0 | 0 | 0 | 0 | 0 | 200 | 3393 | 199.00 | - | 1.00x |
| B03 混合生命期 | 124 | 27 | 4 | 0 | 0 | 0 | 0 | 0 | 124 | 1791 | 0.50 | - | 1.00x |
| B04 交替短区间 | 324 | 27 | 27 | 135 | 1 | 136 | 136 | 135 | 595 | 13383 | 0.50 | 0.0% | 1.84x |
| **复用模式** | | | | | | | | | | | | | |
| C01 单vreg连续20次 | 49 | 27 | 28 | 0 | 1 | 1 | 1 | 0 | 50 | 829 | 0.98 | 0.0% | 1.02x |
| C02 20vreg各1次 | 68 | 27 | 37 | 0 | 11 | 11 | 11 | 0 | 79 | 1446 | 0.99 | 0.0% | 1.16x |
| C03 模拟循环体 | 18 | 27 | 10 | 0 | 0 | 0 | 0 | 0 | 540 | 7379 | 26.67 | - | 1.00x |
| C04 相邻多次使用 | 42 | 27 | 29 | 0 | 2 | 2 | 2 | 0 | 44 | 733 | 0.98 | 0.0% | 1.05x |
| **Eviction 维度** | | | | | | | | | | | | | |
| D01 全部同终点 | 48 | 27 | 37 | 0 | 11 | 11 | 11 | 0 | 59 | 1079 | 0.98 | 0.0% | 1.23x |
| D02 偏斜终点 | 39 | 27 | 24 | 0 | 0 | 0 | 0 | 0 | 39 | 513 | 0.38 | - | 1.00x |
| D03 链式eviction | 58 | 27 | 3 | 0 | 0 | 0 | 0 | 0 | 87 | 1179 | 1.03 | - | 1.00x |
| D04 螺旋交织 | 100 | 27 | 31 | 0 | 4 | 4 | 4 | 0 | 104 | 1840 | 2.00 | 0.0% | 1.04x |
| **栈压力维度** | | | | | | | | | | | | | |
| E01 10波交替 | 310 | 27 | 31 | 0 | 4 | 4 | 4 | 0 | 314 | 4946 | 0.97 | 0.0% | 1.01x |
| E02 级联溢出 | 77 | 27 | 3 | 0 | 0 | 0 | 0 | 0 | 77 | 1174 | 1.30 | - | 1.00x |
| E03 零slot复用 | 101 | 27 | 64 | 0 | 37 | 37 | 37 | 0 | 138 | 2730 | 0.99 | 0.0% | 1.37x |
| **代码膨胀维度** | | | | | | | | | | | | | |
| F01 最大溢出路径 | 600 | 27 | 59 | 10 | 32 | 42 | 42 | 50 | 692 | 12133 | 0.83 | 0.0% | 1.15x |
| F02 膨胀比测量 | 331 | 27 | 29 | 0 | 2 | 2 | 2 | 0 | 333 | 5691 | 1.00 | 0.0% | 1.006x |
| F03 dense_60链式 | 60 | 27 | 42 | 3 | 15 | 18 | 35 | 6 | 160 | 2963 | 1.97 | 0.0% | 1.34x |
| F04 随机块500 | 246 | 27 | 70 | 16 | 43 | 59 | 137 | 68 | 705 | 14398 | 3.13 | 0.0% | 1.41x |

---

报告生成时间：当前会话。测试框架和所有场景构建函数在 `/tmp/topic17_bottleneck_scenarios.py`，可随时复现。