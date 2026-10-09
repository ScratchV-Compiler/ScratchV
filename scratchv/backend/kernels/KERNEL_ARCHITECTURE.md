# 算子内核框架 · 架构文档

> 本目录是「把形状在编译期已知（或部分已知）的热点算子直接降低成 RISC-V 汇编」的
> 内核框架。它与 `scratchv/backend/instruction_select.py` 的通用逐指令下降**并存**，
> 不合并——理由见 §2.3。
>
> **参考实现**：本分支（`dev/scratch`）从干净的 `origin/main` 起步。比赛1 的三份
> int32 内核（`q16_matmul.py` / `vector_ops.py` / `_scaffold.py`）与 72 项真机对照
> 测试只存在于本地 `main` 的 7 个未推送提交里，**是只读参考**，不随本分支演进。

---

## 0. 状态图例与当前进度

本文档描述的是**目标架构**，不是已建成的东西。每个 pass 都带状态标记：

| 标记 | 含义 |
|---|---|
| `[P0]` | **本阶段实现**。当前开发目标是内测比赛2（fp32 三题），只做到「明显优于 `-O0` 参考解」的水平 |
| `[P1]` | **只留接口与契约**，不实现。函数签名、前置条件、验收判据写清楚，函数体留空 |
| `[P2]` | **明确不做**，并写明为什么不做（多半是因为当前成本模型下没有收益） |

> ⚠️ 本阶段的验收判据是**正确性 + 明显优于 `-O0`**，**不是排行榜名次**。
> 两者不等价，细节见 `OPTIMIZATION.md` §1。

---

## 1. 要解决的三类耦合

现有（比赛1）的写法把三件事焊在同一个 `emit_*_kernel` 函数体里，代价是：

| 现象 | 根因 | 本次要拆出的轴 |
|---|---|---|
| 加一道 fp32 题要重写 kernel | 叶子算术（`lw`/`mulh`/`sw`）与循环结构写在同一个函数里 | **② 数值类型** |
| 换指令集要改助记符 | 助记符硬编码在发射逻辑里 | **③ 指令集/目标** |
| 赛题隐藏逐点规模后框架失效 | 「N 是编译期常量」是隐含假设，不是显式输入 | **① 形状知识 + ④ 覆盖策略** |

---

## 2. 四条正交轴

```
                         ┌────────────────────────────────────────────┐
   ① 形状知识 ───────────┤ 编译期知道多少形状？                        │
      ShapeKnowledge     │ Exact | Ladder | Range | Predicate | Unknown│
                         │ ★ 赛题自 2026-10-07 起只公布区间            │
                         └────────────────────────────────────────────┘
                         ┌────────────────────────────────────────────┐
   ② 数值类型 ───────────┤ 算的是什么？                                │
      DtypePolicy        │ Q16.16 / f32 / …：算术规则 + 寄存器银行      │
                         │   + 数值判据（逐位 vs 容差）                │
                         └────────────────────────────────────────────┘
                         ┌────────────────────────────────────────────┐
   ③ 指令集/目标 ────────┤ 在哪台机器上算？                            │
      TargetDesc         │ rv32im / rv32imf：march + 助记符表           │
                         │   + 寄存器银行容量                          │
                         └────────────────────────────────────────────┘
                         ┌────────────────────────────────────────────┐
   ④ 覆盖策略 ───────────┤ 由 ① 展开成「发哪些特化 + 必留通用回退」    │
      CoveragePlan       │ Cover | Residue | GenericPlus               │
                         └────────────────────────────────────────────┘
```

**四条轴互相正交**：改 ② 不动 ③，改 ③ 不动 ①④。这是本架构唯一的核心主张。

### 2.1 形状知识（①）

```python
ShapeKnowledge = (
    Exact({4, 8, 12, …})          # 逐点已知（比赛1 的处境）
  | Ladder(base, step, hi)        # 形如 base, base+step, … ≤ hi
  | Range(lo, hi)                 # 区间内任意整数（赛题公布的就是这个）
  | Predicate(f)                  # 任意谓词，如「N 是 2 的幂」
  | Runtime(header_layout)        # 编译期不知、运行时可从输入头读出
  | Unknown                       # 什么都不知道
)
```

**为什么必须有这一层**：`riscv_problems.size_span()` 是公开的（题面/帮助页/提交页都只给
区间），而 `data_point_sizes` 是内部的——2026-10-07 起平台刻意不再逐点展示规模。所以
「知道 N」不再是可以写死的默认前提，而是一个**输入参数**。

### 2.2 数值类型（②）

一个 `DtypePolicy` 至少声明四件事：

| 成员 | 作用 | 例子 |
|---|---|---|
| `alu` | 逐元素算子 → 目标算术 | f32：`add→FADD.S`、`mul→FMUL.S`；Q16：`mul→` 见下 |
| `load` / `store` | 访存助记符 | f32：`FLW`/`FSW`；int32：`LW`/`SW` |
| `bank` | 数据寄存器银行 | f32：`fp`（独立银行）；int32：`int` |
| `contract` | **数值判据** | f32：`Tolerance(rtol=1e-4, atol=1e-3)`；Q16：`Exact` |

**关键设计决定：代数技巧属于 dtype，不属于循环。** `slli` + `mulh` 替 `mul` + `srai`
是 **Q16.16 在值域 `|a|,|b| ≤ 2¹⁵` 下的**代数恒等式，不是一条通用规则。它应当作为
`DtypePolicy` 的一条**带前置条件的降低规则**登记：

```
mul(Q16) → mulh(slli(a,16), b)     前置: |a| ≤ 2^15 ∧ |b| ≤ 2^15
mul(f32) → fmul.s                  前置: 无
```

前置条件要能被检查（照 `scratchv/optimizer/hoist_safety.py` 的样式），不能只写在注释里。

### 2.3 指令集/目标（③）

`TargetDesc` 是**唯一**可以直接写助记符和寄存器名的地方：

```python
'rv32imf': TargetDesc(
    march='rv32imf',
    banks={'int': INT_POOL,            # 26 个可写整数寄存器
           'fp':  F_POOL},             # 32 个 f 寄存器，可作叶子函数的临时
    load={'int32': 'LW', 'f32': 'FLW'},
    store={'int32': 'SW', 'f32': 'FSW'},
    imm_max=2047,                      # 12 位有符号立即数上限
    line_bytes=64,                     # L1 缓存行
)
```

**为什么要独立一层**：fp32 引入了一个**独立的寄存器银行**。int32 下分块约束是
一条式子 `acc + hold + non_data ≤ 26`；f32 下 acc/hold 在 f 池、指针在 x 池，约束
**分裂成两条独立的**。这意味着 f32 的合法分块集合与 int32 不同——容量必须从
`TargetDesc` 取，不许硬编码 26。

### 2.4 覆盖策略（④）

这是 v1 方案缺的一层，也是「看不到规模」唯一的落点：

```
CoveragePlan(shape_knowledge, kernel, dtype, target)
    → specializations : [Exact(n) | Residue(u, r) | …]
      generic         : GenericVariant        # 必须存在
      dispatch        : 精确表 | 密集区间表 | 直通
      expected_cost   : 由 CostModel 的 CoverageRisk 项估
      risk            : 未命中概率上界
```

四档策略的取舍（详细代价见 `OPTIMIZATION.md` §5）：

| 档 | 适用 | 代价来源 |
|---|---|---|
| `Cover(区间)` | 区间窄（如 matmul `[4,64]` 只有 61 个整数） | 分发表读 1 条 cache 行；未执行的代码**不收费** |
| `Residue(u)` | 区间宽（如 `[64,4096]`） | 偏移仍是立即数，只多了运行时圈数与尾部处理 |
| `GenericPlus` | 兜底，永远存在 | 相对特化约 +5~10% |
| `Unknown`（现状的 `.Lgeneric`） | 兜底 | 相对特化 **2.0×~3.3×** |

> **「未执行的代码不收费」是本成本模型的一条硬性质**：`cost = 执行到的指令数 +
> 15 × 执行期间遇到的 L1 未命中`。i_miss 只统计**实际取到的指令行**。因此覆盖集
> 变大只在运行时付出「分发表那 1 条 d_miss」和「段布局平移需重扫」的代价。

---

## 3. 分层与 IR

```
   [问题描述]  kernel=matmul/add/reducesum, dtype, target, shape_knowledge
        │
  SIR  形状层    ShapeKnowledge + CoveragePlan   ── 决定「发哪些单元」
        │
  LIR  循环层    结构化循环巢；叶子是**类型化算子**
        │        Load(mem, off, ty) / BinOp(op, ty, a, b) / Store(mem, off, ty, v)
        │        地址在 IR 里显式表达为 Addr(base, imm) 或 Addr(reg, imm)
        │        ⇒ 「立即数 vs 指针」是可改写的，强度削减才能成为 pass
        │
  MIR  机器层    dtype 已降低、助记符已定、虚拟寄存器
        │
  ASM  物理层    物理寄存器（按银行分配）
        │
   [自包含 .s]
```

**只有三层，不是 LLVM 那套。** 理由：本架构的价值来自「形状特化 + 手工布局」，
不需要通用优化器；IR 层的唯一职责是给 ②③ 两条轴一个与问题形状无关的落点。

---

## 4. Pass 清单

### A 形状与选型层（机器无关、dtype 无关）

| # | pass | 输入 → 输出 | 前置条件 | 状态 |
|---|---|---|---|---|
| A1 | `ShapeBind` | 题面/区间 → `ShapeKnowledge` | — | `[P0]` |
| A2 | `CoveragePlan` | 形状知识 + kernel → 特化键集合 + 回退 + 分发形态 | **必须存在一个对任意输入正确的回退** | `[P1]` |
| A3 | `LegalityCheck` | 特化单元 → 合法/回退 | 银行容量 ∧ 立即数范围 ∧ workspace 预算 | `[P0]`（简化版） |
| A4 | `PlanSearch` | 单元 → Plan（分块/展开/形态） | 候选必须由 target+dtype 过滤 | `[P1]` |
| A5 | `PlanPin` | 实测表覆盖选型 | 表项必须带数据点校验 | `[P1]` |

### B 循环结构层（机器无关、dtype 无关）

| # | pass | 输入 → 输出 | 前置条件 | 状态 |
|---|---|---|---|---|
| B1 | `LoopBuild` | Plan → LIR | 形状谓词成立 | `[P0]` |
| B2 | `LoopFormLower` | 整块展开 / 真循环 / 部分展开 | 展开体不得超出分支可达范围 | `[P0]`（只做「固定小因子展开」） |
| B3 | `AddrStrengthReduce` | 索引 → 指针归纳；立即数偏移 vs 独立指针 | 偏移需在 `imm_max` 内 | `[P1]` |
| B4 | `InvariantHoist` | k 不变式提到分块循环外 | **必须带前置条件检查** | `[P1]` |
| B5 | `CacheBlocking` | 循环巢 → 分层分块 | 工作集须落入 L1 | `[P1]` **实测判定不需要**：分块后未命中 773 ≈ 强制值 768（`DEVELOPMENT.md` §6.4） |
| B6 | `DispatchLower` | 特化集合 → 分发代码 | 表外必落 generic | `[P0]`（只做比较链/直通） |

### C 数值类型层（dtype 相关、ISA 无关）

| # | pass | 输入 → 输出 | 前置条件 | 状态 |
|---|---|---|---|---|
| C1 | `DtypeLower` | 类型化算子 → 目标算术 | 每条规则带前置条件 | `[P0]` |
| C2 | `AccumLower` | 归约形态 | — | `[P0]` |
| C3 | `ResidueLower` | `Residue(u,r)` → 运行时圈数的循环体 | `u` 整除约束 | `[P1]` |

### D 机器层（ISA 相关）

| # | pass | 输入 → 输出 | 前置条件 | 状态 |
|---|---|---|---|---|
| D1 | `InstSelect` | MIR → 助记符（表驱动） | 目标支持该 (op, dtype) | `[P0]` |
| D2 | `BankRegAlloc` | 虚拟寄存器 → 物理寄存器 | 按银行容量，**不硬编码 26** | `[P1]`（`[P0]` 用手工固定寄存器） |
| D3 | `BranchRelax` | 近距离分支 → 近跳 + 远跳 | 循环体 > ±4KB | `[P1]` |
| D4 | `SectionLayout` | `RODATA_PAD` 对齐扫描 | **换 wrapper/工具链必须重扫** | `[P1]` |
| D5 | `CompressedEncoding` | `.option rvc` | **平台 ISA 闸门会判整题 0 分** | `[P2]`（见 §6.3） |

### E 明确不做

| 项 | 为什么不做 |
|---|---|
| 指令调度 | **成本口径不含周期**（只数指令数与未命中）。调度不改变任何一个，收益恒为 0 |
| 周期估算 / `cycle_estimator` 路线 | 同上，与本成本模型正交 |
| 把内核并入通用后端 | 见下 §5 |

---

## 5. 为什么**不**并入通用后端

| 证据 | 位置 |
|---|---|
| 通用后端遇到 tensor buffer 直接 `raise`（它服务逐标量 ONNX/DSL 图） | `scratchv/compiler.py:527-531` |
| 线性扫描分配器**只接受整数寄存器**，喂 `f0` 直接抛异常 | `scratchv/backend/regalloc_linear.py:212-218` |
| 代价口径是**周期数**，与 `指令数 + 15×未命中` 正交 | `scratchv/backend/cycle_estimator.py` |

并入会同时丢掉「形状特化」和「手工布局」这两个真正产生分数的性质。正确做法是
**并列一条 kernel pipeline，只共享分析类工具**（`pass_manager` 的框架、
`machine_types` 的类型、liveness/usedef）。

---

## 6. 关键不变量（改动前必须核对）

**I1 · 必须存在对任意输入正确的回退。**
包括 `N = 0`、负数、非整除、超出区间。回退路径存在的第一理由是「不能跑飞」。

**I2 · 回退路径必须有质量门限。**
这条是新加的。原先的假设是「回退永远不跑」，但赛题隐藏规模之后**回退可能变成主路径**。
`Unknown` 下 2.0×~3.3× 的差距不是可接受的兜底。

**I3 · 不得产出压缩指令。**
平台 `riscv_runner.isa_violation()` 扫可执行段，**任何非 4 字节指令**（含
`.option rvc` 与手写 `.2byte`）⇒ 判 `isa_violation`，**整题 0 分**（不是扣该点分）。

**I4 · 成本口径与平台一致。**
`cost = 指令数 + MISS_PENALTY(15) × (d_miss + i_miss)`。常数（`FIXED_INSTR`、
`FIXED_MISS`、`LINE_BYTES`）是把空内核放进平台链路**量出来的**，不是拍的。

**I5 · 数值判据由 dtype 决定。**
`Exact`（int32/Q16）⇒ 逐位相等；`Tolerance`（f32）⇒ 容差比对，**且必须同时钉住
cost**，否则「结果对但慢一倍」会被判通过。

**I6 · 寄存器银行容量从 `TargetDesc` 取。**
不许把 `26` 写进任何分块约束。f32 下约束是两条独立的。

**I7 · 段布局（`RODATA_PAD`）是绑死在 wrapper 与工具链上的量。**
换 wrapper、换 clang 版本、换 `march` 都必须重扫。它是纯布局效应，与指令序列无关。

---

## 7. 与平台的关系

**平台侧一行代码都不用改。** `.s` 是唯一接口：

| 契约 | 值 |
|---|---|
| 入口符号 | `cnn_entry`（全局） |
| 入参 | `a0` = 输入首址，`a1` = 输出首址，`a2` = 规模标量 N |
| 内存布局 | `guard(256) \| workspace \| guard(256) \| output \| guard(256)`，`sp` 指向 workspace 顶端 |
| 越界判定 | 任一 guard 区非零 ⇒ 该数据点失败 |
| 编译 | `clang --target=riscv32-linux-gnu -march=<rv32im\|rv32imf> -mabi=ilp32 -nostdlib -static -fuse-ld=lld -Wl,--no-relax` |
| 执行 | `qemu-riscv32`（沙箱内），缓存插件给出指令数与 L1 统计 |
| 计分 | 逐点 `分值 × min(1, 全场最优cost / 本队cost)` |

完整链路与硬事实见 `USAGE.md` §2。

---

## 8. 目录结构

### 8.1 实际建成的（**以这一节为准**）

```
scratchv/backend/kernels/
  KERNEL_ARCHITECTURE.md  USAGE.md  DEVELOPMENT.md  OPTIMIZATION.md
  __init__.py        包声明（必须有，否则 python -m 起不来）
  target.py          TargetDesc + TARGETS（rv32im / rv32imf）
  dtypes.py          DtypePolicy + POLICIES + mac_instrs + zero_acc
  loopgen.py         prologue / epilogue / unrolled_body
  bodies/
    __init__.py      BODIES + PROBLEMS（题册）
    add.py           两档：展开 1 / 展开 u + 余数尾循环
    reducesum.py     同上
    matmul.py        _generic / _blocked + blocked_regs_needed + 派发 build
  pipeline.py        build_program + UNROLL / BLOCKING（两张实测表）
  __main__.py        CLI
```

### 8.2 设计里计划过、但**没有**单独建文件的

| 设计里的名字 | 实际落在哪 |
|---|---|
| `LegalityCheck`（分块容量检查） | `bodies/matmul.blocked_regs_needed()` |
| `LoopFormLower`（展开 vs 真循环） | `bodies/{add,reducesum}.build(unroll=...)` |
| `InstSelect`（助记符表） | `target.py` 的 `load`/`store` + `dtypes.py` 的 `mac` |
| `DtypeLower` | `dtypes.py` 的 `mac` 模板 |
| **`CostModel` 对象** | **没有**。代价只出现在注释与实测表里，**没有可调用的成本模型** |
| `plan.py` / `cost.py` / `coverage.py` / `kir.py` / `isel.py` / `layout.py` | **一个都没建** |

**为什么没按设计分文件**：落地时发现这一阶段的量还不到需要拆那么多文件的程度——
十条 pass 里有六条各自只有几行到几十行，彼此之间也没有可复用的接口。
**拆成一堆空文件反而增加理解成本。** 等哪一条长到需要独立测试时再拆。

### 8.3 没有兑现的设计承诺

> ⚠️ **没有实现 `pass_interface.CompilerPass`。** 上面第 4 节的 pass 清单是按那个
> 契约设计的，实际落地用的是普通函数（`bodies/*.build()`、`pipeline.build_program()`）。
>
> **所以"复用 `pass_manager` 的报告 / 计时 / 变更计数、`--json` / `--markdown`
> 与其他工具一致"这件事，目前没有兑现。** 第 4 节表里的状态标记读作
> 「这个能力在代码里有对应物」，**不是**「这个 pass 作为 pass 对象存在」。

---

## 9. 分期（含实际进度）

| 阶段 | 内容 | 放行条件 | 状态 |
|---|---|---|---|
| **S0** | `target.py` + `dtypes.py` + `bodies/*` + `pipeline.py`：打通内测比赛2 三题，只到 `-O0` 水平 | 正确性通过；cost 明显优于 `-O0` 参考解；body 里不含任何硬编码助记符 | ✅ **已完成**（六题 10/10） |
| **S0.5** | `LoopFormLower`（循环展开 + 余数尾循环） | 区间内每个 N 都正确（含非整除）；总 cost 下降 | ✅ **已完成**（u=32，见 `DEVELOPMENT.md` §5.1） |
| **S0.6** | `LegalityCheck` + 寄存器分块（matmul） | 放不下时抛异常而不是静默出错；分块路径正确 | ✅ **已完成**（(4,4)，8.17→2.98 条/MAC） |
| **S1** | `ShapeKnowledge` + `CoveragePlan` + 分发 | `Range` 下**区间里每个整数**都正确，且 cost 不退化 | ❌ **未做**（见 `DEVELOPMENT.md` §6.2） |
| **S2** | `CacheBlocking` | 逐点 cost 逼近榜首 | ⏭️ **实测判定不需要**——分块后未命中仍是 773 ≈ 强制值 768，没有洞可补（§6.4） |
| **S3** | `kir.py` + 各 pass 真对象化 + `BankRegAlloc` | 现有 `.s` 成本逐点不变 | ❌ 未做 |
| **S4** | `Runtime` 形状（规模写在输入张量头里的题） | 该形态的形状契约测试通过 | ❌ 未做 |

**下一步最有价值的不是 S3，是 S1。** 理由：现在没有分发层，所以 `UNROLL` 只能取
一个全局折中值（32），而实测小 N（64/128）偏好 8（`DEVELOPMENT.md` §5.1）。
S1 能把这部分收益拿回来；S3 只是重构，不产生收益。
