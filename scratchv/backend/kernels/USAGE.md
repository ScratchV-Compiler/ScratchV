# 算子内核框架 · 使用文档

> 这份文档面向**后续接手的人**：你要加一道题、加一个数值类型、或者只是把内测比赛2
> 的三道题交出去。读完 §2 你就知道平台在考什么，读完 §4~§6 你就能自己动手。
>
> 架构与分层见 `ARCHITECTURE.md`；性能从哪来见 `OPTIMIZATION.md`。

> ### ⚠️ 当前状态：文档描述的是**目标**，代码尚未建成
>
> 本分支（`dev/scratch`）从干净的 `origin/main` 起步，`scratchv/backend/kernels/`
> 下**目前只有这三份文档**。§3 的 `python -m scratchv.backend.kernels …` 命令、
> §4~§6 的 `bodies/` / `target.py` / `dtypes.py` **都还没实现**——它们是本阶段
> （内测比赛2）要交付的东西。
>
> 已经能用的只有 §2 的平台契约与 §8 的本地复测工具（`measure.py` / `eval_local.py`
> 在 `/root/workspace/riscv_matmul/` 下，与框架无关）。
>
> 比赛1 的三份可运行内核（`q16_matmul.py` / `vector_ops.py` / `_scaffold.py`）与本
> `README.md` 在**本地 `main` 分支**的 7 个未推送提交里，是只读参考。

---

## 1. 这套东西是干什么的

把「形状在编译期已知、语义固定」的热点算子，**直接生成 RISC-V 汇编**，产出物是
一个自包含的 `.s` 文件，只定义 `cnn_entry`，不改动平台任何代码。

它不是通用编译器。它的全部优势来自两件事，任何改动都不能把这两条弄丢：

1. **形状特化**——尽可能把规模变成立即数；
2. **手工控制布局**——段对齐、寄存器用法、循环形态都自己说了算。

---

## 2. 平台侧硬契约（先读这一节）

这一节是**事实清单**，每条都对应平台源码里的位置。写内核前必须全部知道。

### 2.1 入口与 ABI

| 项 | 值 |
|---|---|
| 入口符号 | `cnn_entry`，必须是全局符号 |
| `a0` | 输入张量首址 |
| `a1` | 输出张量首址 |
| `a2` | **规模标量 N**（matmul 是阶数；add/reducesum/fwht 是向量长度；winograd/spmm 另有输入头） |
| 返回 | 返回即 dump 内存。**不需要**自己写输出 |

**选手必须写尺寸无关的代码**——N 每次都是通过 `a2` 传进来的。

### 2.2 内存布局

```
guard_lo(256) | workspace(W) | guard_mid(256) | output(O) | guard_hi(256)
```

- `W = max(1024, 8·N)`（自描述题按输入元素数给足）
- `O = 输出元素数 × 4`
- `sp` 指向 **workspace 顶端**（栈向下生长落在 workspace 内）
- **任一 guard 区非零 ⇒ 越界写 ⇒ 该数据点判失败**

### 2.3 编译

```
clang --target=riscv32-linux-gnu -march=<rv32im|rv32imf> -mabi=ilp32 \
      -nostdlib -static -fuse-ld=lld -Wl,--no-relax \
      wrapper.s player.s -o execute.elf
```

`-march` 与 `--no-relax` 是**锁死的**（否则指令数随工具链抖动、跨队不可比）。
wrapper 由平台生成，自己写死了 `.option norvc` / `.option norelax`。

### 2.4 ⚠️ ISA 闸门：压缩指令 = 整题 0 分

平台会扫可执行段，**任何非 4 字节指令**都判 `isa_violation`：

```
提交包含 rv32im 之外的指令：有 302 条指令不是 4 字节（压缩扩展 c.*）。
```

判据按**指令长度**而不是助记符（本题工具链不解码 M 扩展，`mulh` 也显示
`<unknown>`），所以 `.option rvc` 与手写 `.2byte` 两种写法**都会被抓到**。

**处罚力度**：`isa_violation` 是**整题 0 分**（`evaluator` 直接 `return 0.0`），
不是只丢该数据点的分。

> 例外：本框架**故意保留**一个压缩编码 pass 的接口（`ARCHITECTURE.md` D5），
> 但默认关闭，且标注为 `[P2]` 不做。要用的人请先想清楚这是规则问题。

### 2.5 成本口径

```
cost = 执行到的指令数 + 15 × (d_miss + i_miss)
```

- `15` = `config.MISS_PENALTY`（可用 `PLATFORM_MISS_PENALTY` 环境变量覆盖）
- L1 参数：指令与数据各 **32KB / 4 路 / 64 字节行**
- 指令数与未命中由 **qemu 缓存插件**在一次运行里同时给出
- **未执行的代码不收费**——`i_miss` 只统计实际取到的指令行

### 2.6 ⚠️ 计分：基准是「全场最优」，不是参考解

这一条最容易误解，务必读两遍：

```
逐点得分 = points_max × min(1, 全场最优cost ÷ 本队cost)
全场最优 = 所有参赛队伍在该数据点上**做对**的最小 cost
```

三条推论：

1. **`data/baseline.json` 里的 `-O0` 参考解不是计分基准。** 它只是评测器存的快照
   （`evaluator._baseline_for`），真分数在**读取榜单时**用动态基准现算
   （`scoring.score_details`）。
2. **超过基准不给额外分。** 领先 2 倍和领先 0.1% 是同一个分数——优化目标是
   **逐点踩到榜首**，不是越快越好。
3. **分数会随全场变强而下降。** 先提交的队拿高分，后提交的队用同样的代码只能拿
   更低的分（这也是平台采用动态基准、而不是评测时固定分数的原因）。

**并列时按最近提交时间排序**（`standings.py`：`(-total, last_submit)`）。

### 2.7 数据点与分值

| 题 | 数据点数 | 每点分值 | 满分 |
|---|---:|---:|---:|
| add / matmul | 10 | 3 | 30 |
| **reducesum** | 10 | **4** | **40** |

（注意 reducesum 是 4 分/点，别按 3 算。）

### 2.8 规模是保密的

平台自 **2026-10-07** 起只公布**区间**（题面 / 帮助页 / 提交页都只显示
`size_span()`，如 `N ∈ [64, 4096]`），排行榜的逐点列也改用「数据点 N」**序号**。
逐个数据点的确切规模只在评测器内部使用。

> **这意味着「按精确 N 特化」不再是可依赖的手段。** 比赛1 那三份交付的跳转表键
> 就是那 10 个精确 N——那是最后一次能这么干。新框架的 `ShapeKnowledge`
> （见 `ARCHITECTURE.md` §2.1）就是为这件事准备的。

---

## 3. 快速开始

```bash
cd /root/workspace/ScratchV

# 生成内测比赛2 的三份提交文件
python -m scratchv.backend.kernels --problem add-fp32       -o /tmp/scv_add_fp32.s
python -m scratchv.backend.kernels --problem reducesum-fp32 -o /tmp/scv_reducesum_fp32.s
python -m scratchv.backend.kernels --problem matmul-fp32    -o /tmp/scv_matmul_fp32.s

# 只看选型，不发射
python -m scratchv.backend.kernels --problem matmul-fp32 --list

# 本地真机复测（用平台同一条链路）
cd /root/riscv-ai-compiler-platform
set -a && . /root/.riscv_platform_env && set +a
PLATFORM_ENABLE_SANDBOX=0 ./.venv/bin/python \
    /root/workspace/riscv_matmul/measure.py /tmp/scv_add_fp32.s add-fp32
```

**提交前必须再跑一次 `eval_local.py`**，因为它和线上完全同源（含 ISA 闸门）：

```bash
PLATFORM_ENABLE_SANDBOX=0 ./.venv/bin/python \
    /root/workspace/riscv_matmul/eval_local.py /tmp/scv_add_fp32.s add-fp32
```

---

## 4. 怎么加一道新题

**目标**：只写语义，不写汇编。

1. 在 `bodies/` 下加一个文件，声明三件事：
   - **语义**：输出怎么从输入算出来（类型化算子序列）
   - **循环形态**：一维/二维/归一化/蝶形/稀疏
   - **形状谓词**：合法形状（如 fwht 要求 N 是 2 的幂）
2. 在 `pipeline.py` 的题册里登记：`problem_id → (kernel, dtype, target, sizes)`。
   这些值**从平台的 `riscv_problems.EVAL_SPECS` 抄**，不要自己编。
3. 跑该题的**区间内逐点**正确性测试（不只是发布的那 10 个点）。
4. 跑 `measure.py` 看 cost，跑 `eval_local.py` 看是否通过 ISA 闸门。

**准入条件**：区间内每个整数都正确；`cost` 明显优于 `-O0` 参考解。

> 常见错误：直接照抄比赛1 那份 kernel 的结构（它含逐 N 跳转表）。新框架下
> **特化是由 `CoveragePlan` 决定的**，不写在 body 里。

---

## 5. 怎么加一个新数值类型（以 fp32 为例）

内测比赛2 就是这件事的实例。改动**只有两个文件**：

### 5.1 `target.py`：加 `rv32imf`

```python
'rv32imf': TargetDesc(
    march='rv32imf',
    banks={'int': INT_POOL, 'fp': F_POOL},   # ★ f0–f31 是独立银行
    load={'int32': 'LW', 'f32': 'FLW'},
    store={'int32': 'SW', 'f32': 'FSW'},
    imm_max=2047,
    line_bytes=64,
)
```

### 5.2 `dtypes.py`：加 `F32Policy`

```python
F32Policy(
    alu={'add': 'FADD.S', 'mul': 'FMUL.S', 'sub': 'FSUB.S'},
    accumulate='FADD.S',
    bank='fp',
    contract=Tolerance(rtol=1e-4, atol=1e-3),   # ★ 从平台 spec 抄，不许自己定
    reduce_rules=[],                            # ★ f32 没有 mulh 这类代数规则
)
```

`rtol` / `atol` 的唯一来源是 `riscv_problems.EVAL_SPECS[problem]['rtol'|'atol']`。

**循环结构、分块、分发、布局、代价模型——零改动。**

### 5.3 三个 fp32 专属的坑

| 坑 | 说明 |
|---|---|
| **判据变了** | 逐位相等失效。容差路径**必须同时钉住 cost**，否则「结果对但慢一倍」会被判通过 |
| **银行分裂** | 分块约束从 `acc+hold+non_data ≤ 26` 变成**两条独立的**（f 池 / x 池）。合法 tile 集合与 int32 **不同** |
| **`panel` 变体语义变了** | Q16 的面板是为省 `slli`；f32 没有移位。但面板**仍然有用**——它把 N 从 B 的寻址里消掉（步长变成 `NR*4` 常量）。所以可用变体要**按 (dtype, target) 查询** |

**顺带一个机会**：f32 matmul 的指令地板更低（`fmul.s` + `fadd.s` = 2 条/MAC，
且**完全不需要 `slli`**）。详见 `OPTIMIZATION.md` §5 入口 7。

---

## 6. 怎么加一个新指令集

在 `target.py` 加一项即可，其余全不动。需要提供的字段见 §5.1。

**验收**：跑通该 target 下的全部题；`.s` 通过 ISA 闸门（该 march 允许的指令集内）。

---

## 7. 排查手册

| 症状 | 判定 | 常见原因 |
|---|---|---|
| `compile_error` | **整题 0 分** | 汇编语法错误；用了目标不支持的指令；`.option` 写错 |
| `isa_violation` | **整题 0 分** | 产出 2 字节指令：`.option rvc` 或手写 `.2byte` |
| `invalid` + `越界写：…的保护区被破坏` | 该点失败 | 写越界。**注意 `sp` 在 workspace 顶端**，栈向下长；工作区布局算错也会命中 guard |
| `invalid` + 数值不符 | 该点失败 | 语义错（尤其取整/移位）；f32 上先看是不是判据用错了（应为容差） |
| `runtime_error` + 段错误 | 该点失败 | 非法访存；用了未对齐的地址；吃掉了不该吃的寄存器 |
| `timeout` | 该点失败 | 死循环；或**指令数超出上限**（单步计数路径下的 `truncated`） |
| 分数远低于预期 | — | 先确认是不是**重复提交没提**，再看是不是逐点崩了某几个规模 |

**吃寄存器的坑**（比赛1 的 `.s` 就依赖这条）：内核把自己当**叶子函数**用掉了
`gp` / `tp` / `s0-s11`。成立条件是：平台编译锁死 `-Wl,--no-relax`（无 gp 相对寻址）、
guest 裸机无 TLS（tp 无意义）、且调用方在 `call cnn_entry` 之后只碰 `a0/a1/a2/a7/t0`。
**三条任一变化都要重审寄存器池。**

---

## 8. ⚠️ 本地工具与线上链路的差异

| 工具 | 走的链路 | 差异 |
|---|---|---|
| `measure.py` | 手搓同一条 `riscv_runner` 链路 | **不跑 ISA 闸门**；额外打出 `i_miss`（评测器只落库 `d_miss`） |
| `eval_local.py` | 直接调 `evaluator.run_evaluation` | 与线上**完全一致**，含 ISA 闸门与容差 |
| `sweep.py` / `*sweep*.py` | 逐候选真机编译+跑+计数 | 选型用，取真实 cost 最小者 |

**`measure.py` 测出来的好数字不代表线上能过。** 提交前一律用 `eval_local.py` 复核。

---

## 9. 文档地图

| 你想知道 | 看哪 |
|---|---|
| 为什么要这么分层 / 有哪些 pass / 哪些做了哪些没做 | `ARCHITECTURE.md` |
| 怎么动手加东西 / 平台在考什么 / 报错怎么查 | 本文档 |
| 性能从哪来 / 现在差在哪 / 下一步该优化什么 | `OPTIMIZATION.md` |
