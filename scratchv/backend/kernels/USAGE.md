# 算子内核框架 · 使用文档（参考手册）

> 这份文档**查东西**用：平台有哪些硬约束、每张表有哪些字段、报错怎么查。
> **要跟着做一遍，看 `DEVELOPMENT.md`**（那是教程，从零到出题，每一步都有可运行代码和期望输出）。
>
> | 你想知道 | 看哪 |
> |---|---|
> | 为什么这么分层、哪些东西做了哪些没做 | `ARCHITECTURE.md` |
> | **怎么做一遍**（含全部代码） | **`DEVELOPMENT.md`** |
> | 平台有哪些硬约束、字段含义、报错怎么查 | **本文档** |
> | 性能从哪来、现在差在哪 | `OPTIMIZATION.md` |

**当前状态**：`scratchv/backend/kernels/` 下的 10 个 `.py` 是**已落地并可运行**的，
内测比赛2 的三题与内测比赛1 的三题共六题全部 10/10 通过平台评测器。
命令一律用 `./scratchv_env/bin/python`（仓库自带的虚拟环境；系统 `python` 没装 onnx 等依赖）。

---

## 1. 这套东西是干什么的

把「形状在编译期已知、语义固定」的热点算子，**直接生成 RISC-V 汇编**，
产出是一个自包含的 `.s`，只定义 `cnn_entry`，**不改动平台任何代码**。

它不是通用编译器。优势来自两件事，任何改动都不能把这两条弄丢：

1. **形状特化**——尽可能把规模变成立即数；
2. **手工控制布局**——段对齐、寄存器用法、循环形态都自己说了算。

产出物当前的档位（详见 `DEVELOPMENT.md` 第 5、6 章）：

| 题 | 手段 | 对 `-O0` 参考解 |
|---|---|---|
| add / reducesum | 循环展开 32 + 余数尾循环 | 4.7× / 7.6× |
| matmul | (4,4) 寄存器分块 | 9.4× |

---

## 2. 平台侧硬契约

这一节是**事实清单**，每条都对应平台源码里的位置。写内核前必须全部知道。

### 2.1 入口与 ABI

| 项 | 值 |
|---|---|
| 入口符号 | `cnn_entry`（全局） |
| `a0` | 输入张量首址 |
| `a1` | 输出张量首址 |
| `a2` | **规模标量 N**（矩阵题是阶数，向量题是长度；有的题形状不止一个标量、写在输入张量头里） |
| 返回 | 返回即 dump 内存，**不需要**自己写输出 |

**必须写尺寸无关的代码**——N 每次都通过 `a2` 传进来。

### 2.2 内存布局

```
guard_lo(256) | workspace(W) | guard_mid(256) | output(O) | guard_hi(256)
```

- `W = max(1024, 8·N)`
- `O = 输出元素数 × 4`
- `sp` 指向 **workspace 顶端**（栈向下生长落在 workspace 内）
- **任一 guard 区非零 ⇒ 越界写 ⇒ 该数据点失败**

### 2.3 编译

```
clang --target=riscv32-linux-gnu -march=<rv32im|rv32imf> -mabi=ilp32 \
      -nostdlib -static -fuse-ld=lld -Wl,--no-relax \
      wrapper.s player.s -o execute.elf
```

`-march` 与 `--no-relax` 是**锁死的**（否则指令数随工具链抖动、跨队不可比）。
wrapper 由平台生成，写死了 `.option norvc` / `.option norelax`。

### 2.4 ⚠️ ISA 闸门：压缩指令 = 整题 0 分

平台扫可执行段，**任何非 4 字节指令**都判 `isa_violation`。判据按**指令长度**而不是
助记符（本题工具链不解码 M 扩展，`mulh` 也显示 `<unknown>`），所以 `.option rvc`
与手写 `.2byte` 两种写法**都会被抓到**。

**处罚是整题 0 分**（`evaluator` 直接 `return 0.0`），不是只丢一个数据点。

所以每个 `.s` 开头必须有 `.option norvc`——`loopgen.prologue()` 已经带上。

### 2.5 成本口径

```
cost = 执行到的指令数 + 15 × (d_miss + i_miss)
```

- `15` = `config.MISS_PENALTY`（`PLATFORM_MISS_PENALTY` 可覆盖）
- L1：指令与数据各 **32KB / 4 路 / 64 字节行**
- 指令数与未命中由 **qemu 缓存插件**在一次运行里同时给出
- **未执行的代码不收费**——`i_miss` 只统计实际取到的指令行

### 2.6 ⚠️ 计分：基准是「全场最优」，不是参考解

```
逐点得分 = points_max × min(1, 全场最优cost ÷ 本队cost)
全场最优 = 所有参赛队伍在该数据点上**做对**的最小 cost
```

三条推论：

1. **`data/baseline.json` 里的 `-O0` 参考解不是计分基准**，只是评测器存的快照。
   真分数在**读取榜单时**用动态基准现算。
2. **超过基准不给额外分。** 领先 2 倍和领先 0.1% 是同一个分数——目标是
   **逐点追到榜首**，不是越快越好。
3. **分数会随全场变强而下降。** 先提交的队拿高分。

**并列时按最近提交时间排序**。

### 2.7 数据点与分值

| 题 | 数据点数 | 每点分值 | 满分 |
|---|---:|---:|---:|
| add / matmul | 10 | 3 | 30 |
| **reducesum** | 10 | **4** | **40** |

### 2.8 规模是保密的

平台自 **2026-10-07** 起只公布**区间**（题面/帮助页/提交页都只显示 `size_span()`，
如 `N ∈ [64, 4096]`），排行榜的逐点列也改用「数据点 N」**序号**。

> **「按精确 N 特化」不再是可依赖的手段。** 现在的代码正是由此停在"通用实现"
> 这一档——**没有分发层**，所以 `UNROLL` 只能取一个全局折中值。
> 详见 `DEVELOPMENT.md` §6.1/§6.2。

---

## 3. 代码地图

```
scratchv/backend/kernels/
├── __init__.py      包声明（必须有）
├── target.py        TargetDesc + TARGETS          ← 加指令集改这里
├── dtypes.py        DtypePolicy + POLICIES        ← 加数值类型改这里
├── loopgen.py       prologue / epilogue / unrolled_body
├── bodies/
│   ├── __init__.py  BODIES + PROBLEMS（题册）      ← 加新题改这里
│   ├── add.py
│   ├── reducesum.py
│   └── matmul.py    两档：_generic / _blocked
├── pipeline.py      build_program + UNROLL / BLOCKING（两张实测表）
└── __main__.py      命令行
```

**一条铁律**：**指令助记符只允许出现在 `target.py` 和 `dtypes.py`**。
`bodies/` 里写了 `flw`/`fadd.s`，就说明耦合回来了。自查：

```bash
grep -rnE '\b(flw|fsw|lw|sw|fadd\.s|fmul\.s)\b' scratchv/backend/kernels/bodies/
# 期望：无输出
```

---

## 4. 命令行

```bash
cd /root/workspace/ScratchV
./scratchv_env/bin/python -m scratchv.backend.kernels --list            # 列出所有题名
./scratchv_env/bin/python -m scratchv.backend.kernels --problem add-fp32 -o /tmp/a.s
./scratchv_env/bin/python -m scratchv.backend.kernels --problem add-fp32  # 打到屏幕
./scratchv_env/bin/python -m scratchv.backend.kernels --problem add-fp32 --unroll 8 -o /tmp/a8.s
```

| 参数 | 作用 |
|---|---|
| `--problem` | 题名，见 `--list` |
| `-o` / `--output` | 输出路径；省略则打到标准输出 |
| `--unroll` | 覆盖展开因子（2 的幂）；省略则用 `UNROLL` 表里的实测值 |
| `--list` | 列出题名与其 (body, dtype, target) |

---

## 5. 三张表长什么样（改东西前先看这里）

### 5.1 `TargetDesc`（`target.py`）——「这台机器是什么」

| 字段 | 含义 |
|---|---|
| `name` / `march` / `mabi` | 机器名、传给 clang 的 `-march` / `-mabi` |
| `int_regs` | **整数寄存器池**（不含 `a0`/`a1`/`a2`，那是平台传进来的参数） |
| `fp_regs` | **浮点寄存器池**；整数目标留空 |
| `load` / `store` | 数值类型 → 访存助记符，如 `{'f32': 'flw'}` |
| `imm_max` | 立即数上限（12 位有符号 = 2047） |
| `line_bytes` | 一级缓存行 |

现有两项：`rv32im`（`fp_regs` 为空）、`rv32imf`（`fp_regs` = f0–f31）。

### 5.2 `DtypePolicy`（`dtypes.py`）——「这种数怎么算」

| 字段 | 含义 |
|---|---|
| `load` / `store` / `add` | 引存与加法助记符 |
| `acc` / `tmp1` / `tmp2` / `prod` | **数据寄存器名**。`prod` 是乘积落点，**必须与两个操作数都不同** |
| `mac` | **一次乘加的指令序列模板**（`{acc}`/`{t1}`/`{t2}`/`{p}` 由 body 填） |
| `zero` | 累加器置零模板（`{acc}` = 累加器，`{r}` = 调用方给的空闲整数寄存器） |
| `bank` | 数据寄存器取自哪个银行：`'fp'` 或 `'int'` |
| `comparison` | `'exact'`（逐位）/ `'tolerance'`（容差） |

**`mac` 为什么不只是一个助记符**：一次乘加在两种数值类型下**形状不同**——
f32 是 `fmul.s` + `fadd.s`（2 条），q16 是 `mul` + `srai 16` + `add`（3 条，
因为 Q16.16 的乘积要先右移 16 位再累加）。只给"乘"这一个助记符，q16 的矩阵乘会算错。
完整翻车记录见 `DEVELOPMENT.md` §2.3。

### 5.3 题册与两张实测表

**`bodies/__init__.py`**：

```python
BODIES  = {'add': add, 'reducesum': reducesum, 'matmul': matmul}
PROBLEMS = {
    'add-fp32': ('add', 'f32', 'rv32imf'),   # 题名 → (body 名, 数值类型, 目标机器)
    …
}
```

**`pipeline.py`**：`UNROLL`（展开因子）与 `BLOCKING`（寄存器分块 `(mr, nr)`），
两者都是**实测量**，不是猜的。换 target 后必须重新扫。

> `BLOCKING` 里没有 `matmul`（q16）是有原因的：q16 的数据寄存器在整数组（23 个），
> (4,4) 需要 25 个，**放不下**——`blocked_regs_needed()` 会算这个数，放不下就抛异常，
> 而不是默默生成错代码。

---

## 6. 加东西的入口

**逐步怎么做（含完整代码）在 `DEVELOPMENT.md` 第 7 章。** 这里只列「改哪个文件」：

| 想加 | 改哪 | 大概多少 |
|---|---|---|
| 一道同族（逐元素一对一）新题 | 新建 `bodies/<name>.py` + 在 `bodies/__init__.py` 登记 | ~20 行 |
| 一个新数值类型 | `dtypes.py` 加一条 policy（也许还要 `target.py` 加一项） | ~30 行 |
| 一个新指令集 | `target.py` 加一项 | ~20 行 |

> ⚠️ **现有仓库里没有任何"逐元素一对一之外"的循环形态。** 新题如果要成对处理
> （蝶形）、要归一化、或者形状写在输入张量头里，你得先在 `loopgen.py` 加一种新的
> 循环形态——**没有可照抄的先例**。

---

## 7. 排查手册

| 症状 | 判定 | 常见原因 |
|---|---|---|
| `该题暂未开放评测` | — | **没传场次**。`-fp32` 的题属于 `riscv-ai-2`；本地脚本要带 `contest=` |
| `compile_error` | **整题 0 分** | 汇编语法错误；用了目标不支持的指令；标签重名（同一文件里有两条路径时容易撞） |
| `isa_violation` | **整题 0 分** | 产出了 2 字节指令：`.option rvc` 或手写 `.2byte` |
| `invalid` + `越界写` | 该点失败 | 写越界。**`sp` 在 workspace 顶端，栈向下长**；偏移算错也会命中 guard |
| `invalid` + 数值不符，**全部**数据点都错 | 该点失败 | 大概率是**寄存器撞车**——某个寄存器在它不该被写的时候被写了 |
| `invalid` + 数值不符，**只有部分列/行**错 | 该点失败 | 常见于寄存器被复用：操作数在多次复用之间必须保持不变 |
| f32 题数值不符 | 该点失败 | 判据用错了。f32 是**容差**（`rtol=1e-4, atol=1e-3`），不是逐位 |
| `runtime_error` + 段错误 | 该点失败 | 非法访存；地址算错（把字节数当元素个数） |
| `timeout` | 该点失败 | 死循环；或指令数超出上限 |
| 本地好、线上崩 | — | 只跑了 `measure.py`（**它不跑 ISA 闸门**），没跑评测器 |

**逐寄存器的排查方法**（上面那两条"数值不符"都靠它）：拿张纸，把循环体里每个
寄存器列出来，**顺着生命周期**问「在这一行它装的是什么？」——
只看出错那一行往往是对的，错在几十行之前。做法见 `DEVELOPMENT.md` §4.4。

**吃寄存器的前提**：内核把自己当**叶子函数**用掉了 `gp` / `tp` / `s0-s11`。
成立条件是：平台编译锁死 `-Wl,--no-relax`、guest 裸机无 TLS、调用方在
`call cnn_entry` 之后只碰 `a0/a1/a2/a7/t0`。**三条任一变化都要重审寄存器池。**

---

## 8. ⚠️ 本地工具与线上链路的差异

| 工具 | 走的链路 | 差异 |
|---|---|---|
| `measure.py` | 手搓同一条 `riscv_runner` 链路 | **不跑 ISA 闸门**；额外打出 `i_miss` |
| `eval_local.py` | 直接调 `evaluator.run_evaluation` | 与线上**完全一致**（含 ISA 闸门与容差）。**但它不传场次**，`-fp32` 的题要用带 `contest=` 的脚本 |
| `sweep.py` / `*sweep*.py` | 逐候选真机编译+跑+计数 | 选型用 |

**`measure.py` 测出来的好数字不代表线上能过。** 提交前一律用评测器复核。

### 8.1 三个常备的一次性脚本

| 脚本 | 干什么 | 为什么需要 |
|---|---|---|
| `/tmp/eval.py` | 评测并打印逐点数字（带场次） | 线上同源 |
| `/tmp/eval1.py` | 同上，不传场次（内测比赛1 用） | — |
| `/tmp/oddn.py` | **用任意 N 测正确性** | 平台只用固定的 10 个数据点，**比赛走不到的代码路径测不到** |

**`oddn.py` 最该常备。** 平台的数据点全是 64 的倍数，所以"余数尾循环"、
"回退路径"、"边界分支"这些**永远不会被评测触发**——不自己造输入测，等于没写。
做法见 `DEVELOPMENT.md` §5.3。

---

## 9. 文档地图

| 文档 | 定位 |
|---|---|
| `DEVELOPMENT.md` | **教程**（2000+ 行）：从零写出六题，含全部可运行代码、实测数据、检查点 |
| 本文档 | **参考手册**：平台契约、字段表、命令行、排查 |
| `ARCHITECTURE.md` | **设计**：四轴分层、pass 清单与状态、目录结构、分期 |
| `OPTIMIZATION.md` | **性能**：代价分解、实测数据、优化入口清单 |
