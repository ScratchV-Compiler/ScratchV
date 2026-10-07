# 算子内核开发教程

> 这份文档带你**从零开始**写出内测比赛2 三道题的提交文件，并告诉你之后怎么把它做快。
>
> 读完之后你应该能：
> 1. 独立写出任意一道「给定输入张量、算出一个输出张量」的题的 RISC-V 汇编内核；
> 2. 把同一个内核换一个数值类型（定点 → 浮点）而不重写循环；
> 3. 用平台给的三个数字（指令数 / 缓存未命中 / cost）判断自己离最优还有多远。
>
> **本文档的每一段代码都是完整、可复制、可运行、并且已经跑通过的。**
> 每一步都给出「你应该看到什么」；如果你的输出不一样，附录 C 有排查表。

---

## 目录

- [读之前：你需要准备什么](#读之前你需要准备什么)
- [第 0 章 · 平台在考什么](#第-0-章--平台在考什么)
- [第 1 章 · 手工写出第一个内核](#第-1-章--手工写出第一个内核)
- [第 2 章 · 把手工的东西变成代码](#第-2-章--把手工的东西变成代码)
- [第 3 章 · 三题的完整实现](#第-3-章--三题的完整实现)
- [第 4 章 · 测量：知道自己在哪](#第-4-章--测量知道自己在哪)
- [第 5 章 · 变快（一）：不需要知道规模的优化](#第-5-章--变快一不需要知道规模的优化)
- [第 6 章 · 变快（二）：需要知道规模的优化](#第-6-章--变快二需要知道规模的优化)
- [第 7 章 · 加一道新题](#第-7-章--加一道新题)
- [第 8 章 · 提交前必须过的三关](#第-8-章--提交前必须过的三关)
- [附录 A · 术语表](#附录-a--术语表)
- [附录 B · 完整代码清单](#附录-b--完整代码清单)
- [附录 C · 排查手册](#附录-c--排查手册)

---

## 读之前：你需要准备什么

### 你需要会什么

| 需要 | 到什么程度 | 不会怎么办 |
|---|---|---|
| Python | 读写函数、字典、`dataclass` | 任何 Python 入门教程前 6 章 |
| RISC-V 汇编 | 知道 `lw`（从内存读一个字）、`add`、`beq`（相等就跳） | 附录 A 的术语表 + 本文档第 1 章会逐个讲 |
| 命令行 | `cd` / 运行 python 脚本 | — |

**不需要**：不需要会写编译器、不需要懂 SSA、不需要知道 LLVM 的 pass 机制。

### 准备环境

```bash
# 1. 仓库
cd /root/workspace/ScratchV

# 2. Python 环境（仓库自带）
./scratchv_env/bin/python --version
# 期望输出：Python 3.11.x（或更高）

# 3. 平台（用来评测，和你最终提交的是同一条链路）
ls /root/riscv-ai-compiler-platform/evaluator.py
# 期望输出：/root/riscv-ai-compiler-platform/evaluator.py
```

### 怎么用这份教程

**按顺序做。** 每一章末尾有 `✅ 检查点`，输出对不上就先解决，别往下走——
后面的内容都建立在前面的基础上。

本文档里的命令如果前面没有 `cd`，默认你在 `/root/workspace/ScratchV` 下。

---

## 第 0 章 · 平台在考什么

写内核之前，必须先把「平台怎么调用你的代码、按什么给分」搞清。这一章全是硬事实，
每一条都在平台源码里能查到。

### 0.1 你的代码长什么样：一个函数

平台要的是一段**自包含的 RISC-V 汇编**（文件后缀 `.s`），里面**只定义**一个全局符号：

```
cnn_entry
```

看图：

```
        平台生成的 wrapper（你不用写）                 你写的（只有这一块）
    ┌──────────────────────────────────┐        ┌──────────────────────┐
    │ 1. 准备输入张量，放到内存某处      │        │                      │
    │ 2. a0 ← 输入首址                 │───────▶│                      │
    │    a1 ← 输出首址                 │        │     cnn_entry:       │
    │    a2 ← 规模 N                   │        │        ...           │
    │ 3. call cnn_entry                │◀───────│        ret           │
    │ 4. 把整块内存 dump 出来做比对      │        │                      │
    └──────────────────────────────────┘        └──────────────────────┘
```

**三个参数**（汇编里的三个寄存器）：

| 寄存器 | 含义 |
|---|---|
| `a0` | 输入张量的**首地址**（内存里的一个地址） |
| `a1` | 输出张量的**首地址** |
| `a2` | **规模 N**（矩阵题是阶数，向量题是长度） |

**你不需要写输入输出、不需要写 main、不需要打印。** 你只要：从 `a0` 指向的内存读数据，
算，把结果写到 `a1` 指向的内存，然后 `ret`。

> **"首地址"是什么意思**：内存可以想象成一条很长的走廊，每间房间存一个数。
> `a0` 是走廊上"第一个数的房间号"。读第 i 个数，就是去 `a0 + i*4` 号房间
> （每个 float32 占 4 个字节，所以下标 i 要乘 4）。

### 0.2 内存长什么样：三段保护带，写出去就挂

平台给你划了一块连续的内存，从左到右是这样：

```
   guard_lo(256字节) │ workspace │ guard_mid(256字节) │ output │ guard_hi(256字节)
   ─────────────────────────────────────────────────────────────────────────────
        保护区         给你当栈/暂存       保护区         结果区        保护区
```

**`a1` 指向的是 `output` 那一段的开头。** 输入数据在 `a0`，它在外面的另一块内存里。

三块 `guard`（保护带）里平时全是 0。**只要有一个字节被写成非 0，这个数据点就判失败**，
报 `越界写：…的保护区被破坏`。

这是为了抓"你写到别人的地盘上了"。新人最常犯的是**写超出 N 个元素**——
比如循环结束条件写错，多写了一个。

还有一个你要知道的：**`sp`（栈指针）指向 workspace 的顶端，栈向下生长**。
如果你要在栈上放东西，`sp` 减小是合法的，减太多会撞到 `guard_lo`。

### 0.3 你的代码怎么被编译：两个参数是锁死的

平台用这条命令编译（`riscv_runner.compile_elf`）：

```bash
clang --target=riscv32-linux-gnu -march=rv32imf -mabi=ilp32 \
      -nostdlib -static -fuse-ld=lld -Wl,--no-relax \
      wrapper.s 你的文件.s -o execute.elf
```

三个词解释：

| 参数 | 意思 |
|---|---|
| `--target=riscv32-linux-gnu` | 目标是一台 32 位 RISC-V 机器 |
| `-march=rv32imf` | 这台机器支持的指令集：**I**（基础整数）+ **M**（乘除）+ **F**（单精度浮点）。比赛1 是 `rv32im`（没有 F） |
| `-mabi=ilp32` | 参数传递约定：整数用 32 位寄存器传（指针放在 `a0`/`a1`） |

**`-march` 是锁死的，你不能改。** 它决定了你能用哪些指令——`rv32imf` 下可以用
`flw`/`fadd.s`，`rv32im` 下只能用整数指令。

### 0.4 ⚠️ 红线一：压缩指令会让整题 0 分

RISC-V 有一个可选的「压缩指令」扩展（C 扩展），能让一部分指令变成 2 个字节。
看起来很美，**但平台上会被判整题 0 分**。

平台会扫描编译产物，**任何不是 4 字节的指令**都会触发：

```
提交包含 rv32im 之外的指令：有 302 条指令不是 4 字节（压缩扩展 c.*）。
本赛题只允许 rv32im 内的指令，且禁止用 .option rvc / .2byte 等方式绕过。
```

**处罚是整题 0 分**，不是只丢一个数据点。

所以你的每个文件开头都必须有这一行：

```asm
.option norvc
```

（`norvc` = no RVC，不许用压缩指令。）第 1 章会给你完整的文件模板。

### 0.5 成本怎么算：三个数字

平台衡量你的内核"贵不贵"，用的是这一个数：

```
cost = 指令数 + 15 × (数据缓存未命中次数 + 指令缓存未命中次数)
```

三个词解释：

| 词 | 意思 |
|---|---|
| **指令数** | 你的程序**执行过**多少条指令。注意是执行过，不是写了多少 |
| **缓存未命中** | CPU 存取内存时，要的那一块不在快速缓存里，就得慢一次。一次算 15 条指令的代价 |
| **指令缓存** | 存"指令"的缓存；你的代码太长、跳来跳去，就会一直从慢内存取指令 |

**为什么关心缓存**：你的内核里最慢的往往不是算，是**从内存取数**。比如做一个 64×64
的矩阵乘，三个矩阵各 16KB，加起来 48KB——而快速缓存只有 32KB，装不下，于是反复去慢内存拿。
实测那一个点就多花了 5 万多代价。

**这条式子给了你一个判断方法**：任何时候你都可以先看"我的 cost 里，指令数和未命中各占多少"，
再决定往哪优化。第 4 章会教你怎么看。

### 0.6 ⚠️ 红线二：平台不告诉你每一档的规模

第 0.1 节说 `a2` 是规模 N。你可能想"那我按 N 生成 10 份特化代码就行了"。

**不行。** 平台自 2026-10-07 起只公布**规模区间**（题面写 `N ∈ [64, 4096]`），
**逐个数据点是多少不公布**。

这直接决定了你的内核分两类：

| 类型 | 例子 | 能不能按 N 特化 |
|---|---|---|
| **区间窄** | matmul，`N ∈ [4, 64]`，只有 61 个整数 | 能，而且**不花代价**（没执行到的代码不计入 cost） |
| **区间宽** | add/reducesum，`N ∈ [64, 4096]`，4000 多个整数 | 不能穷举，得想别的办法（第 6 章） |

你现在只需要记住：**"先写一个对任意 N 都正确的版本"永远是第一步**，
因为它是所有其他优化的兜底。

### 0.7 计分：基准是全场最优，不是参考解

```
你在这个数据点的得分 = 该点满分 × min(1, 全场最优 cost ÷ 你的 cost)
```

三条推论：

1. **超过基准不给额外分。** 领先两倍和领先 0.1% 是同一个分数。
2. **`-O0` 参考解不是计分基准**（它只是平台上的一份快照）。真基准是**所有队伍里最小的那个 cost**。
3. **分数会随别人变强而下降**，所以"先提交"本身有价值。

**所以你的目标不是"越快越好"，是"每个数据点都追到当前榜首"。**

### ✅ 检查点

不看文档回答：

1. `a2` 里是什么？输出应该写到哪个寄存器指向的内存？
2. `.option norvc` 少了会怎样？
3. 你的 cost 里指令数是 1000、未命中是 200，cost 是多少？

答案：1) 规模 N；`a1`。2) 可能生成 2 字节指令 → 整题 0 分。3) 1000 + 15×200 = 4000。

---

## 第 1 章 · 手工写出第一个内核

这一章我们**手写**（不用任何框架）一个浮点向量加法内核，把它跑起来。
目的是让你亲眼看到"一个内核由哪几块组成"——第 2 章再把它变成代码。

### 1.1 题目

内测比赛2 的 `add-fp32`：

```
输入：2N 个 float32，前 N 个是 A，后 N 个是 B
输出：N 个 float32，C[i] = A[i] + B[i]
```

对应到寄存器：

- `a0` 指向 A 的第一个数；**A 后面紧跟 B**，所以 B 的第一个数在 `a0 + N*4`
- `a1` 指向 C 的第一个数
- `a2` = N

### 1.2 完整文件

新建 `/tmp/my_add.s`，内容如下。**这是完整的文件**，不是片段：

```asm
    .option norvc
    .option norelax

    .text
    .balign 4
    .globl cnn_entry
    .type cnn_entry, @function
cnn_entry:
    blez a2, .Lret          # 如果 N <= 0，直接返回
    slli t0, a2, 2          # t0 = N * 4   （每个数 4 字节，算字节数）
    add  t1, a0, t0         # t1 = &B[0]   （跳过 A 的 N 个数）
.Lloop:
    flw  ft0, 0(a0)         # ft0 = A[i]
    flw  ft1, 0(t1)         # ft1 = B[i]
    fadd.s ft0, ft0, ft1    # ft0 = A[i] + B[i]
    fsw  ft0, 0(a1)         # C[i] = ft0
    addi a0, a0, 4          # A 的指针往后走一个数
    addi t1, t1, 4          # B 的指针往后走一个数
    addi a1, a1, 4          # C 的指针往后走一个数
    addi a2, a2, -1         # 剩余个数 -1
    bnez a2, .Lloop         # 还没处理完就跳回循环开头
.Lret:
    ret
```

### 1.3 逐行讲解

**开头 7 行是每个内核都要有的模板**：

| 行 | 作用 |
|---|---|
| `.option norvc` | 禁止压缩指令（第 0.4 节的红线） |
| `.option norelax` | 禁止链接器优化寻址方式（保证指令数稳定，跨队可比） |
| `.text` | 下面是指令，不是数据 |
| `.balign 4` | 对齐到 4 字节边界 |
| `.globl cnn_entry` | 让 `cnn_entry` 这个名字对链接器可见（平台靠它找到入口） |
| `.type cnn_entry, @function` | 声明它是一个函数 |
| `cnn_entry:` | **标签**：这一行有一个地址，别的指令可以跳到这里 |

**然后是三条准备指令**：

| 指令 | 意思 |
|---|---|
| `blez a2, .Lret` | 如果 `a2` ≤ 0，跳到 `.Lret`（`blez` = branch if ≤ zero）。这是为了应付 N=0 这种边界，避免循环一次都不进却先减了计数 |
| `slli t0, a2, 2` | `t0 = a2 << 2`，即 **N × 4**。左移 2 位等于乘 4。为什么要乘 4？因为一个 float32 占 4 个**字节**，而地址是按字节编的 |
| `add t1, a0, t0` | `t1 = a0 + N×4`，也就是 **B 的第一个数的地址** |

> **"标签"是什么**：`cnn_entry:` 和 `.Lloop:` 这类以冒号结尾的行，不产生指令，
> 只是给某个位置起个名字。`blez a2, .Lret` 里的 `.Lret` 就是"跳到那个名字所在的位置"。
> 以 `.L` 开头的标签是**局部标签**（local），不会出现在最终符号表里——这是惯例，你也照做。

**再是循环体**（`.Lloop:` 到 `bnez`），9 条指令：

| 指令 | 意思 |
|---|---|
| `flw ft0, 0(a0)` | **F**loat **L**oad **W**ord：从地址 `a0+0` 读一个 32 位数进浮点寄存器 `ft0` |
| `flw ft1, 0(t1)` | 同上，从 B 读 |
| `fadd.s ft0, ft0, ft1` | 单精度浮点加法 |
| `fsw ft0, 0(a1)` | **F**loat **S**tore **W**ord：把 `ft0` 写到地址 `a1+0` |
| `addi a0, a0, 4` | `a0 += 4`（add immediate），指针往后挪一个 float32 |
| `addi t1, t1, 4` | B 的指针往后挪 |
| `addi a1, a1, 4` | C 的指针往后挪 |
| `addi a2, a2, -1` | 剩余个数减一 |
| `bnez a2, .Lloop` | **b**ranch if **n**ot **ez**ero：`a2` 不为 0 就跳回 `.Lloop` |

**最后**：

| 指令 | 意思 |
|---|---|
| `.Lret:` | 标签 |
| `ret` | 返回（跳到调用者那里） |

### 1.4 跑起来

先确认它能编译（这一步不需要平台）：

```bash
clang --target=riscv32-linux-gnu -march=rv32imf -mabi=ilp32 -c /tmp/my_add.s -o /tmp/my_add.o
echo $?
# 期望输出：0    （0 表示成功；非 0 会打印错误原因）
```

再看看数一下指令：

```bash
llvm-objdump -d /tmp/my_add.o | grep -cE '^\s*[0-9a-f]+:'
# 期望输出：13
```

13 条 = 3 条准备 + 9 条循环体 + 1 条 `ret`。

> 你会看到 `llvm-objdump` 把 `flw`/`fadd.s` 显示成 `<unknown>`。
> **这是正常的**——这套工具链的反汇编器没启用 F 扩展的解码。它照样能算准每条指令
> **占几个字节**（这正是第 0.4 节那道红线用的判据）。

现在用平台评测器跑一遍真数据（10 个数据点）：

```bash
cd /root/riscv-ai-compiler-platform
set -a && . /root/.riscv_platform_env && set +a

cat > /tmp/eval.py <<'EOF'
import sys, os
sys.path.insert(0, '/root/riscv-ai-compiler-platform')
os.chdir('/root/riscv-ai-compiler-platform')
os.environ.setdefault('PLATFORM_ENABLE_SANDBOX', '0')
from app import app
from evaluator import run_evaluation
code, problem, contest = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
with app.app_context():
    score, d = run_evaluation(0, 'local', problem, code, contest=contest)
print(f"结论: {d.get('verdict')}   得分: {score}   说明: {d.get('message')}")
for c in d.get('cases') or []:
    print(f"  N={c['size']:>5}  指令数={str(c['instructions']):>8}  "
          f"数据未命中={str(c['d_miss']):>5}  cost={str(c['cost']):>8}  "
          f"基准={str(c['baseline']):>8}  {c['verdict']}")
EOF

PLATFORM_ENABLE_SANDBOX=0 ./.venv/bin/python /tmp/eval.py /tmp/my_add.s add-fp32 riscv-ai-2
```

**期望输出**（节选最后几行）：

```
结论: accepted   得分: 30.0   说明: 全部 10/10 个数据点通过，得分 30.0/30
  N= 3072  指令数=   27675  数据未命中=  580  cost=   36420  基准=   73364  accepted
  N= 4096  指令数=   36890  数据未命中=  772  cost=   48515  基准=   97747  accepted
```

**注意 `contest=riscv-ai-2` 这个参数。** `add-fp32` 属于场次 `riscv-ai-2`；
不写的话平台会去默认场次（内测比赛1）找，那里没有 `-fp32` 的题，你会得到
`该题暂未开放评测`。

> **`36890 ÷ 4096 = 9.0`** —— 正好是上面循环体的 9 条指令。
> 这个数就是你的"每条元素的成本"：**越小越好**。

### 1.5 试一件事：换成定点数

比赛1 的 `add` 是同一道题，只是数值类型换成 Q16.16 定点（用整数指令模拟小数）。

**同一个文件，改 4 个地方就行**：

| 原来（float32） | 改成（Q16.16 定点） |
|---|---|
| `flw  ft0, 0(a0)` | `lw   t2, 0(a0)` |
| `flw  ft1, 0(t1)` | `lw   t3, 0(t1)` |
| `fadd.s ft0, ft0, ft1` | `add  t2, t2, t3` |
| `fsw  ft0, 0(a1)` | `sw   t2, 0(a1)` |

编译时把 `-march=rv32imf` 换成 `-march=rv32im`，评测时把 `add-fp32`/`riscv-ai-2`
换成 `add`/不传场次。

**记住这个感觉**：改的是四个**叶子操作**，循环结构一条没动。
第 2 章要做的，就是把这四行从代码里"提出来"，变成一个**参数**。

### 1.6 小结：一个内核由哪几块组成

```
cnn_entry:
    ① 模板      .option norvc / .globl / 标签
    ② 准备      N 是不是 0？算字节数？指针初始化？
    ③ 循环      读 → 算 → 写 → 指针前进 → 计数 → 判断
    ④ 收尾      ret
```

**这四块里，只有 ③ 中间那句"算"是跟数值类型有关的。** 其余三块对
float32 和定点完全一样。第 2 章的框架就是把这句话变成数据。

### ✅ 检查点

1. 把 `/tmp/my_add.s` 的 `blez a2, .Lret` 删掉，N=0 时会发生什么？
2. `slli t0, a2, 2` 里的 2 能不能改成 3？改了会怎样？
3. 循环体为什么是 9 条而不是 4 条？

答案：1) 会多走一轮循环、写坏内存（越界）。2) 变成乘 8，`t1` 会指错地方，读到 A 的中间。
3) 4 条是"读读加写"，另外 5 条是"三个指针各前进、计数减一、判断跳回"。

---

## 第 2 章 · 把手工的东西变成代码

### 2.1 为什么要框架

如果你只做一道题，第 1 章的写法就够了，**不需要任何框架**。

框架的价值在**第 3 道题之后**开始显现：

| 没有框架 | 有框架 |
|---|---|
| 加 fp32 版 → 把三道题的循环体各改一遍 | 加一个表项 |
| 加一个新指令集 → 改所有题 | 加一个表项 |
| 加一道新题 → 从零写 | 写一段语义，约 20 行 |

**做框架的唯一目的：让"数值类型"和"指令集"变成数据，而不是散在代码里的字面量。**

### 2.2 第一个文件：`target.py`（这台机器是什么）

新建 `scratchv/backend/kernels/target.py`：

```python
# -*- coding: utf-8 -*-
"""目标机器描述：唯一允许写指令助记符和寄存器名的地方。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetDesc:
    name: str                    # 这台机器的名字，如 'rv32imf'
    march: str                   # 编译时传给 clang 的 -march
    mabi: str                    # 编译时传给 clang 的 -mabi
    int_regs: tuple[str, ...]    # 整数寄存器池（指针、计数用）
    fp_regs: tuple[str, ...]     # 浮点寄存器池（算浮点用；没有就留空）
    load: dict[str, str]         # 数值类型 → 载入指令
    store: dict[str, str]        # 数值类型 → 存储指令
    imm_max: int                 # 立即数上限（见下面"什么是立即数"）
    line_bytes: int              # 一级缓存的行大小，第 4 章会用到


# 整数寄存器：a0/a1/a2 是平台传进来的参数，不动；其余都可用
INT_POOL = ('s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
            't0', 't1', 't2', 't3', 't4', 't5',
            'a3', 'a4', 'a5', 'a6', 'a7')

# 浮点寄存器：f0 到 f31
FP_POOL = tuple(f'f{i}' for i in range(32))


TARGETS: dict[str, TargetDesc] = {
    'rv32im': TargetDesc(
        name='rv32im', march='rv32im', mabi='ilp32',
        int_regs=INT_POOL, fp_regs=(),
        load={'int32': 'LW'}, store={'int32': 'SW'},
        imm_max=2047, line_bytes=64,
    ),
    'rv32imf': TargetDesc(
        name='rv32imf', march='rv32imf', mabi='ilp32',
        int_regs=INT_POOL, fp_regs=FP_POOL,
        load={'int32': 'LW', 'f32': 'FLW'},      # ← 同一种算子，多了一种数
        store={'int32': 'SW', 'f32': 'FSW'},
        imm_max=2047, line_bytes=64,
    ),
}
```

**什么是立即数**：指令里直接写在指令内部的常数，比如 `addi a0, a0, 4` 里的 `4`。
它只有 12 位有符号的空间，**范围是 −2048 到 2047**。超过这个范围就得先用 `li`
把数装进寄存器再运算——多一条指令。`imm_max` 记着这个上限，第 6 章会用到。

### 2.3 第二个文件：`dtypes.py`（这种数怎么算）

新建 `scratchv/backend/kernels/dtypes.py`：

```python
# -*- coding: utf-8 -*-
"""数值类型描述：把"这种数怎么算"变成数据。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DtypePolicy:
    name: str
    load: str                    # 载入助记符
    store: str                   # 存储助记符
    add: str                     # 加法助记符
    acc: str                     # 累加器寄存器名
    tmp1: str                    # 临时寄存器 1
    tmp2: str                    # 临时寄存器 2
    mac: tuple[str, ...]         # ★ 一次乘加的指令序列；{acc}/{t1}/{t2} 由 body 填
    zero: tuple[str, ...]        # 累加器置零模板；{acc} = 累加器，{r} = 空闲整数寄存器
    comparison: str              # 'exact'（逐位）或 'tolerance'（容差）


POLICIES: dict[str, DtypePolicy] = {
    'q16': DtypePolicy(
        name='q16', load='lw', store='sw', add='add',
        # 整数路径下累加器和临时值都是整数寄存器；用 s2/s3/s4，
        # 与三个 body 用到的 t0-t6 / a3-a5 不相交。
        acc='s2', tmp1='s3', tmp2='s4',
        # ★ Q16.16 的乘积要**先右移 16 位**再累加（赛题语义 `Σ (A×B) >> 16`）。
        mac=('mul {t1}, {t1}, {t2}', 'srai {t1}, {t1}, 16', 'add {acc}, {acc}, {t1}'),
        zero=('li {acc}, 0',),
        comparison='exact',
    ),
    'f32': DtypePolicy(
        name='f32', load='flw', store='fsw', add='fadd.s',
        # 浮点路径下累加器和临时值都在浮点寄存器组，与整数寄存器天然不相交。
        acc='ft0', tmp1='ft1', tmp2='ft2',
        # 浮点没有定点重标定，乘完直接累加。
        mac=('fmul.s {t1}, {t1}, {t2}', 'fadd.s {acc}, {acc}, {t1}'),
        zero=('li {r}, 0', 'fmv.w.x {acc}, {r}'),
        comparison='tolerance',          # 浮点不能逐位比，要看平台给的容差
    ),
}


def mac_instrs(dtype: DtypePolicy) -> list[str]:
    """把一次乘加的模板填上寄存器名。"""
    return [line.format(acc=dtype.acc, t1=dtype.tmp1, t2=dtype.tmp2)
            for line in dtype.mac]


def zero_acc(dtype: DtypePolicy, scratch: str) -> list[str]:
    """把累加器置零。

    `scratch` 必须是一个**调用方确定此刻空闲**的整数寄存器。
    为什么不能写死一个？第 3.3 节有一个真实的翻车例子。
    """
    return [line.format(acc=dtype.acc, r=scratch) for line in dtype.zero]
```

**为什么 `zero` 要带占位符**：浮点寄存器不能直接用来装 0 这个整数，
必须借一个整数寄存器中转（`fmv.w.x` 就是"把整数寄存器的位原样搬进浮点寄存器"）。
用哪个整数寄存器**取决于调用它的地方有没有在用**——所以由调用方给。

#### ⚠️ 为什么是 `mac` 而不是 `mul`——抽象层级不够（实测撞到的）

最初的 policy 里只有一个 `mul` 助记符，默认"乘法"在两种数值类型下是同一样东西。
**错了**：

| | 一次乘加（MAC）＝ 什么 |
|---|---|
| **f32** | `fmul.s` + `fadd.s`（2 条） |
| **q16** | `mul` + **`srai 16`** + `add`（3 条） |

Q16.16 的赛题语义是 `Σ ((A×B) >> 16)`——**每个乘积要先右移 16 位再累加**。
所以"乘"在两边的**形状不一样**。

只给 `mul` 的后果：q16 的 `matmul` 汇编能过、`add`/`reducesum` 全对，
**但 matmul 0/10**（`invalid`，所有数据点数值不符）。把 `mac` 提进 policy 就好了。

> **教训**：**policy 的粒度要落在「语义单元」上，不是「指令」上。**
> 内测比赛1 那份交付把 q16 的 MAC 优化成了 `mulh(a << 16, b)` + `add`（2 条）——
> 那是同一个 `mac` 字段的**另一个取值**，不是另一个抽象层。
> **先定对粒度，优化才有地方放。**

### 2.4 一条必须守住的规矩

> **助记符（`flw`、`fadd.s`、`lw`…）只允许出现在 `target.py` 和 `dtypes.py` 里。**

如果你在后面写的 `bodies/` 里写了 `flw`，那么"加一个新指令集就得把所有题再改一遍"
——框架就白做了。写完记得自查：

```bash
grep -rnE '\b(flw|fsw|lw|sw|fadd\.s|fmul\.s)\b' scratchv/backend/kernels/bodies/
# 期望：没有任何输出
```

### ✅ 检查点

1. 如果要加一个 `rv32im` 带 P 扩展的 target，改哪个文件？
2. `f32` 的 `comparison` 为什么是 `'tolerance'` 而不是 `'exact'`？
3. 为什么 `zero` 不能直接写 `('li t6, 0', 'fmv.w.x ft0, t6')`？

答案：1) 只改 `target.py`。2) 浮点加法不满足结合律，换个顺序结果就不同，平台给的是容差
（`rtol=1e-4, atol=1e-3`）。3) 因为 `t6` 在别的地方可能正在用——第 3.3 节有实例。

---

## 第 3 章 · 三题的完整实现

### 3.1 公共部分：`loopgen.py`

新建 `scratchv/backend/kernels/loopgen.py`：

```python
# -*- coding: utf-8 -*-
"""把一段叶子算子套进循环里，产出汇编文本。"""


def prologue(entry: str = 'cnn_entry') -> list[str]:
    """每个 .s 的开头模板。`.option norvc` 不能少 —— 见教程第 0.4 节。"""
    return [
        '    .option norvc',
        '    .option norelax',
        '',
        '    .text',
        '    .balign 4',
        f'    .globl {entry}',
        f'    .type {entry}, @function',
        f'{entry}:',
    ]


def epilogue() -> list[str]:
    return ['.Lret:', '    ret', '']
```

这两个函数就是第 1 章那个模板，提取出来复用。

### 3.2 `add`：`bodies/add.py`

新建 `scratchv/backend/kernels/bodies/add.py`。**完整文件**：

```python
# -*- coding: utf-8 -*-
"""C[i] = A[i] + B[i]

入口：a0 = [A(N) 后面紧跟 B(N)]，a1 = C，a2 = N
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue


def build(target, dtype) -> list[str]:
    L = dtype.load           # 载入助记符：f32 是 flw，q16 是 lw
    S = dtype.store          # 存储助记符
    ADD = dtype.add          # 加法助记符
    T1, T2 = dtype.tmp1, dtype.tmp2

    return [
        *prologue(),
        '    blez a2, .Lret',                # N <= 0 直接返回
        '    slli t0, a2, 2',                # t0 = N * 4
        '    add  t1, a0, t0',               # t1 = &B[0]
        '.Lloop:',
        f'    {L}  {T1}, 0(a0)',             # 读 A[i]
        f'    {L}  {T2}, 0(t1)',             # 读 B[i]
        f'    {ADD} {T1}, {T1}, {T2}',       # 算
        f'    {S}  {T1}, 0(a1)',             # 写 C[i]
        '    addi a0, a0, 4',                # 三个指针各前进一个元素
        '    addi t1, t1, 4',
        '    addi a1, a1, 4',
        '    addi a2, a2, -1',               # 剩余个数 -1
        '    bnez a2, .Lloop',
        *epilogue(),
    ]
```

**和手写版对比**：载入/存储/加法三个助记符，加上两个临时寄存器，
全部从 `dtype` 取。**注意这里没有任何指令助记符字面量，也没有写死的寄存器名。**

它的输出（`f32` 时）：

```asm
    .option norvc
    .option norelax

    .text
    .balign 4
    .globl cnn_entry
    .type cnn_entry, @function
cnn_entry:
    blez a2, .Lret
    slli t0, a2, 2
    add  t1, a0, t0
.Lloop:
    flw  ft1, 0(a0)
    flw  ft2, 0(t1)
    fadd.s ft1, ft1, ft2
    fsw  ft1, 0(a1)
    addi a0, a0, 4
    addi t1, t1, 4
    addi a1, a1, 4
    addi a2, a2, -1
    bnez a2, .Lloop
.Lret:
    ret
```

**9 条指令/元素。** 记住这个数，第 5 章要把它降到 5.4。

### 3.3 `matmul` 与一个真实的坑：`bodies/matmul.py`

先看题目：`C[i][j] = Σₖ A[i][k] · B[k][j]`，A 和 B 各 N×N，行主序（一行一行存）。
入口：`a0 = [A(N²) 后面紧跟 B(N²)]`，`a1 = C`，`a2 = N`。

三层循环直接写（**完整文件**）：

```python
# -*- coding: utf-8 -*-
"""C[i][j] = sum_k A[i][k] * B[k][j]

入口：a0 = [A(N*N) 后面紧跟 B(N*N)]，a1 = C，a2 = N
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue
from scratchv.backend.kernels.dtypes import zero_acc, mac_instrs


def build(target, dtype) -> list[str]:
    L = dtype.load
    S = dtype.store
    ACC, T1, T2 = dtype.acc, dtype.tmp1, dtype.tmp2

    return [
        *prologue(),
        '    blez a2, .Lret',
        '    mul  t0, a2, a2',              # t0 = N * N
        '    slli t0, t0, 2',               # t0 = N * N * 4  （B 的起始偏移）
        '    add  t5, a0, t0',              # t5 = &B[0]
        '    slli t6, a2, 2',               # t6 = 行步长 = N * 4
        '    mv   t3, a0',                  # t3 = &A[i][0]
        '    mv   t4, a1',                  # t4 = &C[i][0]
        '    li   t1, 0',                   # i = 0
        '.Li:',
        '    li   t2, 0',                   # j = 0
        '.Lj:',
        *zero_acc(dtype, 's0'),             # ★ 累加器置零：用 s0，见下面的坑
        '    mv   a3, t3',                  # a3 = &A[i][0]
        '    slli a4, t2, 2',               # a4 = j * 4
        '    add  a4, t5, a4',              # a4 = &B[0][j]
        '    li   a5, 0',                   # k = 0
        '.Lk:',
        f'    {L}  {T1}, 0(a3)',            # 读 A[i][k]
        f'    {L}  {T2}, 0(a4)',            # 读 B[k][j]
        *mac_instrs(dtype),                 # ★ 乘加：f32 是 2 条，q16 是 3 条（含 >>16）
        '    addi a3, a3, 4',               # A 的 k 方向 +1
        '    add  a4, a4, t6',              # B 的 k 方向 +1 行
        '    addi a5, a5, 1',               # k++
        '    bne  a5, a2, .Lk',             # k != N 就继续
        '    slli t0, t2, 2',               # 算出 C[i][j] 的地址
        '    add  t0, t4, t0',
        f'    {S}  {ACC}, 0(t0)',           # 写回
        '    addi t2, t2, 1',               # j++
        '    bne  t2, a2, .Lj',
        '    add  t3, t3, t6',              # A 的行指针 +1 行
        '    add  t4, t4, t6',              # C 的行指针 +1 行
        '    addi t1, t1, 1',               # i++
        '    bne  t1, a2, .Li',
        *epilogue(),
    ]
```

#### ⚠️ 那个坑：`t6` 被吃掉了

我第一版把 `zero_acc` 的暂存寄存器写成 `t6`：

```python
*zero_acc(dtype, 't6'),          # ← 错！
```

结果：**10 个数据点全部算错**（`invalid 第 0 个输出数值不符`），
而 `add` 和 `reducesum` 全部通过。

**为什么**：`t6` 在上面第 5 行被赋成了**行步长 `N*4`**：

```
slli t6, a2, 2               # t6 = N * 4   ← 行步长
...
.Lj:
    li   t6, 0               # ← 置零把它清零了！
    fmv.w.x ft0, t6
    ...
    add  a4, a4, t6          # 想把 a4 往下挪一行，但 t6 已经是 0
```

于是 B 的 k 方向永远不前进，A 的行也不前进——**算的是错的，但汇编器不报错**。

**教训**：暂存寄存器必须是"调用点确定空闲"的那个。这就是为什么 `zero` 是模板而不是写死的两行。
改成 `s0` 之后：

```
结论: accepted   得分: 30.0   说明: 全部 10/10 个数据点通过，得分 30.0/30
  N=   64  指令数= 2142559  数据未命中=  773  cost= 2154214  基准= 7459265  accepted
```

`2142559 ÷ 64³ = 8.17` —— **8.17 条指令/MAC**（MAC = 一次乘加）。

### 3.4 `reducesum`：`bodies/reducesum.py`

`out[0] = Σ x[i]`，入口：`a0 = x(N)`，`a1 = out`，`a2 = N`。

```python
# -*- coding: utf-8 -*-
"""out[0] = sum x[i]

入口：a0 = x(N)，a1 = out，a2 = N
"""

from scratchv.backend.kernels.loopgen import prologue, epilogue
from scratchv.backend.kernels.dtypes import zero_acc


def build(target, dtype) -> list[str]:
    L = dtype.load
    S = dtype.store
    ADD = dtype.add
    ACC, T1 = dtype.acc, dtype.tmp1

    return [
        *prologue(),
        '    blez a2, .Lret',
        *zero_acc(dtype, 't6'),                 # t6 在这道题里没别的用途，安全
        '.Lloop:',
        f'    {L}  {T1}, 0(a0)',
        f'    {ADD} {ACC}, {ACC}, {T1}',
        '    addi a0, a0, 4',
        '    addi a2, a2, -1',
        '    bnez a2, .Lloop',
        f'    {S}  {ACC}, 0(a1)',
        *epilogue(),
    ]
```

**5 条指令/元素。**

#### ⚠️ 第二个坑：`mul` 不够，要 `mac`

把 dtype 换成 q16 去跑内测比赛1 时：

```
add        q16 accepted  得分=30.0   ← 过
reducesum  q16 accepted  得分=40.0   ← 过
matmul     q16 invalid   得分=0.0    ← 0/10，全部数值不符
```

**原因不是 bug，是抽象层级错了**：Q16.16 的赛题语义是 `Σ ((A×B) >> 16)`——
**每个乘积要先右移 16 位再累加**。而 policy 里只有一个 `mul` 助记符，body 写的是
"乘完直接累加"（对 f32 对，对 q16 错）。

修法见 §2.3：把 `mul` 换成 **`mac`（一次乘加的整条指令序列）**。
改完之后六题全过（见 §3.7）。

### 3.5 题册与组装

新建 `scratchv/backend/kernels/bodies/__init__.py`：

```python
# -*- coding: utf-8 -*-
"""题册：题名 → （用哪个 body，哪种数值，哪台机器）。"""

from scratchv.backend.kernels.bodies import add, matmul, reducesum

BODIES = {
    'add': add,
    'reducesum': reducesum,
    'matmul': matmul,
}

# 题名 → (body 名, 数值类型, 目标机器)
PROBLEMS = {
    'add-fp32':       ('add',       'f32', 'rv32imf'),
    'reducesum-fp32': ('reducesum', 'f32', 'rv32imf'),
    'matmul-fp32':    ('matmul',    'f32', 'rv32imf'),

    # 比赛1 的三题：只换数值类型和目标机器，body 一个字都不用改
    'add':            ('add',       'q16', 'rv32im'),
    'reducesum':      ('reducesum', 'q16', 'rv32im'),
    'matmul':         ('matmul',    'q16', 'rv32im'),
}
```

**注意最后三行。** 内测比赛1 的三题在这里是**零成本接入**的——因为 body 不写死助记符，
把"乘加的形态"放在了 policy 的 `mac` 字段里（§2.3 的第二个坑）。
这就是第 2.4 节那条规矩的回报。

还要新建一个空的 `scratchv/backend/kernels/__init__.py`（让这个目录成为一个包）。

新建 `scratchv/backend/kernels/pipeline.py`：

```python
# -*- coding: utf-8 -*-
"""组装：题名 → 完整的 .s 文本。"""

from scratchv.backend.kernels.bodies import BODIES, PROBLEMS
from scratchv.backend.kernels.dtypes import POLICIES
from scratchv.backend.kernels.target import TARGETS


def build_program(problem: str) -> str:
    body_name, dtype_name, target_name = PROBLEMS[problem]
    body = BODIES[body_name]
    dtype = POLICIES[dtype_name]
    target = TARGETS[target_name]
    return '\n'.join(body.build(target, dtype))
```

新建 `scratchv/backend/kernels/__main__.py`：

```python
# -*- coding: utf-8 -*-
"""命令行入口：

    python -m scratchv.backend.kernels --problem add-fp32 -o out.s
    python -m scratchv.backend.kernels --list
"""

import argparse

from scratchv.backend.kernels.bodies import PROBLEMS
from scratchv.backend.kernels.pipeline import build_program


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='ScratchV 算子内核生成器')
    ap.add_argument('--problem', help='题名，见 --list')
    ap.add_argument('-o', '--output', help='输出 .s 路径（默认打印到屏幕）')
    ap.add_argument('--list', action='store_true', help='列出所有题名')
    args = ap.parse_args(argv)

    if args.list:
        for name, (body, dtype, target) in sorted(PROBLEMS.items()):
            print(f'{name:<18} body={body:<10} dtype={dtype:<5} target={target}')
        return 0

    if not args.problem:
        ap.error('要么给 --problem，要么给 --list')

    text = build_program(args.problem)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'已写出 {args.output}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

### 3.6 跑通六题

```bash
cd /root/workspace/ScratchV

./scratchv_env/bin/python -m scratchv.backend.kernels --list
# 期望输出（6 行）：
# add                body=add        dtype=q16   target=rv32im
# add-fp32           body=add        dtype=f32   target=rv32imf
# ...

./scratchv_env/bin/python -m scratchv.backend.kernels --problem add-fp32 -o /tmp/scv_add_fp32.s
# 期望输出：已写出 /tmp/scv_add_fp32.s
```

然后生成并评测全部六题：

```bash
cd /root/workspace/ScratchV
for t in add reducesum matmul; do
  for p in $t ${t}-fp32; do
    ./scratchv_env/bin/python -m scratchv.backend.kernels --problem $p -o /tmp/scv_$p.s
  done
done

cd /root/riscv-ai-compiler-platform
set -a && . /root/.riscv_platform_env && set +a
PY=./.venv/bin/python

echo "── 内测比赛2（fp32）──"
for t in add reducesum matmul; do
  PLATFORM_ENABLE_SANDBOX=0 $PY /tmp/eval.py /tmp/scv_${t}-fp32.s ${t}-fp32 riscv-ai-2 | head -1
done

echo "── 内测比赛1（q16，不传场次就是默认场次）──"
for t in add reducesum matmul; do
  PLATFORM_ENABLE_SANDBOX=0 $PY /tmp/eval1.py /tmp/scv_$t.s $t | tail -1
done
```

（`/tmp/eval1.py` = `/tmp/eval.py` 去掉 `contest` 参数，见第 1.4 节。）

**期望输出（这份教程的作者自测结果）**：

```
── 内测比赛2（fp32）──
  add-fp32         结论: accepted   得分: 30.0   说明: 全部 10/10 个数据点通过，得分 30.0/30
  reducesum-fp32   结论: accepted   得分: 40.0   说明: 全部 10/10 个数据点通过，得分 40.0/40
  matmul-fp32      结论: accepted   得分: 30.0   说明: 全部 10/10 个数据点通过，得分 30.0/30
── 内测比赛1（q16）──
  add               accepted  得分=30.0  全部 10/10 个数据点通过
  reducesum         accepted  得分=40.0  全部 10/10 个数据点通过
  matmul            accepted  得分=30.0  全部 10/10 个数据点通过
```

（`reducesum` 满分是 40 不是 30，因为它每点 4 分。）

**现在你已经有 O0 档的实现了。** 对照第 0.7 节的参考解：

| 题（最大数据点） | 参考解 cost | 你的 cost | 快多少 |
|---|---:|---:|---:|
| add-fp32（N=4096） | 97,747 | 48,515 | **2.01×** |
| reducesum-fp32（N=4096） | 73,650 | 24,437 | **3.01×** |
| matmul-fp32（N=64） | 7,459,265 | 2,154,214 | **3.46×** |

### ✅ 检查点

1. 现在要加一道 `abs-fp32`，你要改几个文件？
2. `bodies/matmul.py` 里为什么用 `s0` 而不是 `t6` 做置零暂存？
3. `mac` 字段为什么不能简化成一个 `mul` 助记符？
4. 如果把 `PROBLEMS` 里 `add-fp32` 的 dtype 改成 `q16` 但 target 不改，会发生什么？

答案：
1) 两个——新建 `bodies/abs.py`，再在 `bodies/__init__.py` 里加两行（`BODIES` + `PROBLEMS`）。
2) 因为 `t6` 存着行步长 `N*4`。
3) 因为一次乘加在两种数值类型下**形状不同**：f32 是 `fmul.s`+`fadd.s`（2 条），
q16 是 `mul`+`srai 16`+`add`（3 条）。只给"乘"这一个助记符，q16 的矩阵乘就会算错。
4) 会用 `lw` 载入浮点数据（助记符来自 q16 policy）——数值会完全错乱，
但**汇编器不会报错**。

---

## 第 4 章 · 测量：知道自己在哪

### 4.1 只看三个数字

| 数字 | 含义 | 什么时候它是瓶颈 |
|---|---|---|
| **指令数** | 执行了多少条指令 | 通常是大头。降低它 = 少算、少搬指针 |
| **数据未命中** | 从内存取数没命中缓存的次数 | 数据大（如 matmul N=64）时突然暴涨 |
| **指令未命中** | 取指令没命中缓存的次数 | 代码太长、跳来跳去时 |

`cost = 指令数 + 15 × (两个未命中之和)`。

### 4.2 怎么读

第 1.4 节的 `/tmp/eval.py` 已经会打印这三个数。拿 add-fp32 举例：

```
  N=  64  指令数=     602  数据未命中=   14  cost=     857  基准=    1705
  N=4096  指令数=   36890  数据未命中=  772  cost=   48515  基准=   97747
```

**怎么念这张表**：

- **指令数 ÷ N = 每元素成本。** `36890 ÷ 4096 = 9.0`，正好是循环体的 9 条。
  `602 ÷ 64 = 9.4`（多出来的 0.4 是固定的准备指令摊薄后剩下的）。
- **未命中在大 N 时才显著。** N=64 时未命中只值 `15×14 = 210`（占总 cost 24%）；
  N=4096 时值 `15×772 = 11580`（占 24%）。
- **`cost` 与"基准"比，就是你的位置。** `97747 ÷ 48515 = 2.01`。

### 4.3 一个必须养成的习惯

**每次改完代码，都先看"指令数和未命中各占多少"，再决定往哪优化。**

这两项的修法完全不同：

| 瓶颈在 | 怎么修 | 看第几章 |
|---|---|---|
| 指令数 | 摊薄循环开销、减少访存（寄存器分块） | 第 5、6 章 |
| 数据未命中 | 改循环顺序 / 分块，提升缓存局部性 | 第 6 章 |

**凭感觉优化 = 白干。** 有一个具体的例子：比赛1 的内核曾经为了"修缓存"去做分块，
后来发现那个版本的未命中**已经接近理论最小值**，分块是无用功——第 6 章会讲怎么看。

### ✅ 检查点

1. 某内核 N=1024：指令数 5000，数据未命中 300，指令未命中 20。它的 cost 是多少？
2. 上题里，指令数和未命中各占 cost 的百分之多少？该先优化哪个？

答案：1) 5000 + 15×(300+20) = 9800。2) 指令数占 51%，未命中占 49%。
——两边都差不多，先做第 5 章（改一个函数三题都受益）。

---

## 第 5 章 · 变快（一）：不需要知道规模的优化

这一章的两个技巧**对任何 N 都适用**，不需要知道数据点的具体规模。

### 5.1 技巧一：展开循环（`unroll`）

**问题**：看第 3.2 节 add 的循环体——9 条指令里，只有 4 条在真正干活
（两次读、一次加、一次写），另外 5 条是"三个指针各前进一次、计数减一、判断跳回"。

**每处理一个元素就要付一次这 5 条的开销。**

**办法**：一次处理 `unroll` 个元素，开销就摊到 `unroll` 个元素上。

一次处理 4 个的写法：

```asm
.Lloop:
    flw  ft0, 0(a0)
    flw  ft1, 0(t1)
    fadd.s ft0, ft0, ft1
    fsw  ft0, 0(a1)

    flw  ft0, 4(a0)             # ← 偏移变成 4
    flw  ft1, 4(t1)
    fadd.s ft0, ft0, ft1
    fsw  ft0, 4(a1)

    flw  ft0, 8(a0)             # ← 偏移变成 8
    flw  ft1, 8(t1)
    fadd.s ft0, ft0, ft1
    fsw  ft0, 8(a1)

    flw  ft0, 12(a0)            # ← 偏移变成 12
    flw  ft1, 12(t1)
    fadd.s ft0, ft0, ft1
    fsw  ft0, 12(a1)

    addi a0, a0, 16             # 指针一次走 4 个元素
    addi t1, t1, 16
    addi a1, a1, 16
    addi a2, a2, -4
    bnez a2, .Lloop
```

**每元素**：`(4×4 + 5) ÷ 4 = 5.25` 条，从 9 条降下来。

**代码里怎么写**（`loopgen.py` 加一个函数）：

```python
def unrolled_body(dtype, unroll: int) -> list[str]:
    """生成展开 unroll 次的循环体（不含指针前进和计数）。"""
    out = []
    for k in range(unroll):
        off = k * 4                       # 每个元素 4 字节
        out += [
            f'    {dtype.load}  ft0, {off}(a0)',
            f'    {dtype.load}  ft1, {off}(t1)',
            f'    {dtype.add} ft0, ft0, ft1',
            f'    {dtype.store} ft0, {off}(a1)',
        ]
    return out
```

**`unroll` 取多大**：不是越大越好。

原因：循环体越长，占的**指令缓存行**越多（一行 64 字节，约 16 条指令），
而每行未命中要付 15 的代价。展开 8 次时循环体约 37 条 ≈ 2.5 行；
展开 16 次约 69 条 ≈ 4.3 行——**省下的循环开销追不回多付的取指代价**。

**怎么定**：扫。写个小脚本，`unroll` 从 1 到 16 各生成一份、各跑一遍 `eval.py`，
取总 cost 最小的。比赛1 量出来的甜点是 **8**。

> ⚠️ **这个值是绑在当前缓存几何上的。** 换指令集、换机器，都要重量。
> 不要把它当成常数抄进代码。

### 5.2 技巧二：让偏移自己说话

第 5.1 节的展开里，`flw ft0, 4(a0)` 用的是**立即数偏移**——地址直接写在指令里，
不用另外维护指针。

**能省什么**：看 add 的循环体，B 的指针 `t1` 需要单独一个寄存器 + 每轮一条 `addi`。
如果 `N` 满足 `N*4 + 最大偏移 ≤ 2047`（第 2.2 节的 `imm_max`），
B 就可以直接用 `a0` 加偏移寻址：

```asm
    flw  ft1, 4096(a0)      # B[i]，不额外占寄存器
```

**省下一个寄存器 + 每轮一条 `addi`。**

**代价**：这需要**知道 N**。而平台不给——见下一章。

### ✅ 检查点

1. add 展开 8 次后，每元素多少条指令？
2. 为什么 `unroll` 不能无限大？
3. `flw ft1, 4096(a0)` 里的 4096 是什么意思？

答案：1) `(8×4 + 5) ÷ 8 = 4.625` 条。2) 循环体占的指令缓存行会变多，
每行未命中付 15，超过省下的循环开销就亏了。3) 从 `a0` 往后 4096 字节，
也就是第 1024 个 float32（因为 4096÷4=1024）。

---

## 第 6 章 · 变快（二）：需要知道规模的优化

### 6.1 平台不告诉你 N——这句话的分量

第 0.6 节说过：题面只公布区间。**这砍掉了整整一类优化。**

原本最有效的路子是：**按 N 生成 10 份特化代码，入口用跳转表分发**。
比赛1 就是这么做的。现在不知道 N，这条路怎么走？

先想清楚：**N 在生成的代码里只以两种身份出现**：

| 身份 | 例子 | 消掉它的代价 |
|---|---|---|
| **立即数** | `flw ft1, 4096(a0)`、展开体的 `4(a0)` | 换成独立指针，每轮多一条 `addi` |
| **循环次数** | `addi a2, a2, -8` 里的 8、循环总共跑几轮 | 换成运行时计数，多一两次取指 |

**把这两种身份都消掉，剩下的就是"通用实现"**——也就是你第 3 章写的那份。
所以"不知道 N"的代价，就等于"从特化版退回通用版"的代价。

修好这个代价，只有三条路：

| 路 | 什么时候能用 | 代价 |
|---|---|---|
| **① 全区间覆盖** | 区间窄。matmul 是 `N ∈ [4, 64]`，**只有 61 个整数** | **几乎为零**——见下面的关键性质 |
| **② 按展开粒度取余数类** | 区间宽（`[64, 4096]`）。按 `N % unroll` 分 `unroll` 类，每类一份代码，轮数是运行时值 | add/reducesum 只差 ~3%；matmul 要多几个行指针 |
| **③ 只保持通用实现** | 拿不准 | 就是你现在这份，放弃这部分收益 |

### 6.2 一条关键性质：没执行到的代码不收费

回到第 0.5 节的 cost 式子：

```
cost = 执行到的指令数 + 15 × (执行期间遇到的未命中)
```

**指令未命中只统计"实际取到过的指令行"。** 你写进文件里但没被执行的分支，
一次代价都不花。

**所以"全区间覆盖"是免费的**：对 matmul 把 N=4 到 N=64 各生成一份（61 份），
入口按 `a2` 分发过去——只多付"读跳转表那一条数据未命中"。

对 add/reducesum 区间太宽（4000 多个整数），用路 ② 更合适。

**这就是为什么先问"区间有多宽"**，再决定用哪条路。

### 6.3 寄存器分块（matmul 的指令数大头）

现在看 matmul 的 `8.17 条/MAC` 能不能降。

**问题**：看第 3.3 节的 `.Lk` 循环——每做一次乘加（MAC），要付：

```
    flw  ft1, 0(a3)        读 A
    flw  ft2, 0(a4)        读 B
    fmul.s ft1, ft1, ft2   乘
    fadd.s ft0, ft0, ft1   加
    addi a3, a3, 4         A 指针 +1
    add  a4, a4, t6        B 指针 +1 行
    addi a5, a5, 1         k++
    bne  a5, a2, .Lk       判断
```

**8 条里只有 2 条是真正的乘法相关**，其余 6 条是"取数 + 维护指针"。

**办法**：一次算一小块。比如一次算 4 行 × 4 列（记作 4×4 分块）：

- 载入 A 的 **4 个**值（`A[i..i+3][k]`）
- 载入 B 的 **4 个**值（`B[k][j..j+3]`）
- 做 **16 次**乘加

这样"取数 + 维护指针"的 6 条被 **16 次乘加**分摊：

```
每条 MAC 的指令数 = 2 + (4 + 4) / (4 × 4) = 2.5
```

**从 8.17 降到 2.5**——这是 matmul 最大的一笔收益。

**分块多大**：受**寄存器数量**限制。

```
累加器数(MR×NR) + 保留的 B 值(NR) + 指针等非数据寄存器 ≤ 可用的数据寄存器数
```

**⚠️ 浮点题的约束和整数题不一样**：浮点的累加器放在**浮点寄存器组**
（f0–f31，32 个），指针放在**整数寄存器组**（约 23 个）。**两组独立**，
所以浮点能放更大的块。**从 `TargetDesc` 取容量，不要写死一个数。**

### 6.4 做完之后量一次未命中（这一步最值钱）

**先看一个反例。** 比赛1 那份分块实现实测：

| N | 三个矩阵共 | 强制未命中 | 实测未命中 |
|---:|---:|---:|---:|
| 48 | 27KB | 432 | **451**（≈ 强制，已经最优） |
| 64 | 48KB | 768 | **3634**（强制的 4.7 倍） |

N=64 时三个矩阵 48KB，**装不下 32KB 的一级缓存**，于是反复去慢内存拿。

**但是**——你现在这份 O0 通用实现的实测是：

| N | 强制未命中 | **O0 通用循环的实测未命中** |
|---:|---:|---:|
| 64 | 768 | **773** |

**773 ≈ 768，已经接近最优。没有洞可补。**

**这说明什么**：那个"未命中暴涨"是**做寄存器分块挖出来的**，
不是通用循环本来就有的。分块之后工作集变大、复用模式变了，才会开始抖。

**所以顺序是**：做 6.3 的寄存器分块 → **再量一次**未命中 →
涨过 768 了才去考虑缓存分块；没涨就跳过。

> **这一条是本文档里最值钱的经验**：同一个优化技术，在一种实现下是必需的，
> 在另一种实现下是无用功。**判据不是"这个技术好不好"，是"我的未命中比强制值高多少"。**

### ✅ 检查点

1. matmul 区间是 `[4, 64]`，全区间覆盖要生成多少份？为什么说这几乎免费？
2. 4×4 分块之后每条 MAC 多少指令？
3. 为什么"做完分块要再量一次未命中"？

答案：1) 61 份。因为没执行到的代码不计入 cost。
2) `2 + (4+4)/16 = 2.5` 条。3) 因为分块会改变工作集大小，可能让缓存开始抖动
——而这不是"优化不够"，是"优化引入了新问题"。

---

## 第 7 章 · 加一道新题

### 7.1 同族题（最省事）

假设要加 `abs-fp32`：`out[i] = |x[i]|`。

1. 写 `bodies/abs.py`，照 `bodies/reducesum.py` 的结构，把叶子换成"取绝对值"；
2. 在 `bodies/__init__.py` 的 `BODIES` 和 `PROBLEMS` 各加一行；
3. 跑区间内逐点正确性 + 第 8 章的三关。

**不用碰 `target.py` / `dtypes.py` / `pipeline.py`。** 约 20 行。

### 7.2 需要新循环形态的题（内测比赛3 的 fwht）

fwht（哈达玛变换）是**蝶形**结构：`log₂N` 个阶段，每个阶段把数据按某个跨步成对加减。

| 改哪 | 做什么 |
|---|---|
| `loopgen.py` | 加一个 `butterfly_body(...)`，和 `unrolled_body` 并列 |
| `bodies/fwht.py` | 声明阶段数 = log₂N、每阶段的跨步怎么变 |
| `bodies/__init__.py` | 登记 |

**关键**：**加减本身仍然从 `dtype` 取**——所以 fp32 版本是白送的。
你在 `bodies/fwht.py` 里写的是"这里要做一次加法"，不是 `fadd.s`。

### 7.3 形状在运行时的题（内测比赛3 的 winograd / spmm）

这两题的形状**不在编译期**，而是写在输入张量的开头几个 int32 里。所以：

| 改哪 | 做什么 |
|---|---|
| `bodies/winograd.py` | 入口先读头部，把形状读进寄存器 |
| `loopgen.py` | 加"步长用寄存器而非立即数"的循环形态 |

**这类题没有现成的路。** 实用建议：**先把 7.1 / 7.2 走通再碰它**。

---

## 第 8 章 · 提交前必须过的三关

| 关 | 命令 | 不通过意味着 |
|---|---|---|
| **正确** | `/tmp/eval.py <你的.s> <题名> riscv-ai-2` | 有数据点算错。浮点题是**容差**比对（`rtol=1e-4, atol=1e-3`），不是逐位 |
| **合规** | 同上（评测器含指令集检查） | 产出了 2 字节指令 → **整题 0 分** |
| **不比原来慢** | 同上，逐点对比上一次的 cost | **任何一步都不许让某个数据点变差** |

⚠️ **别只用 `measure.py`**：它和平台同链路，但**不跑指令集检查**。
它测出来的好数字，线上可能直接 0 分。提交前一律用评测器复核。

**自查清单**（改完逐条对）：

- [ ] `grep -rnE '\b(flw|fsw|lw|sw|fadd\.s|fmul\.s)\b' scratchv/backend/kernels/bodies/` 无输出
- [ ] 寄存器容量从 `TargetDesc` 取，没有写死的数字
- [ ] `unroll` 这类实测量，换 target 后重新量过
- [ ] 现有全部题的产物**逐字节不变**（除非你故意改了那一道）
- [ ] 三题全过

---

## 附录 A · 术语表

| 词 | 意思 |
|---|---|
| **`.s` 文件** | 汇编源文件。平台要的提交就是它 |
| **`cnn_entry`** | 你必须定义的入口符号。平台从这里开始执行你的代码 |
| **`a0` / `a1` / `a2`** | 平台传给你的三个参数：输入首址 / 输出首址 / 规模 N |
| **寄存器** | CPU 内部的小格子，比内存快得多。整数用 `x0`-`x31`（别名 `a0`-`a7`、`t0`-`t6`、`s0`-`s11`），浮点用 `f0`-`f31` |
| **标签** | 以冒号结尾的行，给一个位置起名，供跳转使用。`.L` 开头的是局部标签 |
| **立即数** | 直接写在指令里的常数，如 `addi a0, a0, 4` 里的 4。范围 −2048..2047 |
| **立即数偏移** | 地址写成 `偏移(基址寄存器)`，如 `flw ft0, 8(a0)` = 从 `a0+8` 读 |
| **MAC** | 一次乘加（multiply-accumulate）。矩阵乘的基本操作单位 |
| **行主序** | 矩阵按"一行接一行"存进内存 |
| **ABI** | 函数调用约定：参数放哪、返回值放哪、哪些寄存器不能乱动 |
| **指令数** | 你的程序**执行过**的指令条数（不是写了多少条） |
| **缓存未命中** | 要的数据/指令不在快速缓存里，得去慢内存取。一次算 15 条指令的代价 |
| **L1 / 一级缓存** | 最快的那层缓存。本题配置：32KB，4 路，每行 64 字节 |
| **强制未命中** | 数据第一次被访问时**必然**发生的未命中，无法避免。是"理论下限" |
| **cost** | `指令数 + 15 × 未命中总数`。平台衡量你贵不贵的唯一标准 |
| **容差比对** | 浮点结果不要求逐位相同，只要落在 `rtol`/`atol` 范围内 |
| **Q16.16** | 定点格式：用 int32 表示"真实值 × 65536"，高 16 位整数部分、低 16 位小数部分 |

---

## 附录 B · 完整代码清单

| 文件 | 内容 | 参考章节 |
|---|---|---|
| `scratchv/backend/kernels/target.py` | `TargetDesc` + `TARGETS`（rv32im / rv32imf） | 2.2 |
| `scratchv/backend/kernels/dtypes.py` | `DtypePolicy` + `POLICIES`（q16 / f32）+ `zero_acc` | 2.3 |
| `scratchv/backend/kernels/loopgen.py` | `prologue` / `epilogue`（+ 第 5 章的 `unrolled_body`） | 3.1、5.1 |
| `scratchv/backend/kernels/bodies/add.py` | add 的语义 | 3.2 |
| `scratchv/backend/kernels/bodies/reducesum.py` | reducesum 的语义 | 3.4 |
| `scratchv/backend/kernels/bodies/matmul.py` | matmul 的语义 | 3.3 |
| `scratchv/backend/kernels/bodies/__init__.py` | `BODIES` + `PROBLEMS`（题册） | 3.5 |
| `scratchv/backend/kernels/pipeline.py` | `build_program` | 3.5 |
| `scratchv/backend/kernels/__main__.py` | 命令行 | 3.5 |
| `/tmp/eval.py` | 本地评测脚本（本文档多处用到） | 1.4 |

---

## 附录 C · 排查手册

| 症状 | 最可能的原因 | 怎么确认 |
|---|---|---|
| `该题暂未开放评测` | 没传场次。`-fp32` 的题属于 `riscv-ai-2` | 评测命令末尾加 `riscv-ai-2` |
| 整题 0 分，说明里带 `isa_violation` | 产出了 2 字节指令 | 检查文件开头有没有 `.option norvc` |
| `越界写：…的保护区被破坏` | 循环多写了一轮，或偏移算错 | 检查循环结束条件；注意 `sp` 指向 workspace 顶端，栈向下长 |
| `数值不符`，且**所有**数据点都错 | 大概率是寄存器撞车（第 3.3 节的坑） | 逐个检查你在循环里用到的每个寄存器，"它此刻装的是什么" |
| `数值不符`，只有浮点题错 | 用了逐位判据。浮点要用容差 | 平台 spec 里 `comparison='tolerance'` |
| `runtime_error` + 段错误 | 地址算错（如把字节数当元素个数） | 每个元素 4 字节——索引要乘 4 |
| 本地好、线上崩 | 只跑了 `measure.py`，没跑评测器 | 用 `/tmp/eval.py` 复核 |
| 改了 target，别的题跟着坏 | 助记符漏进了 `bodies/` | 跑第 8 章的自查 grep |
