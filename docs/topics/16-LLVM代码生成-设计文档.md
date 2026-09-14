# 课题16 LLVM 代码生成后端（库路径）技术设计文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/backend/llvm_codegen.py`（库路径 LLVM IR 生成器）、`tests/test_llvm_codegen*.py`  
> 功能范围：库路径 `--backend llvm` 的正确性修复与算子补全——SSA 唯一命名、类型/常量合法化、`_emit_for` CFG 修复、控制流 `br_if` 修复、目标 triple 可配置、8 个 NN 算子（conv/gemm/matmul/dot/maxpool/softmax/gelu/sigmoid）的真实循环生成；验收工具 `llvm-as`  
> 状态说明：本文档描述**目标实现**。当前 `llvm_codegen.py` 存在下述已实测缺陷，文档给出修复设计；实现前代码未修改。  

---

## 一、功能介绍

### 1.1 功能概述

LLVM 代码生成后端负责把 ScratchV IR（`Program`/`Function`/`BasicBlock`/`Instruction`）翻译为 LLVM IR 文本（`.ll`），供 `llvm-as`、`opt`、`llc`、`lli` 使用。项目有两条 LLVM 路径：

| | 路径 A（库路径，本课题） | 路径 B（Standalone，不修改） |
|---|---|---|
| 文件 | `scratchv/backend/llvm_codegen.py` | `scratchv/standalone/onnx_to_llvm_standalone.py` |
| 输入 | ScratchV IR `Program` | ONNXModel（手工解析） |
| 算子状态 | conv/softmax/dot 等为占位实现，数值错误 | float32 完整循环生成 |
| 目标 | 正确、可汇编、结构可断言 | 完整可执行 CNN |

#### 当前缺陷清单（已在 LLVM 10 / `llvm-as` 下实测复现）

| 编号 | 缺陷 | 触发输入 | 实测结果 |
|------|------|----------|----------|
| P1 | GELU 重复定义同名 SSA | `y = gelu(x)` | `multiple definition of local value named 'x3_2'` |
| P2 | Sigmoid 重复定义同名 SSA | `sigmoid` 算子 | `multiple definition of local value named 'v_1_1'` |
| P3 | for 循环变量从未定义 | `for i = 0, 3` + 循环体引用 `i` | `use of undefined value '%v_1_1'` |
| P4 | 嵌套 for 标签重复/悬空 | 双重 `for` | 标签 `loop_exit_12` 重复定义且 `%loop_exit_4` 未定义；`_loop_context` 单槽被内层覆盖 |
| P5 | 整数常量走浮点指令 | `load_const(3, INT32)` | `%v = fadd i32 3, 0.0` → `floating point constant invalid for type` |
| P6 | 十进制浮点字面量不可表示 | ONNX initializer 如 `-6.987718e-02` | LLVM 10 要求 float 十进制常量**精确可表示**；`fadd float -6.987718e-02, 0.0` 报错 |
| P7 | `br_if` 条件类型错误 | `ExtendedDSLParser` 的 `if/while` | `br i1 %a`（`%a: float`）→ `'%a' defined with type 'float' but expected 'i1'`；`cmp_op` 属性被忽略 |
| P8 | 6 个张量算子占位、数值错误 | conv/gemm/matmul/dot/maxpool/softmax | softmax 只发 `expf(x)`；conv 发 `fadd 0.0, 0.0`；maxpool/reshape 直通；matmul/dot/gemm 发标量乘——**无循环、无 GEP、无 MAC** |
| P9 | 目标 triple 硬编码 | 模块头 | `target triple = "riscv64-unknown-elf"` 对所有用途写死，宿主 `lli`/交叉目标不可配置 |

#### 修复后期望能力

- 任意由 `DSLParser`/`ExtendedDSLParser`/`ONNXParser` 产生的、**被后端接受**的 IR，经 `LLVMCodegen.emit()` 输出后 `llvm-as` 全部通过；明确不支持或语义无法保证的输入（见 2.4.8 与 2.6）抛 `LLVMCodegenError` 而不是输出静默错误或非法 IR。
- 8 个算子的 IR 具有与 standalone 路径一致的循环嵌套结构、GEP 地址计算、浮点 MAC，可被 `lli`/`opt` 进一步消费。
- 每个 SSA 名字全函数唯一、每个标签唯一定义、每个基本块以终止指令结束。
- **DSL 变量语义为静态 SSA**：DSL 前端每次赋值产生一个新的 IR `Value`（`dsl_parser.py:114` 的 `_vars[name] = result`），IR 无 phi；循环携带累加、`while` 条件变量重赋值、`if/else` 合流等跨迭代/跨分支的变量语义**不属于本课题范围**（见 §三 用例 2 与 §5.7）。

### 1.2 设计目标

- **正确可汇编**：`llvm-as file.ll -o /dev/null` 零错误是硬门槛，优先于性能与可读性。
- **结构可对齐**：循环结构、索引公式、GEP 形式、浮点常量编码与 `onnx_to_llvm_standalone.py`（下称 standalone）保持一致，便于两条路径对比。
- **嵌套安全**：for/if/while 任意深度嵌套，标签与 SSA 名无冲突。
- **数值可验证**：小规模张量场景可用 `lli` 执行并断言数值（整数与 0.5 等可精确表示值）。
- **改动收敛**：只改库路径与其测试；standalone 不动；不引入 ScratchV IR→LLVM 的高级优化（不做 mem2reg、循环展开、向量化）。

---

## 二、设计规范

### 2.1 总体架构

`LLVMCodegen` 采用**单趟文本生成**：遍历 `Program.functions`，逐函数遍历 `BasicBlock`，逐指令分派到 `_emit_<opcode>`。新增/重构三类基础设施：

1. **命名器**（`SSANamer`）：寄存器与标签分开计数，保证唯一性。
2. **控制流状态机**：显式跟踪“当前块是否已终止”，所有标签经由 `_start_block()` 打开。
3. **张量存储约定**：IR `Value` 携带 `shape`，非空即视为指针（`float*` 等）；标量参与张量算子时溢写（spill）到 `alloca`，退化为 1 元素张量。

不改变 `emit()`/`save()` 的调用契约：`scratchv/compiler.py:388-390` 的 `LLVMCodegen(program).emit()` 继续工作。

### 2.2 类型规则

| IR dtype | 标量 LLVM 类型 | 指针 LLVM 类型 | 常量写法 | 允许的指令族 |
|----------|----------------|----------------|----------|--------------|
| `FLOAT32` | `float` | `float*` | `0.0`/`1.0`/`-1.0` 或 64 位十六进制（见下） | `fadd/fsub/fmul/fdiv/fneg/fcmp/call @expf @tanhf` |
| `FLOAT64` | `double` | `double*` | 同上 | 同上（`@exp @tanh`） |
| `INT32` | `i32` | `i32*` | 十进制整数（如 `3`、`-1`） | `add/sub/mul/sdiv/srem/icmp` |
| `INT64` | `i64` | `i64*` | 十进制整数 | 同上 |

约束规则：

- **不隐式转换**：二元指令两侧类型必须一致；索引、循环计数、维度一律 `i32`。
- **混型算术显式转换**：当操作数类型与结果类型不一致时插入转换指令——int→float 用 `sitofp <ity> v to <fty>`，float→int 用 `fptosi <fty> v to <ity>`（结果类型为整型时）；常量直接按结果类型格式化，不生成转换。例：`s = add(s, i)`（`s: float`、`i: i32`）⇒ `%if = sitofp i32 %i to float` + `%r = fadd float %s, %if`。该规则覆盖 DSL `for` 循环体把循环变量（i32）与 float 变量混用的常见写法。
- **张量指针判定**：`Value.shape != ()` 或该值由 `OpCode.ALLOCA` 定义 ⇒ 类型为“元素类型 + `*`”。函数参数、返回值、算子操作数/结果统一遵循此规则。
- **整数常量禁止浮点算子**：`load_const` 目标为整型时，生成 `%r = add i32 0, <imm>`（或直接把常量作为使用点立即数），绝不生成 `fadd i32 ...`。
- **浮点常量必须精确可表示**：LLVM IR 十进制浮点常量必须能被目标类型精确表示（LLVM 10 校验严格）。统一使用 standalone 的编码算法：
  - 先经 `struct.pack("<f", v)` 舍入到 float32；
  - 再取该值的 IEEE-754 双精度位型，输出 `0x%016X`；
  - `0.0`/`1.0`/`-1.0` 允许直接十进制短写。
- **`i1` 仅用于条件**：`br i1`、`select i1`；比较指令 `fcmp/icmp` 产生 `i1`，不得把 `float` 直接当条件。

### 2.3 SSA 唯一命名规则

- 所有局部寄存器名由 `SSANamer.fresh(hint)` 产生，格式 `%<sanitized>_<n>`，`n` 在**整个函数内单调递增、永不复用**；函数内与函数间均不依赖 LLVM 自动改名。
- 寄存器计数器与标签计数器**分离**（现有实现共用 `_block_counter`，是 P1/P2 的根因之一）。
- 合法字符集：`[A-Za-z0-9_.]`，首字符必须为字母/`_`；`sanitize()` 将非法字符替换为 `_`，空串/数字开头加前缀 `v_`。
- **一次定义**：同一 SSA 名在函数内只允许一次 `= ...` 定义（参数行除外）。
  - `_dest(instr)` 幂等：同一 IR `Value` 重复出现时返回已绑定引用，不再生成新定义；
  - 多指令展开的算子（gelu/sigmoid）**每一步都用 fresh 中间名**，最终结果单独用 fresh 名，禁止“复用 dst 名写三行”（P2 根因）。
- **循环变量**：IR 循环变量 `Value` 不直接作为 SSA 定义，而是在 header 中 `load` 一次得到 `%<iv>_ld_<n>`，该 load 及其结果支配 body 与 exit；循环体与循环后的引用统一绑定到它（P3 根因修复）。
- 标签名由 `SSANamer.fresh_label(hint)` 产生（无 `%` 前缀），进入 `defined_labels` 集合；重复定义时追加后缀，绝不输出两个同名标签。
- 中间名 hint 约定（增强可读性）：

| 算子 | 寄存器 hint | 标签 hint |
|------|-------------|-----------|
| gelu | `gelu_t1, gelu_x3, gelu_inner, gelu_tanh, gelu_p1, gelu_hx` | — |
| sigmoid | `sig_neg, sig_exp, sig_den, sig_out` | — |
| dot | `dot_i, dot_acc, dot_prod` | `dot_i` |
| matmul | `mm_i, mm_j, mm_k, mm_acc` | `mm_i/mm_j/mm_k` |
| gemm | `gemm_i, gemm_j, gemm_k, gemm_acc` | `gemm_i/j/k` |
| conv | `conv_oc/oh/ow/ic/kh/kw, conv_acc, conv_mac, conv_skip` | `conv_oc/.../conv_kw` |
| maxpool | `mp_c/oh/ow/kh/kw, mp_max` | `mp_c/.../mp_kw` |
| softmax | `sm_i1/i2/i3, sm_max, sm_sum, sm_e` | `sm_max/sm_sum/sm_div` |

### 2.4 控制流合法性规则

LLVM 基本块要求：**每个块恰好一条终止指令（`ret`/`br`/`br i1`）且在块尾**；标签唯一定义；所有使用被定义支配。对应实现规则：

1. **块状态机**：字段 `self._terminated: bool`。`_start_block(label)` 的语义是“结束当前块并打开新块”：
   - 若 `not self._terminated`，先发射 `br label %<label>`（显式落空转跳转）；
   - 打印 `<label>:`，将 `_terminated` 置 `False`，登记 `defined_labels`。
   - 函数第一个块（entry）特殊：不打印标签，直接开始发射，避免无意义入口标签。
2. **终止登记**：发射 `ret`/`br`/`br i1` 时置 `_terminated = True`；若已终止又收到普通指令或新标签，先自动开 `unreachable_<n>` 新块，确保任何时刻不产生“终止符后还有指令”的非法文本。
3. **循环规范形**（`_emit_for` 修复核心）：
   ```
   preheader:  iv_ptr = alloca i32, i32 1 ; store i32 <START>, i32* iv_ptr ; br label %<hdr>
   <hdr>:      %iv = load i32, i32* iv_ptr
               %c  = icmp slt i32 %iv, <END>
               br i1 %c, label %<bdy>, label %<ext>
   <bdy>:      ... 循环体 ... (引用 %iv)
   latch:      %cur = load i32, i32* iv_ptr ; %nxt = add i32 %cur, <STEP>
               store i32 %nxt, i32* iv_ptr ; br label %<hdr>
   <ext>:      ... 后续指令 ...
   ```
   与现有实现的两处差异：**preheader 必须跳到 header 而不是 body**（现有 `br label %body` 跳过条件判断）；**循环上下文用栈**（现有单槽 `_loop_context` 导致嵌套覆盖与标签重复）。`START`/`STEP` 取自 `attrs["start"]`/`attrs["step"]`（缺省 0/1）；`END` 为 `attrs["end"]`。
4. **循环栈**：`self._loop_stack: list[LoopContext]`，FOR 压入、ENDFOR 弹出；无匹配 ENDFOR 时报错（`LLVMCodegenError`）而非生成残缺 IR。
5. **`br_if` 双形态**：
   - 形态 A（库 IRBuilder）：`operands=[i1 条件]`，`target="T,F"` → 直接 `br i1`；
   - 形态 B（ExtendedDSLParser）：`operands=[lhs, rhs]`，`attrs["cmp_op"] ∈ {==,!=,<,<=,>,>=}` → 先按操作数类型发射 `fcmp`（浮点）或 `icmp`（整型），再 `br i1`；
   - 比较谓词映射：`==→oeq/eq`、`!=→one/ne`、`<→olt/slt`、`<=→ole/sle`、`>→ogt/sgt`、`>=→oge/sge`。
6. **禁止跨块裸值**：不生成 `phi`；所有跨块传递走 `alloca/load/store`（memory-based SSA），天然满足支配关系，也与 standalone 现有写法一致。
7. **alloca 位置**：循环计数器、累加器、张量缓冲区等统一登记到函数入口 prologue（在第一个 IR 块发射前输出），避免运行时在循环内分配栈导致栈膨胀。
8. **fail-loud 边界（禁止静默错误）**：下列输入不再输出静默错误或非法 IR，而是抛 `LLVMCodegenError`：
   - 函数内 `ret <value>` 与 `ret void` 混用（`_validate_return_types`）；
   - 无 handler 的 opcode（transpose/concat 等，`_emit_unsupported`）；
   - `while` 循环条件操作数在循环体（可达区块、无 `ret`）内从未被定义——DSL 静态 SSA 下该循环不可能退出（`_check_unbounded_loops`）；
   - 张量算子（dot/matmul/gemm/conv/maxpool/softmax）所需的元素数大于操作数实际元素数，或操作数为标量而所需元素数 > 1（`_require_elements`）；
   - 元素级算子（add/sub/mul/div/neg/exp/relu/gelu/sigmoid）的目标带 shape 时按逐元素语义生成（`_emit_map`）；操作数与目标元素数不一致时抛错；
   - softmax 非 `axis=-1` 输入（`_emit_softmax`）。

### 2.5 算子 IR 生成规范

通用地址/索引规则：所有偏移用 `i32` 算术手算（`mul`/`add`），再经单索引 GEP `getelementptr <ty>, <ty>* base, i32 off` 取元素地址。循环变量名与 standalone 对齐（NCHW 布局）。

#### 2.5.1 dot（点积）

维度：`attrs["length"]`（DSL `len:`/`length:` 键均接受），缺省 1。  
循环嵌套：1 层。

```
%acc = alloca float, i32 1 ; store float 0.0
i ∈ [0, length):
    %ap = gep a, i ; %av = load
    %bp = gep b, i ; %bv = load
    %pr = fmul %av, %bv
    %acc_v = load %acc ; %acc_n = fadd %acc_v, %pr ; store %acc
result = load %acc
```

#### 2.5.2 matmul（矩阵乘）

维度：`attrs` 的 `m/n/k`（`IRBuilder.matmul(a,b,m,n,k)`），或由 `a.shape=(m,k)`、`b.shape=(k,n)` 推断，缺省 1。  
循环嵌套：3 层（i、j、k），行主序 C[m,n] = A[m,k]·B[k,n]。

```
for i ∈ [0,m):
  for j ∈ [0,n):
    acc = 0.0
    for kk ∈ [0,k):
      av = A[i*k + kk]
      bv = B[kk*n + j]
      acc += av * bv
    C[i*n + j] = acc
```

#### 2.5.3 gemm（通用矩阵乘 + 偏置）

维度：M/K 取自 A 的 shape，N 取自 W 的 shape（`trans_b` 时 `W.shape[0]`，否则 `W.shape[1]`）或结果 shape，缺省 1。  
`trans_b` 属性名兼容 `trans_b`（IRBuilder）与 `transB`（ONNXParser）。  
循环嵌套：3 层，acc 初始为 `bias[j]`。

```
for i ∈ [0,M):
  for j ∈ [0,N):
    acc = bias[j]
    for k ∈ [0,K):
      av = A[i*K + k]
      wv = W[trans_b ? (j*K + k) : (k*N + j)]
      acc += av * wv
    C[i*N + j] = acc
```

#### 2.5.4 conv（2D 卷积，NCHW）

维度：输入 `x.shape=(C,H,W)`（或含 N 的 `(N,C,H,W)`，取后三维）；权重 `w.shape=(Cout,Cin,K,K)`；步长/填充/输出通道取自 `attrs`，兼容两种键名：`stride/strides`、`padding/pads`、`kernel_size/kernel_shape`、`out_channels`。输出尺寸 `Ho=(H+2P-K)/S+1`、`Wo` 同理；shape 缺失时退化为 `C=H=W=1`。bias 操作数缺失时按常量 `0.0` 处理。  
循环嵌套：6 层（oc、oh、ow、ic、kh、kw），kh/kw 内做边界判断。

```
for oc ∈ [0,Cout):
  w_oc = oc*Cin*K*K ; out_oc = oc*Ho*Wo
  for oh ∈ [0,Ho):
    ohS = oh*S
    for ow ∈ [0,Wo):
      acc = bias[oc] ; owS = ow*S
      for ic ∈ [0,Cin):
        in_ic = ic*H*W ; w_ic = ic*K*K
        for kh ∈ [0,K):
          ih = ohS + kh - P ; ok_h = (ih >= 0) & (ih < H) ; in_h = ih*W ; w_kh = kh*K
          for kw ∈ [0,K):
            iw = owS + kw - P ; ok = ok_h & (iw >= 0) & (iw < W)
            br i1 ok, label %mac, label %skip
          %mac:
            xv = x[in_ic + in_h + iw]
            wv = W[w_oc + w_ic + w_kh + kw]
            acc += xv * wv ; br label %skip
          %skip:
      out[out_oc + oh*Wo + ow] = acc
```

#### 2.5.5 maxpool（2D 最大池化，NCHW）

维度：输入 `x.shape=(C,H,W)`；`attrs["kernel"]`（兼容 `kernel_shape`）、`attrs["stride"]`（兼容 `strides`）；`Ho=(H-K)/S+1`。无 padding。  
循环嵌套：5 层（c、oh、ow、kh、kw）。

```
for c ∈ [0,C):
  for oh ∈ [0,Ho):
    for ow ∈ [0,Wo):
      m = -3.4e38
      for kh ∈ [0,K):
        for kw ∈ [0,K):
          ih = oh*S + kh ; iw = ow*S + kw
          v = x[c*H*W + ih*W + iw]
          m = select(fcmp ogt v, m, v, m)
      out[c*Ho*Wo + oh*Wo + ow] = m
```

#### 2.5.6 softmax（axis=-1，数值稳定三趟）

维度：输入 `shape[-1]`（无 shape 时取 `attrs["length"]`，缺省 1）；rank>1 时按 `prod(shape[:-1])` 行逐行处理，输出缓冲大小为 `prod(shape)`。非 `axis=-1` 抛 `LLVMCodegenError`。  
循环嵌套：row 循环（rank>1 时）内套 3 个顺序单层循环（max、sum-exp、div），均减最大值以保证稳定；分母在第三趟重算 `exp`，不额外开临时数组。

```
for row ∈ [0, prod(shape[:-1])):
    base = row * shape[-1]
    pass1 (max):  m = -3.4e38 ; for i: m = max(m, x[base+i])
    pass2 (sum):  s = 0.0     ; for i: s += expf(x[base+i] - m)
    pass3 (div):  for i: out[base+i] = expf(x[base+i] - m) / s
```

数值校验点：N=1 时 `exp(0)/exp(0) = 1.0`；N=2 且输入 `[0,0]` 时输出 `[0.5,0.5]`；shape `(2,2)` 且输入全 0 时输出 4 个 `0.5`（2D 回归用例）。

#### 2.5.7 gelu（标量激活）

`GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`，逐步 fresh 命名，`float` 调 `@tanhf`，`double` 调 `@tanh`：

```
%gelu_t1    = fmul <ty> %x, %x
%gelu_x3    = fmul <ty> %gelu_t1, %x
%gelu_in1   = fmul <ty> 0.044715, %gelu_x3
%gelu_in2   = fadd <ty> %gelu_in1, %x
%gelu_inner = fmul <ty> %gelu_in2, 0.7978845608028654
%gelu_tanh  = call <ty> @tanh[f](<ty> %gelu_inner)
%gelu_p1    = fadd <ty> 1.0, %gelu_tanh
%gelu_hx    = fmul <ty> %x, 0.5
%dst        = fmul <ty> %gelu_hx, %gelu_p1     ; 唯一一次定义 dst
```

#### 2.5.8 sigmoid（标量激活）

`Sigmoid(x) = 1 / (1 + e^(−x))`，逐步 fresh 命名，禁止三行复用同一 `%dst`：

```
%sig_neg = fneg <ty> %x
%sig_exp = call <ty> @exp[f](<ty> %sig_neg)
%sig_den = fadd <ty> 1.0, %sig_exp
%dst     = fdiv <ty> 1.0, %sig_den            ; 唯一一次定义 dst
```

### 2.6 张量存储与退化规则

库路径 IR 没有 standalone 的 `MemoryPlan`/workspace，因此约定：

- **指针操作数**：`Value.shape != ()` 或由 `ALLOCA` 定义 ⇒ 该值引用本身就是 `float*`/`double*`/`i32*`。
- **标量退化**：张量算子的某操作数是标量（shape 为空）时，在当前位置 spill 到 `alloca <ty>, i32 1`（hint `spin`），按 1 元素张量使用。**仅当算子所需元素数为 1 时成立**；若所需元素数 > 1（如 `dot(a,b,len:4)` 的 a/b 为标量）则抛 `LLVMCodegenError`，不再按 attrs 维度循环导致越界读（`_require_elements`）。
- **结果缓冲**：
  - `instr.dest` 由 `ALLOCA` 定义 → 直接写该 alloca 指针；
  - `dest.shape` 非空且为元素级算子（add/sub/mul/div/neg/exp/relu/gelu/sigmoid）→ 逐元素循环写入 `_alloc_slot(element_ty, prod(shape))` 缓冲（`_emit_map`）；操作数为标量或单元素张量时广播，其他元素数不匹配则抛错；
  - `dest.shape` 非空且为点积/矩阵类算子 → `_alloc_slot(element_ty, prod(shape))` 得到缓冲并绑定 `dest` 名为该指针；
  - `dest.shape` 为空 → 分配 1 元素缓冲运行循环，最后 `load` 首元素产生标量 `%dst`（1 元素时与完整语义等价）。
- **已知边界**：ONNXParser 的输出 `Value` 可能不含 shape，此时按 1 元素缓冲处理——`llvm-as` 合法性不受影响，但**运行时数值仅在显式 shape 的 IR 上有保证**；完整 ONNX 张量语义属于 standalone 路径职责，不在本课题范围。
- **函数签名**：参数/返回值按 2.2 的指针判定生成 `float*` 等；`ret` 张量时 `ret float* %buf`。
- **循环栈变量与张量缓冲的区别**：前者固定 `alloca i32`，后者大小由 shape/维度乘积决定；均登记进函数入口 prologue。

### 2.7 与 standalone 路径的对齐约定

| 维度 | standalone | 库路径（本设计） | 关系 |
|------|------------|------------------|------|
| 数据布局 | NCHW | NCHW | 一致 |
| 循环结构 | alloca 计数器 + header/latch + `icmp slt` + `br i1` | 同 | 一致 |
| 地址计算 | 单索引 `getelementptr <ty>, <ty>* base, i32 off` | 同 | 一致 |
| MAC | `fmul` + `fadd`，累加器走 alloca | 同 | 一致 |
| max/激活 | `fcmp` + `select` | 同 | 一致 |
| 浮点常量 | `_float_to_llvm_hex`（float32 舍入后取 double 位型） | 同算法 | 一致 |
| 循环嵌套顺序 | conv 6 层 / gemm 3 层 / maxpool 5 层 | 同 | 一致 |
| 激活函数 | 自带 `inline_expf`（无 libm） | 调 libm `@expf/@tanhf` | **有意不同**（库路径无内联序列；不影响 `llvm-as`） |
| 函数布局 | 每算子一个函数 + `@main_graph` | 内联进当前 IR 函数 | **有意不同**（库路径无图调度与 workspace） |
| 张量元数据 | ONNX shape + MemoryPlan | `Value.shape` + `Instruction.attrs` | 语义等价 |
| 属性键名 | `strides/pads/kernel_shape` | 两套键名都接受 | 兼容 |

### 2.8 目标 triple 规则

- 构造函数签名扩展为 `LLVMCodegen(program, target_triple=None)`；`None` 时**省略** `target triple` 行（目标无关 IR，`llvm-as`/`opt`/`lli` 均可用）。
- 显式传入字符串原样输出，例如 `"riscv64-unknown-elf"`（与 standalone 对齐）或 `"x86_64-unknown-linux-gnu"`（宿主执行）。
- 不修改 `compiler.py`/`main.py`：`--backend llvm` 保持现有调用方式，默认得到目标无关 IR；需要 triple 的场景走 Python API。
- 该设计消除 P9；`target triple` 对 `llvm-as` 结果无影响，但影响 `llc`/`lli` 目标选择与宿主执行。

### 2.9 合法/非法 IR 示例

**合法（本设计的目标输出形态）**：

```llvm
; 循环规范形：preheader -> header -> body -> latch -> header，exit 由 header 分出
define float @sum_for(float %x) {
entry:
  %iv_ptr_1 = alloca i32, i32 1
  store i32 0, i32* %iv_ptr_1
  br label %loop_i_hdr_2
loop_i_hdr_2:
  %iv_ld_3 = load i32, i32* %iv_ptr_1
  %cond_4 = icmp slt i32 %iv_ld_3, 3
  br i1 %cond_4, label %loop_i_bdy_5, label %loop_i_ext_6
loop_i_bdy_5:
  %ivf_7 = sitofp i32 %iv_ld_3 to float   ; i32 IV -> float（混型算术显式转换）
  %y_8 = fadd float %x, %ivf_7            ; 循环变量已被 load 定义并支配 body
  %iv_cur_9 = load i32, i32* %iv_ptr_1
  %iv_nxt_10 = add i32 %iv_cur_9, 1
  store i32 %iv_nxt_10, i32* %iv_ptr_1
  br label %loop_i_hdr_2
loop_i_ext_6:
  ret float %y_8
}
```

**非法示例（均为当前实现的真实输出）**：

1. 重复定义 SSA（P1/P2）：
   ```llvm
   %x3_2 = fmul float %x, %x
   %x3_2 = fmul float %x3_2, %x     ; error: multiple definition of local value named 'x3_2'
   ```
2. 循环变量未定义（P3）：
   ```llvm
   %v_2_9 = fadd float %x, %v_1_1   ; error: use of undefined value '%v_1_1'
   ```
3. 跳转跳过 header + 标签重复/悬空（P4）：
   ```llvm
   br label %loop_body_3             ; 语义错误：首轮不做条件判断
   ...
   loop_exit_12:                     ; error: duplicate label（嵌套第二次 ENDFOR 输出）
   br i1 %cond_8, ..., label %loop_exit_4   ; error: use of undefined value '%loop_exit_4'
   ```
4. 类型错误（P5/P7）：
   ```llvm
   %v_1_1 = fadd i32 3, 0.0          ; error: floating point constant invalid for type
   br i1 %a, label %t, label %f      ; error: '%a' defined with type 'float' but expected 'i1'
   ```
5. 浮点常量不可表示（P6）：
   ```llvm
   %v_1_1 = fadd float -6.987718e-02, 0.0   ; error: floating point constant invalid for type
   ; 正确写法：fadd float 0xBFB1E37880000000, 0.0
   ```

---

## 三、测试设计

统一测试骨架（`tests/test_llvm_codegen_llvm_tools.py`）：

```python
LLVM_AS = shutil.which("llvm-as")
LLI = shutil.which("lli")
pytestmark = pytest.mark.skipif(LLVM_AS is None, reason="llvm-as not installed")

def assemble(ll_text: str, tmp_path) -> subprocess.CompletedProcess:
    p = tmp_path / "m.ll"; p.write_text(ll_text)
    return subprocess.run([LLVM_AS, str(p), "-o", "/dev/null"],
                          capture_output=True, text=True)
```

数值验证采用 **harness 拼接法**：被测 IR 函数命名为 `kernel`，测试在其生成的模块文本后拼接一段手写 `define i32 @main()`（用 `alloca/store` 造数组、`call` 调用 kernel、`fcmp oeq`/`br` 比较期望值，成功返回 0），先 `llvm-as` 组装再用 `lli` 执行，`returncode == 0` 即数值通过；`lli` 缺失时 `skip`。

### 测试用例 1：激活链 SSA 唯一性（gelu + sigmoid）

- **输入 IR**（`IRBuilder`）：
  ```python
  b = IRBuilder(); b.new_function("kernel"); b.new_block("entry")
  x = b.make_value("x")                 # float 参数
  g = b.gelu(x); s = b.sigmoid(g); b.ret(s)
  ```
- **预期输出**：`llvm-as` 退出码 0；不得出现 `multiple definition`。
- **关键指令片段**：`fmul float`、`fadd float`、`call float @tanhf`、`fneg float`、`call float @expf`、`fdiv float 1.0`；正则统计每个 `%name =` 恰好一次。
- **数值语义验证**：harness 断言 `gelu(0.0) == 0.0` 且 `sigmoid(0.0) == 0.5`（`fcmp oeq`，两值均可精确表示）；`lli` 返回 0。

### 测试用例 2：嵌套 for 的 CFG 正确性

- **输入 IR**（`DSLParser`）：
  ```python
  program = DSLParser().parse(
      "s = add(s, 0.0)\n"
      "for i = 0, 3\n"
      "for j = 0, 3\n"
      "s = add(s, i)\n"
      "endfor\n"
      "endfor\n"
      "return s")
  ```
- **预期输出**：`llvm-as` 退出码 0；修复后的标签形如 `loop_i_hdr_*` / `loop_i_bdy_*` / `loop_i_ext_*`，每类各出现 2 次（内外层各一），无重复定义。
- **关键指令片段**：首块（entry）以 `br label %loop_i_hdr_*` 结束（**preheader 直接跳 header 而非 body**）、`icmp slt i32`、`br i1`、`add i32`、回边 `br label %loop_i_hdr_*`；两个 IV 各有 `load i32, i32*` 定义，循环体混型加法含 `sitofp i32 ... to float`。
- **数值语义验证（静态 SSA 语义，见 §5.7）**：DSL 每次赋值产生新 `Value`，`s` 在循环体每轮读到的都是循环前的值，故实际返回 `2.0`（最后一次内层迭代的 `0 + i`），**不是** 9.0。回归用例 `test_lli_dsl_nested_for_static_ssa_value` 显式断言 2.0 并标注该限制。
- **正确累加形式**（IR 层 memory-based SSA）：用 `IRBuilder` 显式 `alloca`/`load`/`store` 累加得到 9.0，见用例 3b 与 `test_lli_nested_for_accumulator`。

### 测试用例 3：张量算子结构 + 小规模数值（dot/matmul/gemm/maxpool/conv）

- **输入 IR**：构造 `kernel` 函数，参数为带 shape 的 `float*`（如 `a.shape=(4,)`、`b.shape=(4,)`），分别生成 dot/matmul/gemm/maxpool/conv 五个程序；测试端拼接 harness 分配数组并写入常量：
  - dot：`a=[1,2,3,4]`、`b=[1,1,1,1]` → `10.0`
  - matmul 1×1：`a=2`、`b=3` → `6.0`
  - gemm 1×1：`a=2`、`w=3`、`bias=0.5` → `6.5`
  - maxpool：`x=[[1,2],[3,4]]`，`K=2,S=1` → `4.0`
  - conv 1×1×1：`x=3`、`w=2`、`bias=1`，`K=1,S=1,P=0` → `7.0`
- **预期输出**：五个程序 `llvm-as` 均退出码 0。
- **关键指令片段**：每个算子至少出现 `getelementptr float, float*`（GEP）、`fmul float` + `fadd float`（MAC）、`icmp slt i32` + `br i1`（循环）；dot 3 个标签、matmul/gemm 9 个标签（3 层×3）、maxpool 15 个、conv 18 个（可由正则计数断言，>= 对应下界）。
- **数值语义验证**：harness 断言上表期望值（整数与 0.5 均可精确比较），`lli` 返回 0。

### 测试用例 4：softmax 数值语义（含 1 元素退化）

- **输入 IR**：`kernel(float* %x)`，`x.shape=(2,)`，`dest.shape=(2,)`，调用 `softmax(x, axis:-1)`；再构造 `shape=(1,)` 的版本。
- **预期输出**：`llvm-as` 退出码 0。
- **关键指令片段**：三条独立循环标签（`sm_max_*`、`sm_sum_*`、`sm_div_*`），`fcmp ogt`、`call float @expf`、`fdiv float`；**不得**只有单个 `call @expf`（当前占位实现特征）。
- **数值语义验证**：harness 对 `x=[0,0]` 断言输出 `[0.5,0.5]`；对 `x=[7]` 断言输出 `[1.0]`；`lli` 返回 0。

### 测试矩阵与验收命令

| 用例 | 覆盖缺陷 | 工具 | 断言类型 | 环境缺失行为 |
|------|----------|------|----------|--------------|
| 1 | P1、P2、P6 | llvm-as + lli | 可汇编 + SSA 唯一 + 数值 | skip |
| 2 | P3、P4、P7 | llvm-as + lli | 可汇编 + CFG 形态 + 数值 | skip |
| 3 | P8（5 算子） | llvm-as + lli | 可汇编 + 结构计数 + 数值 | skip |
| 4 | P8（softmax）、P5 | llvm-as + lli | 可汇编 + 结构 + 数值 | skip |
| 回归 | — | pytest | `tests/test_llvm_codegen.py` 原有用例全绿 | — |

```bash
llvm-as tests_out/case1.ll -o /dev/null        # 期望 rc=0
lli /tmp/case3_dot.bc                          # 期望 rc=0（数值断言通过）
python -m pytest tests/test_llvm_codegen.py tests/test_llvm_codegen_llvm_tools.py -v
```

---

## 四、修改模块与实现步骤

### 4.1 涉及文件

| 文件 | 改动 |
|------|------|
| `scratchv/backend/llvm_codegen.py` | 重构生成器（命名器、状态机、类型/常量、8 算子） |
| `tests/test_llvm_codegen.py` | 扩充单元/结构断言（SSA 唯一、标签唯一、常量写法） |
| `tests/test_llvm_codegen_llvm_tools.py`（新增） | `llvm-as`/`lli` 集成测试，工具缺失 skip |

（注：实际路径可能不同；standalone 路径、`compiler.py`、`main.py`、`ir/types.py` 均不改。）

### 4.2 引入命名器并接管所有名字

新增 `SSANamer` 类（`fresh`/`fresh_label`/`sanitize`/`register_definition`），`LLVMCodegen` 持有实例；删除 `_block_counter` 的双重职责（寄存器/标签合一）。`_fresh()` 改为委托命名器；`_start_block()` 登记标签，重复定义自动追加后缀。

### 4.3 类型与常量修复

- `_value_type(val)`：按 `shape`/ALLOCA 来源判定标量或指针；`self._ref_types: dict[str, str]` 记录每个 SSA 引用的 LLVM 类型。
- `_infer_type()` 保留但必须返回合法类型；`_llvm_const_val` 浮点分支改为十六进制（复用 standalone 的 `_float_to_llvm_hex` 算法），整型分支返回十进制且断言不用于浮点算子。
- `_emit_load_const`：按 dest dtype 分派 `fadd <float-ty>`（浮点）或 `add <int-ty> 0, imm`（整型）。
- `_emit_load/_emit_store`：以“值类型”为准构造 `load <ty>, <ty>*` / `store <ty> v, <ty>* p`。
- `_emit_return`：按引用类型生成 `ret float*` 等；函数返回类型推断同步使用 `_value_type`。
- 外部声明保持不变并补充 double 变体（已有 `@exp/@tanh`）。

### 4.4 控制流修复

- 新增 `_start_block(label)`、`_terminate(line)`、`_ensure_terminator()`，字段 `_terminated`、`defined_labels`。
- 重写 `_emit_for/_emit_endfor`：循环规范形（2.4 节），`_loop_stack` 栈，IV 绑定 header load，无匹配 ENDFOR 抛 `LLVMCodegenError`。
- `_emit_br_if` 支持 `cmp_op` 形态 B；`_emit_label` 与 `_emit_block` 统一走 `_start_block`，杜绝重复标签。

### 4.5 张量算子实现

按 2.5 实现 `_emit_dot/_emit_matmul/_emit_gemm/_emit_conv/_emit_maxpool/_emit_softmax`；新增 `_alloc_slot`、`_dest_buffer`、`_ptr_of` 基础设施；删除全部 `; placeholder`/`passthrough` 文本与伪计算。

### 4.6 激活函数修复

重写 `_emit_gelu/_emit_sigmoid`：全部中间值 fresh 命名，dst 仅定义一次；`float/double` 按类型选 `@tanhf/@tanh`、`@expf/@exp`。

### 4.7 triple 可配置

`__init__(self, program, target_triple: str | None = None)`；`emit()` 中 `target_triple is None` 则省略该行。保持位置参数兼容。

### 4.8 测试与集成

按第三章新增/扩充测试；`make test`（`python3 -m pytest tests/`）全绿；`benchmarks/test_benchmark.py::test_codegen_llvm` 仅断言非空输出，需保持通过。

### 4.9 回归与验收

- 全量：`python -m pytest tests/ -q` 无新增失败；
- LLVM：`llvm-as` 对 8 算子样例与 ONNX CNN（`models/graph/cnn.onnx`）输出全通过；
- 数值：`lli` 执行第三章四个用例 harness 全通过（环境缺失 skip 不视为通过，CI 有工具时须真实执行）；
- 不做：standalone 路径、RISC-V 路径、IR 类型定义、编译器驱动改动。

---

## 五、附录

### 5.1 修复后的 GELU + Sigmoid 生成示例（预期）

```llvm
; LLVM IR generated by ScratchV
; ModuleID = "scratchv_module"
; target triple omitted (target-independent)

declare float @expf(float) nounwind readonly
declare float @tanhf(float) nounwind readonly

define float @kernel(float %x) {
; --- entry ---
  %gelu_t1_1 = fmul float %x, %x
  %gelu_x3_2 = fmul float %gelu_t1_1, %x
  %gelu_in1_3 = fmul float 0x3FA6E4E260000000, %gelu_x3_2
  %gelu_in2_4 = fadd float %gelu_in1_3, %x
  %gelu_inner_5 = fmul float %gelu_in2_4, 0x3FE9884540000000
  %gelu_tanh_6 = call float @tanhf(float %gelu_inner_5)
  %gelu_p1_7 = fadd float 1.0, %gelu_tanh_6
  %gelu_hx_8 = fmul float %x, 0.5
  %v_2_9 = fmul float %gelu_hx_8, %gelu_p1_7
  %sig_neg_10 = fneg float %v_2_9
  %sig_exp_11 = call float @expf(float %sig_neg_10)
  %sig_den_12 = fadd float 1.0, %sig_exp_11
  %v_3_13 = fdiv float 1.0, %sig_den_12
  ret float %v_3_13
}
```

### 5.2 修复后的 for 循环示例（预期）

见 2.9 节合法示例；嵌套时两组 `loop_*_hdr/bdy/ext` 各自唯一，内层循环位于外层 body 块内，回边分别指向各自 header。

### 5.3 conv 内层 MAC 片段（预期）

```llvm
%conv_ih_31 = sub i32 %conv_ohS_27, 1
%conv_okh_33 = icmp sge i32 %conv_ih_31, 0
%conv_okh_34 = icmp slt i32 %conv_ih_31, 8
%conv_okh_35 = and i1 %conv_okh_33, %conv_okh_34
...
%conv_xoff_44 = add i32 %conv_in_ic_36, %conv_in_h_43
%conv_xoff_45 = add i32 %conv_xoff_44, %conv_iw_41
%conv_xp_46 = getelementptr float, float* %x, i32 %conv_xoff_45
%conv_xv_47 = load float, float* %conv_xp_46
%conv_wv_50 = load float, float* %conv_wp_49
%conv_pr_51 = fmul float %conv_xv_47, %conv_wv_50
%conv_acc_n_53 = fadd float %conv_acc_v_52, %conv_pr_51
```

### 5.4 softmax 三趟循环骨架（预期）

```llvm
sm_max_hdr_2:
  %sm_i1_ld_3 = load i32, i32* %sm_i1_ptr_1
  %sm_c1_4 = icmp slt i32 %sm_i1_ld_3, 2
  br i1 %sm_c1_4, label %sm_max_bdy_5, label %sm_max_ext_6
sm_max_bdy_5:
  %sm_mv_9 = load float, float* %sm_mp_7
  %sm_cmp_10 = fcmp ogt float %sm_xv_8, %sm_mv_9
  %sm_new_11 = select i1 %sm_cmp_10, float %sm_xv_8, float %sm_mv_9
  store float %sm_new_11, float* %sm_mp_7
  br label %sm_max_hdr_2
sm_max_ext_6:
  ... sum-exp 循环 ... div 循环 ...
```

### 5.5 验证命令速查

```bash
# 汇编合法性（硬门槛）
llvm-as output.ll -o /dev/null
# 优化与反汇编
opt -O3 output.ll -S -o /dev/null
# 宿主执行（数值测试）
llvm-as output.ll -o output.bc && lli output.bc; echo "rc=$?"
# 交叉编译到 RISC-V（需显式 triple）
llc -march=riscv64 output.ll -o output.s
```

### 5.6 参考资料

- 课题文档：`docs/topics/16-LLVM代码生成.md`
- standalone 参考实现：`scratchv/standalone/onnx_to_llvm_standalone.py`（`LLVMIRBuilder`、`LLVMCNNGenerator`）
- LLVM IR 类型/常量规则：LLVM Language Reference（LangRef）
- 设计文档模板：`设计文档模板.md`

### 5.7 已知限制：DSL 变量为静态 SSA 语义（非本课题范围）

DSL 前端（`dsl_parser.py`、`dsl_extended.py`）的 `_vars[name]` 只保存**最近一次赋值**的 `Value`，每次赋值产生一个新的 IR `Value`；IR 无 phi（`types.py` 的 `phi_nodes` 未被任何 pass 使用），因此跨迭代/跨分支的变量语义无法成立。当前行为与回归锁定如下：

| 模式 | 示例 | 实际行为 | 回归测试 |
|------|------|----------|----------|
| for 循环累加 | `s = add(s, i)` | 循环体每轮读循环前的 `s`，嵌套双重循环返回 `2.0`（非 9.0） | `test_lli_dsl_nested_for_static_ssa_value` |
| 循环携带变量 | `for ...: x = add(x, 1.0)` | 返回 `4.0`（非 5.0） | `test_lli_dsl_loop_carried_static_ssa_value` |
| `while` 条件变量重赋值 | `while (i < 10): i = add(i, 1.0)` | header 永远读旧槽 → 不可能退出；后端 `_check_unbounded_loops` 抛 `LLVMCodegenError` | `test_while_reassigned_condition_raises` |
| `if/else` 合流 | 两个分支分别给 `y` 赋值 | 合流后恒读最后解析分支（else）的槽；走 then 时该槽未初始化（未定义值） | `test_dsl_if_else_merge_reads_last_parsed_branch` |

**正确的累加/循环携带写法是 IR 层的显式 `alloca`/`load`/`store`**（用例 3b、`test_lli_nested_for_accumulator` 得到 9.0）。把 DSL 可变变量降级为 alloca/load/store 属于"DSL 变量语义"独立课题，不在本分支范围内（评审 §4.1）；本分支只做上述 fail-loud 守卫与文档/回归锁定，禁止静默错误。
