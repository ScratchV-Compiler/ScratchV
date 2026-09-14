# 课题 29：SIMD 向量化设计文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/ir/types.py`（向量 OpCode）、`scratchv/ir/builder.py`（向量构造 API）、`scratchv/optimizer/vectorize.py`（新增 Vectorizer）、`scratchv/backend/vector_scalar.py`（新增逐 lane 标量展开）、`scratchv/backend/instruction_select.py`（接线 + P0 常量操作数修复）、`scratchv/backend/riscv_encoder.py`（向量助记符显式拒绝）、`scratchv/compiler.py` / `scratchv/main.py`（config/CLI 接线）  
> 只读依赖（一期不改）：`scratchv/simulator/rv32_emulator.py`、`scratchv/standalone/benchmark.py`  
> 功能范围：一期 = IR 层向量指令 + FOR loop strip-mining 向量化 + 后端逐 lane 标量展开（可用现有 RV32IM 仿真器执行验证语义）；二期 = RVV 文本发射规划（本期只定义映射表、接口与拒绝策略，不实现发射）

---

## 一、功能介绍

### 1.1 功能概述

#### 现状盘点（2026-09-14，HEAD `d146515`）

| 层 | 现状 | 结论 |
|----|------|------|
| IR `OpCode` | 仅标量：`ADD/SUB/MUL/DIV/RELU/...`（`scratchv/ir/types.py:14`） | 无任何向量操作定义 |
| IR 循环 | `FOR/ENDFOR` 仅支持常量 `start/end/step`，后端 `_select_endfor` 恒按 `+1` 递增（`instruction_select.py:210`） | 循环可变换面窄，但可用 strip 索引绕过 |
| 后端 | `InstructionSelector` 逐 IR op 选择 RV32IM 机器指令 | 遇到未知 opcode 直接 `ValueError`（安全网） |
| 编码器 | `riscv_encoder.py` 仅 RV32IM；未知助记符 `raise ValueError`（`:492`） | 无 V/P 扩展，且未知指令一律拒绝 |
| 仿真器 | `RV32Emulator` 仅 RV32IM；未识别 opcode 静默跳过 | 不新增向量状态，保持零改动 |
| Benchmark | `standalone/benchmark.py` 只按 RV32IM 分类计数 | 一期标量降级产物可直接测量 |

#### 目标重定义（诚实前提）

课题概述中“per-MAC 指令数从 ~12 降至 ~3-5”**不是本课题的现实前提**，原因：

1. ScratchV 当前可执行链路是 **RV32IM 标量 + Q16.16 定点**，lane 宽度为 i32；
2. **P-extension 无法打包 i32 lane**：其 SIMD 语义作用于 16/8-bit 打包子字（`add16/mul16/...`）。要吃到 P 的红利，必须先量化到 i16，属于模型/前端改造，不在本课题范围；
3. **V-extension 在 RV32 上的工具链在本仓库完全缺失**：无 RVV 编码器、无 RVV 仿真器、无 RVV 汇编器（零依赖约束下亦不引入外部工具链）；Spike/TinyFive 现有适配均按 RV32IM 口径；
4. 因此本期把课题切为两期，**一期只交付语义基础设施与可执行验证，不承诺性能**。

#### 分期定义

| 期 | 交付 | 可执行性 | 性能声明 |
|----|------|----------|----------|
| **一期** | IR 向量指令最小集；FOR 循环 strip-mining 向量化 pass；向量 op 的**后端逐 lane 标量展开**；编码器向量助记符拒绝；完整测试与对拍 | 生成纯 RV32IM 机器码，可在现有 `RV32Emulator` / `benchmark.py` 执行 | 只给理论指令数分析（标注**未测量**） |
| **二期（规划）** | RVV 文本发射 `RVVTextEmitter`；`--vector-isa v` 通路；二进制编码与仿真接入（依赖外部工具链） | 只允许文本发射；二进制路径显式拒绝 | 依赖外部仿真器实测，本期不产生数字 |

一期向量化经过后端展开后，语义与标量基线**逐指令同构**：每条标量指令与基线循环体执行的指令一一对应（仅迭代分组变化），因此对合法输入两者结果位精确一致。这是本课题可验证性的基石。

### 1.2 设计目标

- **语义等价**：向量化 + 标量展开后的程序与标量基线在合法输入上结果位精确一致（整数 i32 运算，无浮点重排问题）。
- **零运行时改动**：一期不修改仿真器、不修改编码器语义；编码器只增加“显式拒绝”护栏，避免未知指令被静默误编码。
- **可观测**：`--dump-ir` 能看到向量 IR；向量化决策有结构化报告（每条循环的接受/拒绝原因）。
- **可回退**：`--vectorize` 默认关闭；关闭时管线与现状逐字节一致。
- **不虚报**：性能结论只能是理论指令数分析，并标注未测量；不把 loop unroll 带来的开销摊销包装成 SIMD 收益。
- **与课题 28 解耦**：`OpCode` 采用“枚举末尾追加 + 注释分区”约定，避免与课题 28 的扩展指令选择在同一区域产生合并冲突。
- **为二期留口**：向量 op 的选择入口按 `opcode.is_vector()` 分派，二期可用一个 RVV 发射器替换标量展开器，接口不变。

---

## 二、设计规范

### 2.1 一期向量 IR 最小指令集

#### 2.1.1 记法约定

- `W` = strip 宽度（lane 数），一期取值 `{2, 4}`，默认 `4`。
- 向量值：`Value.shape == (W,)`，`dtype` 为 lane 元素类型（Q16.16 场景为 `DataType.INT32`；`FLOAT32` 视为同一 i32 载荷的别名）。
- `elem_bytes = 4`：每个 lane 占 4 字节（与现有 `LW/SW` 一致）。
- 向量指令 `attrs` 必须包含 `width`；内存类另含 `elem_bytes`、`align`。
- 地址操作数 `addr` 是标量 `Value`，表示 **lane 0 的字节地址**；lane k 的地址为 `addr + k*elem_bytes`。

#### 2.1.2 指令定义（BNF）

```
vector_op ::= vload_op | vstore_op | vbcast_op | vbin_op | vrelu_op

vload_op  ::= "vload"  vreg "," sval
vstore_op ::= "vstore" sval "," vreg
vbcast_op ::= "vbcast" vreg "," sval
vbin_op   ::= ("vadd" | "vsub" | "vmul" | "vdiv") vreg "," vreg "," vreg
vrelu_op  ::= "vrelu"  vreg "," vreg
```

| IR opcode（enum 名 / 字符串） | 形式 | 必需 attrs | 语义（∀k ∈ [0,W)） |
|---|---|---|---|
| `VLOAD` / `vload` | `%vd = vload %addr` | `width`, `elem_bytes=4`, `align=4` | `vd[k] = Mem32[addr + 4k]`（有符号 i32 读） |
| `VSTORE` / `vstore` | `vstore %addr, %vs` | `width`, `elem_bytes=4`, `align=4` | `Mem32[addr + 4k] = vs[k]` |
| `VBCAST` / `vbcast` | `%vd = vbcast %s` | `width` | `vd[k] = s`（标量广播） |
| `VADD` / `vadd` | `%vd = vadd %va, %vb` | `width` | `vd[k] = (va[k] + vb[k]) mod 2^32` |
| `VSUB` / `vsub` | `%vd = vsub %va, %vb` | `width` | `vd[k] = (va[k] - vb[k]) mod 2^32` |
| `VMUL` / `vmul` | `%vd = vmul %va, %vb` | `width` | `vd[k] = (va[k] * vb[k]) mod 2^32`（与标量 `MUL` 同语义） |
| `VDIV` / `vdiv` | `%vd = vdiv %va, %vb` | `width` | `vb[k] != 0 ? va[k] / vb[k] : 0xFFFFFFFF`（与 `RV32Emulator` 的 `DIV` 除零行为一致，`rv32_emulator.py:267`） |
| `VRELU` / `vrelu` | `%vd = vrelu %va` | `width` | `vd[k] = max(va[k], 0)` |

补充规则：

- `OpCode.is_vector()` 返回 `{VLOAD,VSTORE,VBCAST,VADD,VSUB,VMUL,VDIV,VRELU}` 的成员判定。
- 向量 op 的 `dest`（若有）必须 `shape == (width,)`；`VSTORE` 无 `dest`。
- 二元向量 op 的两个操作数必须是同 `width`、同 `dtype` 的向量值（`VBCAST` 的结果视为向量值）。
- 一期**不定义**跨 lane 归约（`VSUM/VDOT/VMAX`）与跨 lane 置换（`VSHUFFLE`）；见 2.6。

### 2.2 向量化 pass 规则

#### 2.2.1 可向量化循环模式

一期只处理**单块直落（straight-line）FOR 区域**，判定按以下顺序执行，任一失败即整环拒绝（不做部分向量化）：

```
FOR 区域 ::= FOR(iv) instr* ENDFOR

C1 结构：区域内不得出现 FOR/ENDFOR/BR/BR_IF/LABEL/RETURN
C2 界   ：attrs.start == 0（常量）；attrs.end == n（常量 int）；attrs.step == 1
C3 规模 ：n >= W
C4 元素模式：区域内至少一条 LW/STORE，其地址形如
            addr = ADD(base, MUL(iv, c))   且 c.const_value == elem_bytes (4)
            其中 base 在区域外定义（循环不变量）
C5 元素树：区域内每个标量指令必须属于下列之一
            (a) 地址链：MUL(iv, 4) / ADD(base, off) / LOAD / STORE
            (b) 元素表达式：操作数为 {常量, 区域外定义值, 已分类的元素结果}
                且 opcode ∈ {ADD, SUB, MUL, DIV, RELU, NEG}
            (c) LOAD_CONST
C6 无跨迭代标量：区域内的定义只被区域内指令使用（SSA 名唯一，天然满足）；
            iv 只允许出现在地址链的 MUL 中（其他使用 → 拒绝）；
            区域内的定义（含原 iv、区域内 LOAD/常量结果）不得在 FOR 区域外
            被使用（live-out → 拒绝，region-value-escapes），因为重写会替换或
            删除这些定义，否则会产生悬空 SSA 引用
C7 无别名（浅别名分析，保守）：
            (a) 同 base：仅允许“唯一 LOAD + 唯一 STORE 且 lane 偏移一致”的
                原地模式（`a[i]=f(a[i])`）；同一 base 多次 STORE → 拒绝
            (b) 异 base：仅当两个 base 均为常量绝对地址、且
                [base, base + n*elem_bytes) 区间可证不重叠时才允许；
                其余（异名指针、`src = sub(out, 4)` 等可重叠形态）→ 拒绝
```

合法的可向量化循环模式（一期支持的三类）：

| 模式 | 标量体形态（伪 IR） | 向量体 |
|------|---------------------|--------|
| P1 一元 map | `v=LOAD(a+4i); r=RELU(v); STORE(o+4i,r)` | `va=VLOAD; vr=VRELU(va); VSTORE` |
| P2 二元 map | `va=LOAD(a+4i); vb=LOAD(b+4i); r=MUL(va,vb); STORE(o+4i,r)` | `VLOAD,VLOAD,VMUL,VSTORE` |
| P3 广播 map | `va=LOAD(a+4i); r=DIV(va, K); STORE(...)` | `VLOAD,VBCAST,VDIV,VSTORE` |

链式表达式（P4）是 P1–P3 的递归组合：`r = RELU(MUL(LOAD(a+4i), LOAD(b+4i)))`。

#### 2.2.2 strip-mining 变换规则

设 `n = end - start`（`start == 0`），`strips = n // W`，`rem = n % W`：

```
原区域：  FOR(iv: [0, n)) B(iv) ENDFOR

变换后：
    FOR(vs: [0, strips); attrs += {vector_width=W, elem_bytes=4, orig_trip=n})
        B_vec(vs)                      # 每条 LOAD 地址 (vs*W)*4 起、宽 W
    ENDFOR
    [rem > 0]  FOR(ri: [strips*W, n))  B'(ri)  ENDFOR   # B' = 原标量体克隆
```

- `B_vec(vs)`：把 `B` 中所有 `base + iv*4` 地址改写为 `base + vs*(W*4)`；对应 op 替换为向量 op（`LOAD→VLOAD`、`STORE→VSTORE`、`RELU→VRELU`、`ADD/SUB/MUL/DIV→V*`）。
- `B'(ri)`：原标量体的**深克隆**，defs 与引用同步重命名（后缀 `__rem`），避免与向量体共用 SSA 名导致寄存器分配把两处定义当成同一 vreg。
- 余数循环复用原 `iv` Value（`start`/`end` 为常量），`_select_for` 支持任意常量 `start`，`ENDFOR` 的 `+1` 递增语义正确。
- `rem == 0` 时不生成余数循环；`n < W` 直接拒绝（不做掩码加载，避免越界读）。
- 不允许 `W` 跨迭代变化（strip 索引 vs 的步长为 1，与后端 `ENDFOR` 的 `+1` 递增一致）。

#### 2.2.3 宽度、对齐、规模

- **宽度**：`W ∈ {2,4}`，来自 `--vector-width`（默认 4）。W 只影响本 pass 的向量语义与后端展开；IR 中每指令显式携带 `width`，允许未来混合宽度。
- **对齐**：lane 访问全部是 4 字节 `LW/SW`，**只需 4 字节对齐**。ScratchV 的 Q16.16 元素步长恒为 4，`MemoryPlan` 对 workspace/权重按 64 字节对齐（`standalone/onnx_to_riscv_standalone.py` 的内存规划），因此 `base + 4m` 天然 4 字节对齐；**一期不做 peel/mask**，pass 在 attrs 中记录 `align=4` 供审计。
- **规模**：`min_strips = 1`（即 `n >= W`）。不引入“最小收益阈值”，避免第一期引入无法验证的启发式。

#### 2.2.4 trip count 与余数

- 界必须是常量（现有 IR `FOR` 只支持常量 `start/end/step`）。
- 非整除余数走 `B'` 标量尾声；语义与基线逐个元素一致。
- 动态 trip count（运行时变量）一期不支持；拒绝原因 `non-constant-bounds`。
- `step != 1` 拒绝：后端 `ENDFOR` 恒 `+1`，变换会静默改变迭代空间。
- `start != 0` 拒绝：避免在地址中引入第二个常量基址项，简化一期语义。

#### 2.2.5 拒绝原因（稳定字符串，供测试与报告断言）

| 常量名（`vectorize.py`） | 字符串 | 触发 |
|---|---|---|
| `REASON_NON_CONSTANT_BOUNDS` | `non-constant-bounds` | 界/步长非 int 常量 |
| `REASON_UNSUPPORTED_STEP` | `unsupported-loop-step` | `step != 1` |
| `REASON_UNSUPPORTED_START` | `unsupported-loop-start` | `start != 0` |
| `REASON_TRIP_TOO_SMALL` | `trip-count-too-small` | `n < W` |
| `REASON_NESTED_CONTROL_FLOW` | `nested-control-flow` | C1 违反 |
| `REASON_NO_ELEMENT_PATTERN` | `no-memory-element-pattern` | C4 违反 |
| `REASON_NON_ELEMENTWISE_IV` | `non-elementwise-iv-use` | iv 用于非地址链 |
| `REASON_ALIASING_STORE` | `aliasing-store` | C7 违反 |
| `REASON_UNSUPPORTED_OP` | `unsupported-op` | C5 违反 |
| `REASON_REGION_VALUE_ESCAPES` | `region-value-escapes` | 区域内定义在区域外被使用（live-out） |

拒绝不是错误：管线继续编译标量程序，原因写入 `PassResult.warnings` 与 `Vectorizer.last_report`。

### 2.3 标量降级规则

#### 2.3.1 降级位置与形态

降级发生在**指令选择阶段**（`scratchv/backend/vector_scalar.py` 的 `VectorScalarExpander`，由 `InstructionSelector` 在 `opcode.is_vector()` 时分派）。选择这一层而非再做一次 IR→IR 降级 pass 的理由：

1. 机器指令层可以安全使用 `ADDI rd, rs, imm` 生成 lane 地址（编码器对 `addi` 的 12-bit 立即数支持完整），而 IR 层的常量操作数会被 `InstructionSelector._op` 转成 R 型立即数，触发编码器的静默错误（见 4.2 P0）；
2. 不需要新增 IR pass，`--dump-ir` 保留向量 IR 供观测与测试；
3. 二期只需替换分派目标为 RVV 发射器，`Vectorizer` 与 IR 完全不动。

语义：strip 循环仍是**标量循环**（iv 步长 1，一次迭代处理 W 个元素），每条向量 op 展开为 W 条逐 lane 标量机器指令。

#### 2.3.2 逐 op 展开表（`addr_k = addr + 4k`，k=0 时直接复用 `addr`）

| 向量 op | 机器展开（每 strip 迭代） |
|---|---|
| `VLOAD %vd, %addr` | `ADDI a1, addr, 4` … `ADDI a_{W-1}, addr, 4(W-1)`；`LW vd_k, a_k` ×W |
| `VSTORE %addr, %vs` | 同上地址链；`SW a_k, vs_k` ×W |
| `VBCAST %vd, %s` | 无指令；lane k 全部绑定到 `s`（若 `s` 是常量 → `LI c, imm` 一次） |
| `VADD/SUB/MUL/DIV %vd,%va,%vb` | `ADD/SUB/MUL/DIV vd_k, va_k, vb_k` ×W |
| `VRELU %vd, %va` | `MAX vd_k, va_k, 0` ×W（编码器把 `max` 伪指令展开为分支序列，0 映射 `x0`，行为正确） |

lane 绑定：`VectorScalarExpander` 维护 `lanes: dict[str, list[MachineOperand]]`（逻辑向量值名 → W 个机器操作数），跨指令在本函数内有效；`begin_function()` 时清空。

#### 2.3.3 地址与常量处理约束

- **lane 地址**只用 `ADDI`（4/8/12… 均在 12-bit 有符号范围内），不生成 `ADD rd, rs, imm`。
- **标量常量操作数**（如 `VBCAST` 的常量源、向量 op 引用的常量）必须先 `LI tmp, imm` 物化为 vreg，再参与 R 型运算；这是对现有 P0 缺陷（2.3.4）的一致处理。
- 展开产物全部是机器指令，不含任何 `v*` 助记符；`AsmEmitter` / `LinearScanAllocator` 均无需改动。
- 展开顺序与 lane 顺序固定 `k = 0..W-1`，保证确定性输出（测试可断言 asm 文本）。

#### 2.3.4 后置校验与 P0 前置修复

- **后置校验**：`InstructionSelector.run()` 结束时断言机器指令流中不存在向量指令/向量 vreg 泄漏；`VectorScalarExpander` 遇到无法解析的向量值（无 lane 绑定）抛 `VectorLoweringError`。
- **P0 前置修复（与向量无关的既有标量缺陷，必须先修）**：`InstructionSelector._op()`（`instruction_select.py:63`）把常量操作数直接交给机器层的 R 型指令（`add/sub/mul/div`），`AsmEmitter` 输出 `mul t2, t0, 4`，而编码器 `_reg_num("4")` 对非寄存器文本**静默返回 0**（`riscv_encoder.py:103`），实际编码成 `mul t2, t0, x0`。任何含常量操作数的标量程序（包括本课题的 strip 地址 `MUL(vs, 16)` 与基线循环 `MUL(iv, 4)`）都会算错但“编译成功”。修复位置在指令选择层：R 型处理器遇到常量操作数时先 `LI` 物化到临时 vreg。已在当前 HEAD 复现：
  ```
  li t3, 4
  mul t4, t1, t3      # P0 修复后（正确）
  mul t2, t0, 4       # P0 修复前（编码为 mul t2, t0, x0，错误）
  ```
  该修复不改变编码器、不改变仿真器，不触碰向量语义。

### 2.4 二期 RVV 映射表（规划，仅文本发射）

约束：二期只做**文本发射与拒绝策略**——`--vector-isa v` 时允许输出含 RVV 助记符的 `.s` 文本；任何二进制路径（`assemble_to_binary`）必须抛 `VectorEncodingError`。RVV 实测依赖外部工具链与外部仿真器（见 4.3）。

| 一期向量 op | RVV 1.0 映射（SEW=32, LMUL=1, VLEN=128 假设） | 备注 |
|---|---|---|
| （循环前置） | `vsetvli t0, a0, e32, m1, ta, ma` | 每 strip 一次，可 hoist |
| `VLOAD` | `vle32.v vd, (rs1)` | 元素对齐即可；非对齐支持依赖实现 |
| `VSTORE` | `vse32.v vs, (rs1)` | 同上 |
| `VBCAST` | `vmv.v.x vd, rs1` | i32 lane |
| `VADD` | `vadd.vv vd, va, vb` | |
| `VSUB` | `vsub.vv vd, va, vb` | |
| `VMUL` | `vmul.vv vd, va, vb` | V 1.0 整数乘 |
| `VDIV` | 无整数向量除法 → 标量回退或软件序列 | 二期保留标量降级即可 |
| `VRELU` | `vmax.vx vd, va, x0` | |
| （规划）归约 | `vredsum.vs`, `vwmacc.vv` | 仅二期+ 讨论，用于 DOT/MAC |
| P-extension | 不映射 | i32 lane 无法被 16/8-bit 打包，**不做目标** |

### 2.5 合法/非法示例

**合法示例 L1（P2 二元 map，`n=16, W=4`）**：见附录 5.1，`strips=4, rem=0`，生成 1 个 strip 循环 + 4 条向量 op；展开后每 strip 执行 4×(2 LW + MUL + 4×MAX + SW) + 地址/循环开销。

**合法示例 L2（余数）**：`n=17, W=4` → `strips=4, rem=1`，生成 strip 循环 + 1 次迭代的标量尾声（`FOR ri=[16,17)`）。

**合法示例 L3（原地）**：`a[i] = relu(a[i])`，STORE.base == LOAD.base 且 lane 偏移一致 → 允许。

**非法示例 I1（归约/跨迭代标量）**：
```
for i = 0, 10
    s = add(s, i)      # s 在区域外定义，iv 参与非地址运算
endfor
```
→ `no-memory-element-pattern`（无 iv 地址链）或 `non-elementwise-iv-use`。

**非法示例 I2（跨迭代内存依赖）**：
```
for i = 0, 16
    v = load(a + 4i)
    store(a + 4(i-1), v)     # 读-写同一 base 且偏移不同
endfor
```
→ `aliasing-store`。

**非法示例 I3（动态界）**：`FOR` 无 `start/end` 常量 attrs → `non-constant-bounds`。

**非法示例 I4（嵌套控制流）**：区域内含 `BR_IF` → `nested-control-flow`。

### 2.6 约束与范围边界

- 一期**不改** `scratchv/simulator/rv32_emulator.py`：不新增向量寄存器堆、vtype 状态与向量 opcode 解码。
- 一期**不改** `scratchv/backend/riscv_encoder.py` 的编码语义：只增加 `VectorEncodingError` 显式拒绝（fail-loud 护栏）。
- 一期**不做**归约向量化、跨 lane 运算、动态 trip count、非 4 字节元素、非 0 起点、非 1 步长。
- 一期**不接入** standalone ONNX 直发机器码管线（`onnx_to_riscv_standalone.py` 不经过 IR）；向量化只在 IR 管线（`scratchv` CLI / `CompilerDriver`）生效。
- 性能数字只能是理论分析且标注**未测量**（见 5.2）。

---

## 三、测试设计

一期测试全部为 Python/pytest，落在 `tests/`，可无外部工具链执行。

### 3.1 测试用例 1：向量化后 IR 结构正确

**文件**：`tests/test_vectorize.py::TestVectorizerStructure`

**输入**：用 `IRBuilder` 手写程序（附录 5.1 的标量体：`out[i] = relu(a[i]*b[i])`），
`n=16, W=4`；`Vectorizer(program, width=4).run(program)`。

**预期输出**：
- 恰好 2 个 `FOR` 指令组（strip + 无余数不生成尾声）；
- strip `FOR` attrs：`{start:0, end:4, step:1, vector_width:4, elem_bytes:4, orig_trip:16}`；
- strip 体内恰好 `2×VLOAD + VMUL + VRELU + VSTORE` 各 1 条，`width=4`，向量 dest `shape==(4,)`；
- 向量 op 只出现在 strip 体内，尾声不存在；
- `last_report[0].status == "vectorized"`。

另设两个断言变体：`n=17` 时存在第二个 `FOR`（`start=16,end=17`）且其体内全为标量 op，且克隆 defs 名与向量体不重名；`n=3, W=4` 时拒绝且 IR 的 `dump()` 与输入逐字节相同。

**验证点**：`OpCode.is_vector()` 覆盖、attrs 精确、SSA 名不冲突、拒绝时 IR 不变。

### 3.2 测试用例 2：标量降级语义可执行对拍（现有仿真器）

**文件**：`tests/test_vector_lowering.py::TestEmulatorDifferential`

**输入**：
- 构造标量程序 S：`for i in [0,16): out[i] = max(a[i]*b[i], 0)`，`a/b/out` 基址分别为 `0x400000/0x410000/0x420000`（`load_const` 得到），元素为 i32；
- 程序 V：`Vectorizer(S, width=4)`；程序 B：S 的副本不做向量化；
- 对 B 与 V 分别执行同一条可执行链路：`InstructionSelector`（含 P0 修复）→ `RegisterAllocator(mode="greedy")` → `AsmEmitter` → `assemble_to_binary` → `RV32Emulator`；
- 用 `emu.write_i32` 在基址处填入固定向量（含负数、0、大值，覆盖 `VRELU` 与乘法回绕），运行后读取 `out` 区。

**预期输出**：
- 两个二进制都正常运行到 `RET`（`run()` 返回执行条数）；
- `out_V == out_B`（逐 i32 位相等）；
- `out_V == [max(a[i]*b[i], 0) for i in range(16)]` 的 numpy/整数参考值；
- V 的汇编文本中不出现任何 `v` 前缀助记符（`regex ^\s*v[a-z]` 无命中）。

**验证点**：降级语义等价、可被现有 RV32IM 仿真器执行、编码器不接触向量指令。

> 注：该用例必须走 `reg_alloc="greedy"`（CLI 默认值）。`reg_alloc="linear"` 的 `LinearScanAllocator.emit` 当前把标签输出为 `.label name` 且 `j` 的目标仅存在于注释里（`j  # .Lloop_header_1`），编码阶段标签丢失——属既有缺陷，不在本期范围，见 4.4 已知限制。

### 3.3 测试用例 3：不可向量化场景拒绝

**文件**：`tests/test_vectorize.py::TestVectorizerRejects`

**输入**：四类非法循环（见 2.5 I1–I4），每个循环独立成程序。

**预期输出**：
- `Vectorizer.run(program).changes == 0`（对 I1–I4 无任何变换）；
- `last_report` 中对应记录 `status == "rejected"` 且 `reason` 精确等于预期字符串（`non-elementwise-iv-use` / `no-memory-element-pattern` / `aliasing-store` / `nested-control-flow`）；
- `PassResult.warnings` 中包含含原因字符串的可读消息；
- `program.dump()` 与输入相同（拒绝即不触碰 IR）。

**验证点**：拒绝原因可机读、拒绝无副作用、编译不失败。

### 3.4 测试用例 4：编码器拒绝 RVV 助记符

**文件**：`tests/test_vector_encoder.py::TestVectorMnemonicRejection`

**输入**：`assemble_to_binary("vadd.vv v1, v2, v3")`、`assemble_to_binary("vsetvli t0, a0, e32, m1, ta, ma")`、`assemble_to_binary("vle32.v v1, (a0)")`。

**预期输出**：每次调用抛 `VectorEncodingError`，消息含助记符原文与 “phase 1 / RV32IM only” 说明；标量样例 `"add a0, a1, a2"` 仍正常编码。

**验证点**：fail-loud 护栏存在，且不影响标量编码回归。

### 3.5 覆盖矩阵

| 维度 | 覆盖用例 |
|---|---|
| IR 指令集 | 3.1（全部 8 个向量 op 至少各出现一次；`VBCAST/VSUB/VDIV` 在 P3 变体） |
| 宽度 | 3.1/3.2 使用 W=4；另加 W=2 参数化 |
| trip/余数 | 3.1（整除 16、非整除 17、过小 3） |
| 拒绝 | 3.3 |
| 可执行语义 | 3.2（基线 vs 向量化 vs 参考值三方对拍） |
| 编码器护栏 | 3.4 |
| 集成/CLI | 4.4 列出的 `TestCLIWiring` / `TestDriverIntegration` |

---

## 四、修改模块与实现步骤

### 4.1 涉及文件

| 类别 | 文件 | 改动 |
|---|---|---|
| 修改 | `scratchv/ir/types.py` | `OpCode` 末尾注释分区追加 8 个向量 op；新增 `is_vector()` |
| 修改 | `scratchv/ir/builder.py` | 追加 `vload/vstore/vbcast/vadd/vsub/vmul/vdiv/vrelu` |
| 新增 | `scratchv/optimizer/vectorize.py` | `Vectorizer` pass、`LoopVectorizationRecord`、拒绝原因常量 |
| 新增 | `scratchv/backend/vector_scalar.py` | `VectorScalarExpander`、`VectorLoweringError` |
| 修改 | `scratchv/backend/instruction_select.py` | P0 常量物化修复；`opcode.is_vector()` 分派；函数级 `begin_function()` |
| 修改 | `scratchv/backend/riscv_encoder.py` | `VectorEncodingError` + 向量助记符显式拒绝 |
| 修改 | `scratchv/compiler.py` | `CompilerConfig` 三字段；`_run_vectorizer`；ISA 拒绝；warning 透传 |
| 修改 | `scratchv/main.py` | `--vectorize`、`--vector-width`、`--vector-isa` 与 `args_to_config` 接线 |
| 新增 | `tests/test_vectorize.py`、`tests/test_vector_lowering.py`、`tests/test_vector_encoder.py` | 见第三部分 |
| 修改 | `tests/test_backend.py` | 追加 CLI 接线与 driver 集成用例 |
| 不修改 | `scratchv/simulator/rv32_emulator.py`、`scratchv/standalone/benchmark.py` | 一期只读依赖 |
| 不修改 | `scratchv/standalone/onnx_to_riscv_standalone.py` | 直发机器码管线，不经过 IR |

### 4.2 一期实现步骤

**P0（前置修复，标量正确性）**：修 `InstructionSelector` 常量操作数物化。
- 位置：`instruction_select.py:63` 附近新增 `_op_reg()`（或扩展 `_op`），在 `_select_add/_sub/_mul/_div` 与 `_select_gelu` 的 `DIV` 处使用；对常量操作数先 `LI tmp, imm` 再参与 R 型运算。
- 交付证据：`mul rd, rs, <imm>` 不再出现在任何输出；新增回归测试断言。
- 不做：不改 `riscv_encoder.py` 的编码逻辑（只在 P4 加拒绝护栏）。

**P1（IR 层）**：`types.py` 末尾追加向量分区与 `is_vector()`；`builder.py` 追加 8 个 API（语义见开发文档《接口契约》）。

**P2（Vectorizer）**：新建 `scratchv/optimizer/vectorize.py`，实现 2.2 的判定树、strip-mining 重写、余数克隆与报告；`PassResult.changes` = 被向量化循环数，`message` 形如 `vectorized 1/3 loop(s), width=4`。

**P3（标量展开）**：新建 `scratchv/backend/vector_scalar.py`；`InstructionSelector._select_function` 调 `begin_function()`；`_select_instruction` 先判 `is_vector()` 再走标量分派。

**P4（编码器护栏）**：`riscv_encoder.py` 新增 `VectorEncodingError` 与向量助记符前缀拒绝；消息固定为
`vector instruction '{op}' is not supported: ScratchV phase 1 targets RV32IM only`。

**P5（驱动与 CLI）**：`compiler.py` 增 `vectorize/vector_width/vector_isa` 三字段、`_run_vectorizer()`、`--vector-isa != "scalar"` 的显式失败；`main.py` 增三参数并接入 `args_to_config`。**必须同时改 parser 与 config 两侧**（课题 28 的 `--extended-isel` 只加了 parser、未接入 `args_to_config`，是可复制的反面教材）。

**P6（测试）**：落实第三部分 4 个用例 + CLI 接线 + driver 集成。

**P7（文档/报告）**：`Vectorizer.last_report` 通过 `--dump-ir`（stderr 打印）或编译 warnings 呈现；不新增报告文件格式。

### 4.3 二期实现步骤（规划）

1. **RVV 文本发射器**：`scratchv/backend/rvv_emit.py::RVVTextEmitter`，复用 `Vectorizer` 产物，按 2.4 映射表输出 `vsetvli` + 向量指令文本；`--vector-isa v` 时替换 `VectorScalarExpander`。
2. **编码拒绝**：二进制路径保持 `VectorEncodingError`（RVV 32-bit 编码涉及 vtype/vreg 与可变长度，ScratchV 编码器不扩展）。
3. **外部工具链**（可选、显式 opt-in）：`riscv64-unknown-elf-as`（binutils ≥ 2.40）或 LLVM MC ≥ 14 汇编 RVV 文本；不进入默认依赖。
4. **仿真验证**：Spike `--isa=rv32gcv` 或 QEMU `rv32,v=true`；`benchmark.py` 需新增 VLEN 感知的类别与周期模型后，才能产生测量数字。
5. **实测对比**：只有在 3/4 完成后，才允许把“per-MAC 指令数”从理论分析升级为测量值。

### 4.4 集成与回归测试

- **回归**：`make test`（`python3 -m pytest tests/ -v`，当前约 348+ 用例）必须全绿；`--vectorize` 关闭时不得有任何行为差异。
- **CLI 接线**：`build_arg_parser().parse_args([...,"--vectorize","--vector-width","2","--vector-isa","scalar"])` → `args_to_config()` 三字段逐一断言；`--vector-isa v` 编译返回 `success=False` 且错误消息含 `phase 2`。
- **driver 集成**：`CompilerDriver(CompilerConfig(vectorize=True, reg_alloc="greedy"))` + monkeypatch `_parse` 返回手写 IR，验证 `compile()` 成功且产物 asm 中无 `v*` 助记符。
- **已知限制（不在本期修，文档化）**：
  - `reg_alloc="linear"` 的标签/跳转发射缺陷（3.2 注）；向量化集成路径建议使用 greedy（CLI 默认）。
  - `InstructionSelector._select_alloca` 使用 `vreg("sp")`，会被寄存器分配改名导致 alloca 结果错误；本期样例与测试避免 `alloca`（用 `load_const` 绝对地址作 base）。
  - `--extended-isel`、`--verify-ir` 已有“只加 CLI 未接线”的历史缺口；本期新增参数必须两侧同时改，并有测试守门。

---

## 五、附录

### 5.1 完整 IR 示例

**标量 IR（向量化前，`n=16`）**：

```
fun $main(
  .entry:
    for $iv [start=0 end=16 step=1]
    $c4 = load_const 4 [value=4]
    $off = mul $iv $c4
    $pa = add $a_ptr $off
    $pb = add $b_ptr $off
    $po = add $o_ptr $off
    $va = load $pa
    $vb = load $pb
    $pr = mul $va $vb
    $y = relu $pr
    store $po $y
    endfor
    return
```

**向量 IR（`Vectorizer(width=4)` 之后）**：

```
    for $vs [start=0 end=4 step=1] [vector_width=4 elem_bytes=4 orig_trip=16]
    $c16 = load_const 16 [value=16]
    $boff = mul $vs $c16
    $pa = add $a_ptr $boff
    $pb = add $b_ptr $boff
    $po = add $o_ptr $boff
    $va = vload $pa [width=4 elem_bytes=4 align=4]
    $vb = vload $pb [width=4 elem_bytes=4 align=4]
    $pr = vmul $va $vb [width=4]
    $y = vrelu $pr [width=4]
    vstore $po $y [width=4 elem_bytes=4 align=4]
    endfor
    return
```

**后端展开（伪汇编，每个 strip 迭代）**：

```
    li   t3, 16
    mul  boff, vs, t3        # P0 修复后：常量经 vreg 参与 R 型
    add  pa, a_ptr, boff
    add  pb, b_ptr, boff
    add  po, o_ptr, boff
    lw   vd0, pa
    addi a1, pa, 4
    lw   vd1, a1
    addi a2, pa, 8
    lw   vd2, a2
    addi a3, pa, 12
    lw   vd3, a3
    ...（b 同理；mul ×4；max ×4；sw ×4）
```

余数示例（`n=17`）追加：`FOR ri=[16,17)` 的标量体（克隆，defs 后缀 `__rem`）。

### 5.2 理论指令数分析（未测量）

> **免责声明**：以下全部为静态推演，**未在任何仿真器/硬件上测量**；实测需等二期的工具链接入。任何引用必须保留“未测量”标注。

标量基线（Q16.16 i32，`y[i]=relu(a[i]*b[i])`，当前后端逐元素体）动态指令数约：
`1 (li c4) + 1 (mul off) + 3 (add pa/pb/po) + 2 (lw) + 1 (mul) + ~4 (max 伪指令展开) + 1 (sw) + 2 (addi+j) ≈ 15/元素`。

一期向量化+标量展开（W=4）每 strip 约：
`2 (li c16/mul) + 3 (add) + 4×(2 lw + mul + 4 max + sw) + 2 (addi+j) + 2 (bge 展开) ≈ 41/4 元素 ≈ 10.3/元素`。

结论：一期的收益**仅来自循环开销与地址计算的摊销**（约 10–15% 量级的理论差异，未测量），与 SIMD 无关。真正的 per-MAC 收益必须等二期 RVV 实测；P-extension 对本项目的 i32 lane 不适用。

### 5.3 参考资料

- RISC-V V-extension 规范：<https://github.com/riscv/riscv-v-spec>
- 课题 29 目标文档：`docs/topics/29-SIMD向量化.md`
- 相关课题：课题 28（扩展指令选择 `inst_select_ext.py`）、课题 27（RV32 全量 Benchmark）、课题 19（Standalone RISC-V 编译器）
- 模板：`设计文档模板.md`
