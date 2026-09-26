# 接口冻结文档 v1.0

> 上游文档：[开发计划.md](../开发计划.md) §2.2、§4.3 W1
> 状态：**草案**，待 W1 D4 接口冻结会议确认后升为 v1.0
> 本文档是 W1 的交付物之一，对应 CI Job `docs:interfaces`

---

## 0. 摘要

本文档冻结 ScratchV 编译 Qwen3-0.6B 所需的**四类接口**：

| # | 接口 | 边界 | 现状 |
|---|---|---|---|
| 一 | 前端 | ONNX 文件 → IR `Program` | ✅ 已实现（仅覆盖 CNN 算子） |
| 二 | IR | IR 数据结构与算子契约 | ⚠️ 结构完备，算子缺 2 个、多个为占位 |
| 三 | 后端 | IR → RISC-V 汇编 | ⚠️ 框架完备，**FP32 无选择器**、MATMUL 为占位 |
| 四 | 运行时 | host ↔ RISC-V FFI | ❌ 不存在，本文档首次定义 |

**本次冻结的核心结论**：接口一、二是**扩展**（加算子），接口三、四是**新建**（后端补浮点选择器、运行时从零搭）。缺口清单见 §6。

---

## 1. 冻结规则

1. 冻结后修改接口需 **PR + E2 批准 + 通知全员**，并在本文档 §8 留痕。
2. 版本号：破坏性修改升 `v2.0`，兼容性扩展升 `v1.1`。
3. 任何人**不得**绕过接口直接改对方模块（例如前端不得直接调用后端函数）。
4. 接口的**验证责任**在消费方：E2 验证前端产物，E3 验证 IR 输入契约，E5 验证端到端数值。

---

## 2. 接口一：前端（ONNX → IR）

### 2.1 现状

**入口**（`scratchv/frontend/onnx_parser.py:14,24`）：

```python
class ONNXParser:
    def parse(self, model_path: str) -> Program
```

**异常**：`ONNXParseError`（同文件 `:9`）

### 2.2 v1 冻结定义

```python
# 不变
ONNXParser().parse(model_path: str) -> Program
```

**新增要求**：`parse()` 遇到不支持的算子时**必须抛 `ONNXParseError` 并列出算子名**，不得静默跳过。

> 依据：计划 §4.3 W2 的 `frontend:parse-qwen3` 要求"完整 Qwen3 解析无错误，28 层完整"。静默跳过会让缺算子在 W3 才暴露。

### 2.3 需新增的算子识别

| 算子 | 状态 | 归属周 |
|---|---|---|
| RMSNorm | ❌ 不存在 | W2 |
| RoPE | ❌ 不存在 | W2 |
| SwiGLU | ❌ 不存在 | W2 |
| GQA（分组查询注意力） | ❌ 不存在 | W2 |

**冻结约定**：上述四个是**复合模式**（由基础 ONNX 算子组合而成），识别后**必须 lowering 为 §3 的新增 IR 节点**，不得在前端直接展开成算术序列。

---

## 3. 接口二：IR 数据结构与算子契约

### 3.1 现状（`scratchv/ir/types.py`）

```python
class OpCode(enum.Enum):          # 28 个取值
    ADD SUB MUL DIV NEG EXP LOAD STORE LOAD_CONST ALLOCA
    FOR ENDFOR BR BR_IF LABEL RETURN
    MATMUL RELU MAXPOOL SOFTMAX GELU DOT CONV GEMM SIGMOID
    TRANSPOSE RESHAPE CONCAT

class DataType(enum.Enum):
    FLOAT32 INT32 FLOAT64 INT64

@dataclass
class Value:      name: str; dtype: DataType; is_constant: bool
                  const_value; shape: tuple[int, ...]

@dataclass
class Instruction: opcode: OpCode; dest: Value|None
                   operands: list[Value]; attrs: dict; target: str|None

class Function:   name; params; returns; blocks; locals
class Program:    functions: list[Function]; global_values: list[Value]
```

### 3.2 v1 冻结定义

**以上结构全部冻结，不改。** 新增能力**只通过新增 `OpCode` 取值**获得，这样既有 pass、printer、verifier 无需改动。

### 3.3 需新增的 IR 节点

| 新 OpCode | 语义 | attrs | 归属周 |
|---|---|---|---|
| `RMSNORM` | `x / sqrt(mean(x²)+eps) * weight` | `eps`, `axis` | W2 |
| `ROPE` | 旋转位置编码，partial | `head_dim`, `rope_theta`, `partial_ratio` | W2 |
| `SWIGLU` | `silu(gate) * up` | — | W2 |
| `ATTENTION` | Q/K/V + 因果掩码 → 输出 | `num_heads`, `num_kv_heads`, `head_dim`, `causal` | W2 |

**冻结约定**：
- `ATTENTION` **必须**是单个节点，不得在前端展开为 MatMul/Softmax 序列。理由：计划 §2.2 把"fused attention"列为可选优化——若前端已展开，后续无法融合。
- 形状信息**编译期完全确定**（计划 §1.3：固定 L=256），因此 `Value.shape` 必须全部落实，不得出现动态维度。

---

## 4. 接口三：后端（IR → RISC-V）

### 4.1 现状

| 组件 | 位置 | 状态 |
|---|---|---|
| 指令选择 | `backend/instruction_select.py` | ⚠️ 26/28 opcode 有 handler；**MATMUL 为占位** |
| 扩展选择 | `backend/inst_select_ext.py` | ✅ 有 FLOAT64 选择器（`_select_fadd_d` 等） |
| 机器指令 | `backend/machine_types.py` | ✅ `MachineOp` 含 FP32（`FADD_S FMUL_S FLW FSW`）与 FP64 |
| 汇编发射 | `backend/asm_emit.py` | ✅ 有 FP32 助记符映射 |
| 寄存器分配 | `backend/regalloc_linear.py`, `regalloc_cfg.py` | ✅ |
| 驱动 | `scratchv/compiler.py:439` | ✅ `InstructionSelector(program).run()` → `AsmEmitter(...).emit()` |

### 4.2 ⚠️ 已确认的三处能力缺口（静态勘察结论，待探测验证）

**缺口 1：MATMUL 是占位实现**（`instruction_select.py`）

```python
def _select_matmul(self, instr: Instruction) -> None:
    a_reg = self._op(instr, 0); b_reg = self._op(instr, 1); dst = self._dst(instr)
    if dst:
        self._emit(MachineOp.MUL, dst, a_reg, b_reg, comment="matmul: a * b")
```

只做了一次**标量整数乘**，没有 m/n/k 循环。而 `ir/builder.py:179` 已经传入了 `m=, n=, k=` 属性。**这是 Transformer 最核心的算子。**

**缺口 2：FP32 无选择器**——`asm_emit.py` 有条目、`machine_semantics.py` 有 def/use，但**全后端 0 处 emit** `MachineOp.FMUL_S / FADD_S / FLW / FSW`。FP64 反而有选择器（`inst_select_ext.py`）。

**缺口 3：2 个 opcode 无 handler**——`TRANSPOSE`、`CONCAT` 会直接 `raise ValueError`。Transformer 的 attention 需要 transpose，GQA 需要 concat。

### 4.3 待决议：目标 ISA 是 RV32 还是 RV64？（冻结会议必须拍板）

仓库里两种目标混用，证据：

| 位置 | 目标 |
|---|---|
| `backend/asm_emit.py:3` 文档串 | `riscv64-unknown-elf-gcc` |
| `backend/llvm_codegen.py:51` target triple | `riscv64-unknown-elf` |
| `standalone/onnx_to_riscv_standalone.py` | **RV32IM** |
| `tests/test_standalone_execution.py:29` | `qemu-riscv32` |
| `.github/workflows/ci.yml:94` | `qemu-riscv32` |
| **`开发计划.md` §1.3 / §4.1** | **QEMU riscv64** |
| `backend/machine_types.py` `MachineOp` | 只有 `LW/SW`（32 位），**无 `LD/SD`** |

> `MachineOp` 缺 64 位整数load/store，暗示 IR→asm 路径实际是 **RV32 + F/D**，尽管文档串写的是 riscv64。

**这个决定影响**：工具链选择、`qemu-riscv32/64`、指针宽度（4 vs 8 字节）、ABI（ilp32 vs lp64）、**以及 0.6B 模型全部权重与激活的地址空间规划**。

**建议**：与计划保持一致取 **RV64**，则需确认 `MachineOp` 补齐 `LD/SD` 的成本；若取 RV32，则计划 §1.3/§4.1 需修订。

### 4.4 v1 冻结定义

```python
InstructionSelector(program).run() -> list[MachineInstr]
AsmEmitter(allocated_machine_instrs).emit() -> str   # GNU GAS 语法
```

**新增要求**：选择器遇到无 handler 的 opcode，**必须**在编译期报错并指出 opcode 名（现状 `raise ValueError` 已满足，保留）。

---

## 5. 接口四：运行时 FFI（host ↔ RISC-V）**新增**

### 5.1 现状：不存在

`scratchv/runtime/` 目录不存在。计划 §2.2 明确"运行时 | 无 | Tokenizer、采样、掩码、token 循环 | **新增 `runtime/`**"。

### 5.2 现有先例（可沿用）

`standalone/onnx_to_riscv_standalone.py` 文档串定义了 bare-metal ABI：

```
On entry:
  a0 = pointer to input tensor (float32, NCHW layout)
  a1 = pointer to output buffer
The binary is position-independent (uses auipc for data addressing).
Returns via jalr zero, ra, 0.
```

**冻结建议：沿用这一约定**，降低 E3/E4 的对接成本。

### 5.3 v1 冻结定义（提议）

**入口签名**

```
a0 = pointer to input_ids      (int32 连续数组, 长度 256)
a1 = pointer to attention_mask (float32 连续数组, 长度 256*256)
a2 = pointer to logits 输出缓冲 (float32 连续数组, 长度 256*151936)
返回：jalr zero, ra, 0；a0 = 0 表示成功，非 0 为错误码
```

**内存布局**：一律**行主序、连续、无 padding**。

| 张量 | dtype | 形状 | 元素数 |
|---|---|---|---|
| `input_ids` | int32 | `[1,256]` | 256 |
| `attention_mask` | float32 | `[1,1,256,256]` | 65536 |
| `logits` | float32 | `[1,256,151936]` | 38,895,616 |

> ⚠️ `logits` 约 **148 MB**（FP32）。RV32 的 4 GB 地址空间够用，但需在 W1 确认 QEMU 内存配置（计划风险表已列"QEMU 内存不足"，缓解措施 `-m 8G`）。

**权重**：编译期嵌入或运行时 mmap，**由 W4 决定**（计划 §4.3 W4 `unit:weight-mmap`）。接口层面冻结为"权重对 host 不可见，仅通过入口指针访问"。

**数据类型约定**：v1 内部统一 FP32（计划 §1.3）。`input_ids` 用 int32 而非 int64，避免 RV32 下的 64 位整数运算。

### 5.4 运行时模块结构（提议）

```
scratchv/runtime/
  __init__.py
  tokenizer.py     # prompt → input_ids
  mask.py          # 因果掩码 + padding
  sampler.py       # greedy / top-k / top-p
  runner.py        # token 循环 + FFI 调用
```

---

## 6. 能力缺口与责任分配

| # | 缺口 | 影响周 | 责任 | 依据 |
|---|---|---|---|---|
| 1 | MATMUL 占位（无 m/n/k 循环） | W2–W4 | E3 | §4.2 缺口 1 |
| 2 | FP32 无指令选择器 | W2–W4 | E3 | §4.2 缺口 2 |
| 3 | `TRANSPOSE` 无 handler | W2 | E3 | §4.2 缺口 3 |
| 4 | `CONCAT` 无 handler | W2 | E3 | §4.2 缺口 3 |
| 5 | RMSNorm/RoPE/SwiGLU/GQA 无 IR 节点 | W2 | E2 | §3.3 |
| 6 | 四个算子无前端解析 | W2 | E1 | §2.3 |
| 7 | 运行时模块不存在 | W2 | E4 | §5.1 |
| 8 | 目标 ISA 未定（RV32/RV64） | **W1** | E2 拍板 | §4.3 |
| 9 | FFI 调用约定未实现 | W4 | E4 | §5.3 |

---

## 7. 待决议项（W1 D4 冻结会议逐条拍板）

- [ ] **D1**：目标 ISA 取 RV32 还是 RV64？（§4.3）
- [ ] **D2**：FFI 入口约定是否沿用 `a0/a1` + `jalr zero, ra, 0`？（§5.3）
- [ ] **D3**：`input_ids` 用 int32 是否可接受？（§5.3）
- [ ] **D4**：FP32 选择器补齐 vs 改用 FP64（后者已有选择器）？（§4.2 缺口 2）
- [ ] **D5**：`ATTENTION` 单节点 vs 前端展开？（§3.3）
- [ ] **D6**：权重嵌入 vs mmap 的接口边界？（§5.3）
- [ ] **D7**：接口版本号与变更审批流程确认？（§1）

---

## 8. 变更记录

| 日期 | 版本 | 变更 | 作者 |
|---|---|---|---|
| — | v0.1 | 初稿：四类接口现状勘察 + 缺口清单 | — |
| — | v1.0 | 待 W1 D4 会议确认 | — |
