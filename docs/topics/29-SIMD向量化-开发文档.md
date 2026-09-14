# 课题 29：SIMD 向量化开发文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 配套设计：同目录《设计文档.md》（术语、分期边界、strip-mining 规则、测试设计以设计文档为准）  
> 适用基线：`HEAD d146515`（行号会漂移，实施前以 `git rev-parse HEAD` + grep 复核）  
> 涉及文件：`scratchv/ir/types.py`、`scratchv/ir/builder.py`、`scratchv/optimizer/vectorize.py`（新增）、`scratchv/backend/vector_scalar.py`（新增）、`scratchv/backend/instruction_select.py`、`scratchv/backend/riscv_encoder.py`、`scratchv/compiler.py`、`scratchv/main.py`、`tests/test_vectorize.py`（新增）、`tests/test_vector_lowering.py`（新增）、`tests/test_vector_encoder.py`（新增）、`tests/test_backend.py`  
> 只读依赖（一期禁止修改）：`scratchv/simulator/rv32_emulator.py`、`scratchv/standalone/benchmark.py`、`scratchv/standalone/onnx_to_riscv_standalone.py`

---

## 一、接口契约（精确名称）

> 本节是唯一权威命名表。实现时任何名称与本节不一致，视为接口破坏，需改文档或改测试，不允许“各写各的”。

### 1.1 IR 枚举（`scratchv/ir/types.py`）

| 名称 | 值（`opcode.value`） | 形式 | 必需 attrs |
|---|---|---|---|
| `OpCode.VLOAD` | `"vload"` | `%vd = vload %addr` | `width:int`, `elem_bytes:int=4`, `align:int=4` |
| `OpCode.VSTORE` | `"vstore"` | `vstore %addr, %vs` | 同上 |
| `OpCode.VBCAST` | `"vbcast"` | `%vd = vbcast %s` | `width:int` |
| `OpCode.VADD` | `"vadd"` | `%vd = vadd %va, %vb` | `width:int` |
| `OpCode.VSUB` | `"vsub"` | `%vd = vsub %va, %vb` | `width:int` |
| `OpCode.VMUL` | `"vmul"` | `%vd = vmul %va, %vb` | `width:int` |
| `OpCode.VDIV` | `"vdiv"` | `%vd = vdiv %va, %vb` | `width:int` |
| `OpCode.VRELU` | `"vrelu"` | `%vd = vrelu %va` | `width:int` |

方法：`OpCode.is_vector(self) -> bool`，等价于 `self in _VECTOR_OPS`（模块级 `frozenset`）。

属性键（精确字符串）：`"width"`、`"elem_bytes"`、`"align"`；strip `FOR` 附加键：`"vector_width"`、`"orig_trip"`（`"elem_bytes"` 复用）。

约定：向量 `Value` 满足 `shape == (width,)` 且 `dtype` 为 lane 元素类型；`VSTORE.dest is None`。

### 1.2 Vectorizer 常量与报告（`scratchv/optimizer/vectorize.py`）

拒绝原因常量（值为稳定字符串，测试直接断言）：

```python
REASON_NON_CONSTANT_BOUNDS   = "non-constant-bounds"
REASON_UNSUPPORTED_STEP      = "unsupported-loop-step"
REASON_UNSUPPORTED_START     = "unsupported-loop-start"
REASON_TRIP_TOO_SMALL        = "trip-count-too-small"
REASON_NESTED_CONTROL_FLOW   = "nested-control-flow"
REASON_NO_ELEMENT_PATTERN    = "no-memory-element-pattern"
REASON_NON_ELEMENTWISE_IV    = "non-elementwise-iv-use"
REASON_ALIASING_STORE        = "aliasing-store"
REASON_UNSUPPORTED_OP        = "unsupported-op"
REASON_REGION_VALUE_ESCAPES  = "region-value-escapes"
```

报告记录：

```python
@dataclass
class LoopVectorizationRecord:
    function: str          # Function.name
    block: str             # FOR 所在 BasicBlock.name
    index: int             # FOR 在该 block.instructions 中的下标（变换前）
    status: str            # "vectorized" | "rejected"
    reason: str = ""       # 仅 rejected 时非空，取 REASON_* 之一
    start: int = 0         # 原始 attrs.start
    end: int = 0           # 原始 attrs.end
    width: int = 0         # 实际采用的 W
    strips: int = 0        # n // W
    remainder: int = 0     # n % W
    vector_ops: int = 0    # 生成的向量 op 条数（vectorized 时）
```

### 1.3 Builder API（`scratchv/ir/builder.py`）

```python
def vload(self, addr: Value, *, width: int = 4,
          elem_bytes: int = 4, align: int = 4) -> Value: ...
def vstore(self, addr: Value, vec: Value, *, width: int | None = None,
           elem_bytes: int = 4, align: int = 4) -> Instruction: ...
def vbcast(self, scalar: Value, *, width: int = 4) -> Value: ...
def vadd(self, lhs: Value, rhs: Value, *, width: int | None = None) -> Value: ...
def vsub(self, lhs: Value, rhs: Value, *, width: int | None = None) -> Value: ...
def vmul(self, lhs: Value, rhs: Value, *, width: int | None = None) -> Value: ...
def vdiv(self, lhs: Value, rhs: Value, *, width: int | None = None) -> Value: ...
def vrelu(self, val: Value, *, width: int | None = None) -> Value: ...
```

规则：`width=None` 时从第一个向量操作数的 `shape[0]` 推断，推断失败取 `4`；`dest.dtype` 继承操作数 `dtype`（默认 `DataType.FLOAT32`）；`dest.shape = (width,)`。

### 1.4 Pass 类与签名

```python
# scratchv/optimizer/vectorize.py
class Vectorizer(CompilerPass):
    def __init__(self, program: Program, *, width: int = 4,
                 elem_bytes: int = 4) -> None: ...
    @property
    def name(self) -> str: ...                     # "vectorize"
    def run(self, input_data: Any) -> PassResult: ...
    last_report: list[LoopVectorizationRecord]     # 运行后可读

# scratchv/backend/vector_scalar.py
class VectorLoweringError(ValueError): ...

class VectorScalarExpander:
    def __init__(self, *, default_width: int = 4) -> None: ...
    def begin_function(self, func_name: str) -> None: ...
    def expand(self, instr: Instruction) -> list[MachineInstr]: ...
```

`PassResult` 约定：`data=program`，`changes` = 被向量化循环数，`message` 形如 `"vectorized 1/3 loop(s), width=4"`，`warnings` 每条拒绝一行：`f"loop {func}:{block}[{index}] rejected: {reason}"`。

### 1.5 `CompilerConfig` 字段与 CLI（精确名称）

```python
# scratchv/compiler.py :: CompilerConfig（追加，默认值即“现状”）
vectorize: bool = False        # 关闭时零行为差异
vector_width: int = 4          # 允许 2 或 4
vector_isa: str = "scalar"     # "scalar" | "p" | "v"（p/v 一期显式拒绝）
```

```python
# scratchv/main.py :: build_arg_parser()
parser.add_argument("--vectorize", action="store_true",
                    help="Run FOR-loop strip-mining vectorization (Topic 29, phase 1)")
parser.add_argument("--vector-width", type=int, choices=[2, 4], default=4,
                    help="Vector strip width W (default: 4)")
parser.add_argument("--vector-isa", choices=["scalar", "p", "v"], default="scalar",
                    help="Vector ISA target; 'p'/'v' are rejected until phase 2")
```

```python
# scratchv/main.py :: args_to_config()
vectorize=args.vectorize,
vector_width=args.vector_width,
vector_isa=args.vector_isa,
```

### 1.6 异常与错误消息（精确文本）

| 异常 | 触发 | 消息模板 |
|---|---|---|
| `VectorLoweringError` | 向量值无 lane 绑定 / 宽度不一致 | `f"vector value '{name}' has no lane binding"` |
| `VectorEncodingError` | 编码器遇到 `v*` 助记符 | `f"vector instruction '{op}' is not supported: ScratchV phase 1 targets RV32IM only"` |
| 编译失败（`CompileResult.errors`） | `--vector-isa` 为 `p`/`v` | `f"vector-ISA '{isa}' is not implemented (phase 2); use --vector-isa scalar"` |

`VectorEncodingError` 继承 `ValueError`（兼容既有 `except ValueError` 调用方）；`VectorLoweringError` 同理。

---

## 二、`ir/types.py` 新增段落位置约定（与课题 28 并行）

### 2.1 位置规则

- **只允许在 `OpCode` 枚举末尾追加**，禁止在既有成员之间插入、禁止重排；
- 每个课题拥有独立的注释横幅分区，课题 28 分区在前、课题 29 分区紧随其后；
- 枚举成员一律显式赋值小写字符串，禁止自动 `auto()`；
- 测试只按成员名引用，不依赖枚举顺序。

### 2.2 目标代码形态（`types.py:50` 之后）

```python
    # Shape / data movement
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    CONCAT = "concat"

    # ── Extended instruction selection (Topic 28) ────────────────────
    # 课题 28 的分区；课题 29 不改动、不插入
    # （如 SQRT/ABS/MIN/MAX/IDIV/REM/MOD/LOAD_F64/... 由课题 28 自行追加）

    # ── SIMD vector ops (Topic 29, phase 1) ──────────────────────────
    VLOAD = "vload"
    VSTORE = "vstore"
    VBCAST = "vbcast"
    VADD = "vadd"
    VSUB = "vsub"
    VMUL = "vmul"
    VDIV = "vdiv"
    VRELU = "vrelu"
```

### 2.3 `is_vector()` 与模块级集合

在 `is_control_flow()`（`types.py:69`）之后追加：

```python
    def is_vector(self) -> bool:
        return self in _VECTOR_OPS
```

在 `class OpCode` 定义前（或枚举定义后）定义：

```python
_VECTOR_OPS = frozenset({
    OpCode.VLOAD, OpCode.VSTORE, OpCode.VBCAST,
    OpCode.VADD, OpCode.VSUB, OpCode.VMUL, OpCode.VDIV, OpCode.VRELU,
})
```

注意：`_VECTOR_OPS` 若定义在类体之前，成员引用需写成 `frozenset` 构造放在类之后（Python 类体执行顺序约束），推荐放在模块底部或用字符串集合 + 运行时转换，实施时二选一并加测试。

---

## 三、`ir/builder.py` 新增 API

追加到 `reshape()`（`builder.py:206-209`）之后，风格与既有方法一致：

```python
    def vload(self, addr: Value, *, width: int = 4,
              elem_bytes: int = 4, align: int = 4) -> Value:
        dest = self.make_value()
        dest.shape = (width,)
        self._emit(OpCode.VLOAD, dest, [addr], width=width,
                   elem_bytes=elem_bytes, align=align)
        return dest

    def vstore(self, addr: Value, vec: Value, *, width: int | None = None,
               elem_bytes: int = 4, align: int = 4) -> Instruction:
        w = width if width is not None else _shape_width(vec)
        return self._emit(OpCode.VSTORE, operands=[addr, vec],
                          width=w, elem_bytes=elem_bytes, align=align)
```

其余 `vbcast/vadd/vsub/vmul/vdiv/vrelu` 同构：生成 `dest`（`shape=(w,)`、`dtype` 继承）、`_emit` 带 `width=w`。内部小工具：

```python
def _shape_width(value: Value, default: int = 4) -> int:
    return value.shape[0] if value.shape else default
```

边界：`vstore` 不返回 `Value`（与既有 `store` 一致，返回 `Instruction`）；调用方若把 `vstore` 结果当值使用属误用，测试覆盖。

---

## 四、Vectorizer pass 设计与签名（`scratchv/optimizer/vectorize.py` 新增）

### 4.1 模块结构

```python
"""FOR-loop strip-mining vectorizer (Topic 29, phase 1)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from scratchv.ir.types import OpCode, Value, Instruction, BasicBlock, Function, Program
from scratchv.pass_interface import CompilerPass, PassResult

# REASON_* 常量（见 1.2）
# LoopVectorizationRecord（见 1.2）

class Vectorizer(CompilerPass):
    name -> "vectorize"
    __init__(self, program, *, width=4, elem_bytes=4)
    run(self, input_data) -> PassResult
    # 内部状态
    _counter: int              # 新鲜 Value 命名计数器
    _report: list[LoopVectorizationRecord]
    last_report 属性
```

### 4.2 主流程

```
run(program):
    self._report.clear()
    changes = 0
    for func in program.functions:
        for block in func.blocks:
            i = 0
            while i < len(block.instructions):
                if instr is FOR:
                    region = _collect_region(block, i)     # 找匹配 ENDFOR，限同一 block
                    rec = _try_vectorize(func, block, i, region)
                    self._report.append(rec)
                    if rec.status == "vectorized":
                        changes += 1
                        i = region.end_index + 1            # 跳过重写后的区域
                        continue
                i += 1
    return PassResult(data=program, changes=changes,
                      message=f"vectorized {changes}/{len(self._report)} loop(s), width={self.width}",
                      warnings=[...])
```

### 4.3 内部方法（建议签名）

```python
def _collect_region(self, block, for_index) -> _LoopRegion        # 扫描到配对 ENDFOR；不匹配则整块放弃
def _try_vectorize(self, func, block, for_index, region) -> LoopVectorizationRecord
def _check_bounds(self, for_instr) -> tuple[int, int, str | None] # (n, W?, reason)
def _classify(self, region) -> _Plan | str                        # 返回计划或拒绝原因
def _rewrite(self, block, region, plan) -> tuple[int, int]        # (vector_ops, strips)
def _clone_remainder(self, block, region, plan) -> int            # 返回生成的尾声循环数
def _fresh(self, prefix="v") -> str                               # "v_1", "v_2", ...
```

### 4.4 判定算法（与设计文档 2.2.1 的 C1–C7 一一对应）

1. **C2/C3**：读 `for_instr.attrs["start"|"end"|"step"]`，要求 `int` 常量、`start==0`、`step==1`、`n>=W`；否则返回对应 `REASON_*`。
2. **C1**：区域内出现 `FOR/ENDFOR/BR/BR_IF/LABEL/RETURN` → `REASON_NESTED_CONTROL_FLOW`。
3. **建立局部 def 表**：`defs: dict[str, Instruction]`（`instr.dest.name -> instr`），并记录区域外定义集合 `outer_defs`（函数内所有区域外 defs + `func.params` + 常量）。
4. **C4 地址链识别** `_match_address(use_instr)`：
   - `addr` 的 def 形如 `ADD(x, y)`，其中一侧是 `MUL(iv, c)` 或 `MUL(c, iv)` 且 `c.const_value == elem_bytes`；另一侧 `base ∈ outer_defs`；
   - 同一 `base` 在区域内必须始终配同一种偏移形式；
   - 找到 ≥1 条地址链，否则 `REASON_NO_ELEMENT_PATTERN`。
5. **C5 指令分类**：对区域内每条指令：
   - 严格属于地址链集合 → 分类为 `ADDR`；
   - `opcode ∈ {LOAD_CONST}` → `CONST`；
   - `opcode ∈ {ADD,SUB,MUL,DIV,RELU,NEG}` 且所有操作数 ∈ {常量, outer_defs, 已分类的元素结果} → `ELEM`（操作数中出现 `iv` → `REASON_NON_ELEMENTWISE_IV`）；
   - 其余 → `REASON_UNSUPPORTED_OP`；
   - 分类为 `ELEM`/`CONST` 但结果未被任何 STORE 使用 → `REASON_UNSUPPORTED_OP`（防死代码混入）。
6. **C6 live-out**：函数级扫描，区域内定义的任何值（含原 `iv`、区域内 LOAD 结果与常量）在 `region.body` 之外被使用 → `REASON_REGION_VALUE_ESCAPES`（保守拒绝；重写会替换或删除这些定义）。
7. **C7 别名**（浅别名分析，保守）：同 base 仅允许“唯一 LOAD + 唯一 STORE 且 lane 偏移一致”的原地模式，同 base 多次 STORE → 拒绝；异 base 仅当两者均为常量绝对地址且 `[base, base + n*elem_bytes)` 可证不重叠时允许；其余 → `REASON_ALIASING_STORE`。
8. 通过后构造 `_Plan`：`n, W, strips, rem`，以及按原顺序排列的映射（`LOAD→VLOAD`、`STORE→VSTORE`、`RELU→VRELU`、`ADD→VADD`…）。

### 4.5 重写算法

- 用 `_fresh()` 生成 strip iv 与全部向量中间值的 `Value`（`shape=(W,)`，`dtype` 继承被替换指令的 `dest.dtype`）；
- 地址链重建：`c_scale = load_const(W*elem_bytes)`（新的常量 Value）、`boff = mul(strip_iv, c_scale)`、`pa = add(base, boff)`；
- 原地替换区域内容：`FOR` 的 `dest` 换为 strip iv、`attrs` 换为 `{start:0, end:strips, step:1, vector_width:W, elem_bytes, orig_trip:n}`；把 `ADDR/CONST/ELEM` 指令替换为对应向量指令；`LOAD/STORE` 替换为 `VLOAD/VSTORE`；保留 `ENDFOR`；
- `_rewrite` 返回 `(vector_ops, new_end_index)`，其中 `new_end_index = for_index + 1 + len(new_body)` 是替换后 strip `ENDFOR` 的真实位置（`region.end_index` 在体长变化后失效，余数插入必须用它）；
- **禁改**：区域外指令、`FOR` 之前/`ENDFOR` 之后的指令一律不动。

### 4.6 余数克隆

```python
def _clone_remainder(block, region, plan, original_iv, new_end_index):
    if plan.rem == 0: return 0
    new_for = Instruction(opcode=OpCode.FOR, dest=original_iv, attrs={
        "start": plan.strips * plan.width, "end": plan.n, "step": 1})
    cloned = _clone_instrs(region.body, suffix="__rem")   # 深克隆 + def/use 同步改名
    insert_at = new_end_index + 1                         # strip ENDFOR 之后
    block.instructions[insert_at:insert_at] = [new_for, *cloned, endfor]
    return 1
```

余数块必须插在 strip `ENDFOR` 之后（与 strip 循环同级、且在 `return` 之前）；用旧的 `region.end_index + 1` 会导致新体更长时余数嵌进 strip 循环、新体更短时插到 `return` 之后成为死代码。`_skip_rewritten` 相应改为按块内顶层 `ENDFOR` 扫描定位下一次扫描起点。

`_clone_instrs` 规则：为每个 `dest` 生成新 `Value(name + "__rem")`；克隆指令的 `operands` 中若引用被克隆的旧名则替换为新名，否则保持（区域外引用与常量不动）；`FOR/ENDFOR` 不入克隆体。

### 4.7 确定性

- 遍历顺序 = 函数序 × 块序 × 指令序；
- `_fresh()` 单调计数；
- 不依赖 `set/dict` 迭代顺序做任何输出决策（集合只用于成员判定）。

---

## 五、后端标量降级：实现位置与算法

### 5.1 位置与接线

新增 `scratchv/backend/vector_scalar.py`（约 150 行），并在 `instruction_select.py` 三处接线：

```python
# instruction_select.py:19 __init__ 内
from scratchv.backend.vector_scalar import VectorScalarExpander
self._vector_expander = VectorScalarExpander()

# :38 _select_function 开头
self._vector_expander.begin_function(func.name)

# :47 _select_instruction 开头
if instr.opcode.is_vector():
    self._instructions.extend(self._vector_expander.expand(instr))
    return
```

不新增 `MachineOp`；不修改 `machine_types.py`、`asm_emit.py`、`regalloc_linear.py`、`register_alloc.py`。

### 5.2 展开器状态与算法

```python
class VectorScalarExpander:
    def __init__(self, *, default_width: int = 4) -> None:
        self._default_width = default_width
        self._lanes: dict[str, list[MachineOperand]] = {}
        self._counter = 0

    def begin_function(self, func_name: str) -> None:
        self._lanes.clear(); self._counter = 0

    def expand(self, instr: Instruction) -> list[MachineInstr]:
        op = instr.opcode
        if op is OpCode.VLOAD:  return self._expand_load(instr)
        if op is OpCode.VSTORE: return self._expand_store(instr)
        if op is OpCode.VBCAST: return self._expand_bcast(instr)
        if op in (VADD, VSUB, VMUL, VDIV): return self._expand_binary(instr)
        if op is OpCode.VRELU: return self._expand_unary(instr)
        raise VectorLoweringError(...)
```

核心辅助：

```python
def _lanes_of(self, value: Value, width: int) -> list[MachineOperand]:
    # 向量值 → 已绑定 lane 列表；常量 → LI 物化后广播（不缓存跨指令）
    ...

def _lane_addr(self, base: MachineOperand, k: int) -> list[MachineInstr]:
    # k==0: 直接返回 base；k>0: [ADDI tmp, base, 4*k]
    # 关键：只用 ADDI，禁止 ADD rd, rs, imm（编码器 R 型立即数会静默变 x0）
    ...
```

逐 op：

- `VLOAD`：`addr = _operand_reg(instr.operands[0])`（常量先 LI）；对 k：`a_k = _lane_addr(addr, k)`；`LW vd_k, a_k`；`self._lanes[dest.name] = [vd_k...]`。
- `VSTORE`：同地址链；`SW a_k, vs_k`（`vs_k` 从 `_lanes` 取）。
- `VBCAST`：`src = _operand_reg(scalar)`；`self._lanes[dest.name] = [src] * width`（无机器指令）。
- `VADD/VSUB/VMUL/VDIV`：逐 k `ADD/SUB/MUL/DIV vd_k, va_k, vb_k`；登记 lane。
- `VRELU`：逐 k `MAX vd_k, va_k, MachineOperand.immediate(0)`。
- 宽度一致性：所有 `expand` 先读 `int(instr.attrs.get("width", self._default_width))`；`VSTORE` 的 `vec` lane 数不匹配 → `VectorLoweringError`。

命名约定（确定性，测试可断言前缀）：lane 数据 vreg `f"{name}__lane{k}"`；lane 地址 vreg `f"{name}__addr{k}"`；常量物化 vreg `f"vcst_{counter}"`。

### 5.3 P0 前置修复：R 型常量操作数物化

**现状缺陷（已在 HEAD 复现）**：

```text
IR:  %off = mul %iv %c4        # c4 是 load_const(4) 的 dest（is_constant=True）
ASM: mul t2, t0, 4             # _op() 把常量变 immediate
BIN: 编码器 _reg_num("4") → 0  # riscv_encoder.py:103 对非寄存器文本静默返回 0
实际: mul t2, t0, x0           # 静默错误
```

**修复**（`instruction_select.py`，仅标量路径）：

```python
def _op_reg(self, instr: Instruction, idx: int) -> MachineOperand:
    """Like _op, but materializes constants into a vreg (encoder-safe)."""
    op = instr.operands[idx]
    if op.is_constant and op.const_value is not None:
        tmp = MachineOperand.vreg(f"const__{op.name}")
        self._emit(MachineOp.LI, tmp,
                   MachineOperand.immediate(int(op.const_value)),
                   comment=f"const {op.const_value}")
        return tmp
    return MachineOperand.vreg(op.name)
```

替换点：`_select_add`、`_select_sub`、`_select_mul`、`_select_div`（`instruction_select.py:88-102`）与 `_select_gelu` 的 `DIV dst, dst, immediate(2)`（`:140`）。保留 `_op()` 给立即数安全的消费者（`ADDI`、`MAX ... 0`、`SUB 0, rs`）。

**建议同时做的防护（可选 P0b，非向量功能）**：编码器对 R 型指令的非寄存器操作数 `raise ValueError`，消除“静默变 x0”。若实施，需单独提交并在 commit message 注明与课题 29 无关。

**回归测试**（`tests/test_backend.py` 追加）：编译含 `mul`/`add` 常量操作数的 IR，断言 asm 文本不匹配 `r"^\s*(add|sub|mul|div)\s+\w+,\s*\w+,\s*-?\d"`。

---

## 六、`riscv_encoder.py`：向量指令显式拒绝（一期）

`_encode_line`（`riscv_encoder.py:326`）在取到 `op` 后立即检查：

```python
VECTOR_MNEMONIC_RE = re.compile(
    r"^v(setvli|setivli|set|le|se|lw|sw|add|sub|mul|div|rem|max|min|"
    r"mv|fmv|fadd|fsub|fmul|fdiv|redsum|rgather|slide|merge|macc|nclip|"
    r"widen|narrow|and|or|xor)")
VECTOR_ENCODING_MSG = (
    "vector instruction '{op}' is not supported: "
    "ScratchV phase 1 targets RV32IM only")

class VectorEncodingError(ValueError):
    pass
```

```python
        op = tokens[0].lower()
        if VECTOR_MNEMONIC_RE.match(op):
            raise VectorEncodingError(VECTOR_ENCODING_MSG.format(op=op))
```

要点：

- 该护栏只做 **fail-loud**，不实现任何向量编码；RVV 32-bit 编码涉及 vtype/vreg/可变长度，明确不做；
- 正则只匹配 `v` 开头的 RVV/P 助记符；RV32IM 无 `v` 开头助记符，不产生误伤（新增标量指令若以 `v` 开头需评审）；
- 一期正常路径根本不会触碰该分支（降级后全为标量），它是“防止未来某处漏出向量 op”的保险丝；测试见 10.3。

---

## 七、仿真器无需改动的原因说明

**结论：一期不修改 `scratchv/simulator/rv32_emulator.py` 与 `scratchv/standalone/benchmark.py`。**

理由链：

1. **产物层面无向量指令**：`Vectorizer` 只改 IR；`VectorScalarExpander` 把每条向量 op 展开为 `LW/SW/ADDI/ADD/SUB/MUL/DIV/MAX`，`AsmEmitter` 产出的汇编与编码器产出的二进制里不存在任何 RVV/P 指令。仿真器“本来就只认 RV32IM”，因此无需扩展。
2. **仿真器没有可扩展点**：`RV32Emulator._execute`（`rv32_emulator.py:246`）按 7-bit opcode 分派，没有向量寄存器堆、没有 `vtype/vl/vstart` CSR、没有 RVV 编码表。为未发射的指令预埋解码/状态是死代码，违反“不做无验证的功能”。
3. **未知 opcode 是静默跳过**：`_execute` 对未识别 opcode 没有 `else` 分支，等价于 no-op。这意味着**一旦有向量指令漏进二进制，仿真器不会报错而是算错**。因此一期的防线必须是编码器拒绝（第六节）+ 展开后置断言，而不是“让仿真器报错”。这条是风险控制的核心，写入验收标准。
4. **benchmark.py 同理**：其分类器只识别 RV32IM opcode（`standalone/benchmark.py:48-65`），向量指令会落进 `unknown` 类且可能产生错误周期数。一期没有向量指令进入该工具，故不动；二期需要 VLEN 感知模型时才扩展，并作为二期依赖（第十三节）。
5. **可执行验证仍能覆盖语义**：标量展开后的程序与原标量程序共享同一套指令语义（同构展开），用现有 `RV32Emulator.run()` 对拍即可验证“向量化没有改变结果”。这正是把一期设计成“IR 向量 + 后端降级”的原因。

---

## 八、`compiler.py` 集成点

### 8.1 `CompilerConfig`（`compiler.py:33-75`）

追加三字段（见 1.5），并在 `docstring` 的 Attributes 列表补三行说明。默认值保证关闭时与现状逐字节一致。

### 8.2 `compile()` 中的插入点

```python
# --- 3. Optimize ---            # compiler.py:260-264 之后
if self.config.vectorize:
    vres = self._run_vectorizer(program)
    warnings.extend(vres.warnings)
    if vres.message:
        opt_message = (opt_message + "; " if opt_message else "") + vres.message

# --- IR dump ---                # compiler.py:266-278：此时 dump 可见向量 IR

# --- 4. Code generation ---     # compiler.py:280-287 之前插入 ISA 拒绝
if self.config.vectorize and self.config.vector_isa != "scalar":
    return CompileResult(
        success=False,
        errors=[f"vector-ISA '{self.config.vector_isa}' is not implemented "
                "(phase 2); use --vector-isa scalar"],
        ir_dump=ir_dump,
    )
```

新方法：

```python
    def _run_vectorizer(self, program) -> PassResult:
        from scratchv.optimizer.vectorize import Vectorizer
        vec = Vectorizer(program, width=self.config.vector_width)
        return vec.run(program)
```

DAG 兼容：`_generate_code` 若走 `use_dag_isel`（`compiler.py:393`），向量 IR 无法被 `scratchv_dag` 处理 → 在 `compile()` 的向量化分支里加：

```python
if self.config.use_dag_isel:
    warnings.append("vectorize is incompatible with --dag-isel; falling back to linear isel")
    self.config.use_dag_isel = False
```

### 8.3 warnings 透传

现状 `_run_optimizations` 的返回值只取 `message`（`compiler.py:263`），丢弃 `warnings`。向量化拒绝原因是 `warnings` 的主要出口，必须在新分支 `warnings.extend(vres.warnings)`（如上），不要改 `_PassAdapter`/`PassManager` 语义。

### 8.4 已知限制（集成时必须知晓，不在本期修）

- `reg_alloc="linear"` 的 `LinearScanAllocator.emit` 把标签输出为 `.label name`、跳转目标只放在注释，导致编码阶段标签丢失（实测 `j  # .Lloop_header_1` → `IndexError`）。CLI 默认 `--reg-alloc greedy`，向量化验证路径使用 greedy；`CompilerConfig` 默认 `"linear"` 是既有不一致，**已实现**：向量化开启且 `reg_alloc=="linear"` 时追加 warning 并回退 greedy（见 8.5）。
- `_select_alloca` 使用 `vreg("sp")` 会被寄存器分配改名（alloca 结果错误）。向量化样例与测试一律用 `load_const` 绝对地址作 base，避免 `alloca`；这是既有缺陷，单独跟踪。
- `--extended-isel`、`--verify-ir` 存在“CLI 已声明但 `args_to_config` 未接线”的历史缺口；本期新增三参数必须双侧修改，并以 10.4 的接线测试守门。

### 8.5 `compile()` 前置校验（评审修复 F5/F6/F7）

- `vectorize and backend == "llvm"` → 直接返回 `success=False`，错误消息指明 Phase 1 只支持 RISC-V 后端（LLVM 后端会把向量 op 写成注释丢弃，属静默错误产物）。
- `vector_width < 2`（或非 int）→ 在解析前返回 `success=False`，错误消息 `vector width must be an integer >= 2 (got ...)`；`Vectorizer.__init__` 内部同样调用 `validate_vector_width()` 抛 `ValueError`（API 防呆双保险）。
- `vectorize and reg_alloc == "linear"` → 追加 warning 并把 config 回退到 `"greedy"`（线性扫描标签发射缺陷，见 8.4 第一条）。

---

## 九、`main.py` CLI 接线

1. `build_arg_parser()`（`main.py:24-128`）在 “Topic module flags” 段（`:69-106`）追加 1.5 的三个 `add_argument`。
2. `args_to_config()`（`main.py:135-156`）在 `CompilerConfig(...)` 调用中追加三个关键字实参。
3. `main()` 无需改动：`--vectorize` 生效路径完全走 `CompilerDriver.compile()`。

---

## 十、测试文件与用例

### 10.1 `tests/test_vectorize.py`（新增）

| 用例 | 说明 | 关键断言 |
|---|---|---|
| `TestVectorIrBuilders::test_builder_emits_vector_ops` | 8 个 builder API 各构造一次 | opcode/attrs/`dest.shape` 精确匹配 1.1 |
| `TestVectorIrBuilders::test_vstore_returns_instruction` | `vstore` 返回值类型 | `isinstance(instr, Instruction)` 且 `dest is None` |
| `TestVectorizerStructure::test_strip_mining_no_remainder` | n=16,W=4 | FOR attrs `{0,4,1}` + `vector_width=4`；`2×VLOAD+VMUL+VRELU+VSTORE`；无第二个 FOR |
| `...::test_strip_mining_with_remainder` | n=17,W=4 | 第二个 FOR `start=16,end=17`，体内无向量 op；克隆 defs 不重名 |
| `...::test_width_2` | n=16,W=2（参数化） | strips=8，向量 attrs `width=2` |
| `...::test_trip_too_small_ir_unchanged` | n=3,W=4 | reject；`program.dump()` 前后相同 |
| `TestVectorizerRejects::test_reject_reasons` | 设计文档 I1–I4 | `reason` 字符串逐项相等；`changes == 0` |
| `TestVectorizeReport::test_report_counts` | 1 可向量化 + 1 拒绝 | `message == "vectorized 1/2 loop(s), width=4"`；warnings 各一行 |

构造“可向量化 IR”的辅助函数（两处复用，放本文件顶部）：

```python
def _make_map_loop(n: int) -> Program:
    b = IRBuilder(); b.new_function("main"); b.new_block("entry")
    a = b.load_const(0x400000, dtype=DataType.INT32)
    bb = b.load_const(0x410000, dtype=DataType.INT32)
    o = b.load_const(0x420000, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off)); vb = b.load(b.add(bb, off))
    r = b.relu(b.mul(va, vb))
    b.store(b.add(o, off), r)
    b.endfor(); b.ret()
    return b.program
```

### 10.2 `tests/test_vector_lowering.py`（新增，含现有仿真器执行验证）

| 用例 | 说明 |
|---|---|
| `TestLoweringSanity::test_no_vector_mnemonic_in_asm` | 向量化 → 后端 → asm；正则 `^\s*v[a-z]` 无命中；selector 结束态无向量 opcode |
| `TestLoweringSanity::test_lane_addressing_uses_addi` | asm 中 lane 地址用 `addi`；不出现 `add .* , -?\d+$` 这类 R 型立即数 |
| `TestEmulatorDifferential::test_vectorized_matches_scalar` | 见下 |
| `TestLoweringErrors::test_missing_lane_binding` | 手造只含 `VADD` 的孤儿 IR → `VectorLoweringError` |

**对拍骨架（已用当前 HEAD 预验证可行，含 P0 修复）**：

```python
def _compile_and_run(program, base_in_a=0x400000, base_in_b=0x410000,
                     base_out=0x420000, a=None, b=None) -> list[int]:
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.register_alloc import RegisterAllocator
    from scratchv.backend.asm_emit import AsmEmitter
    from scratchv.backend.riscv_encoder import assemble_to_binary
    from scratchv.simulator.rv32_emulator import RV32Emulator

    mi = InstructionSelector(program).run()
    asm = AsmEmitter(RegisterAllocator(mi, mode="greedy").run()).emit()
    bin_ = assemble_to_binary(asm)                    # P0 修复后编码正确

    emu = RV32Emulator()
    emu.load_code(bytes(bin_))                        # 代码从 0 起，数据区 ≥ 0x400000
    for i, x in enumerate(a): emu.write_i32(base_in_a + 4*i, int(x))
    for i, x in enumerate(b): emu.write_i32(base_in_b + 4*i, int(x))
    emu.run(max_instr=100000)                         # 停在 jalr x0, ra, 0
    return [emu.read_i32(base_out + 4*i) for i in range(len(a))]
```

断言：`run(S) == run(Vectorizer(S, width=4)) == [max(x*y, 0) for x, y in zip(a, b)]`（i32 回绕按 Python 位运算给出期望值）；输入含负数与 0；`emu._instr_count` 仅 `print` 记录，**不做性能断言**（避免把未测量数字变成 CI 门槛）。

### 10.3 `tests/test_vector_encoder.py`（新增）

```python
@pytest.mark.parametrize("text", [
    "vadd.vv v1, v2, v3",
    "vsetvli t0, a0, e32, m1, ta, ma",
    "vle32.v v1, (a0)",
    "vse32.v v1, (a0)",
    "vmv.v.x v1, a0",
])
def test_vector_mnemonic_rejected(text):
    with pytest.raises(VectorEncodingError) as ei:
        assemble_to_binary(text)
    assert "phase 1 targets RV32IM only" in str(ei.value)

def test_scalar_still_encodes():
    assert len(assemble_to_binary("add a0, a1, a2\n")) == 4
```

### 10.4 `tests/test_backend.py`（追加）

| 用例 | 说明 |
|---|---|
| `TestCLIWiring::test_vector_flags_parse_and_convert` | `build_arg_parser().parse_args(["m.onnx","--vectorize","--vector-width","2","--vector-isa","scalar"])` → `args_to_config()` 三字段断言（守门课题 28 的漏接线问题） |
| `TestCLIWiring::test_vector_isa_defaults_scalar` | 默认 `vectorize is False`、`vector_width == 4`、`vector_isa == "scalar"` |
| `TestDriverIntegration::test_compile_vectorized_program` | monkeypatch `CompilerDriver._parse` 返回 10.1 的 IR；`CompilerConfig(vectorize=True, reg_alloc="greedy")`；`compile()` 成功、asm 无 `v*`、`warnings` 含向量化 message |
| `TestDriverIntegration::test_vector_isa_v_rejected` | `vectorize=True, vector_isa="v"` → `result.success is False` 且错误文本含 `phase 2` |
| `TestConstantOperandMaterialization::test_no_rtype_immediate` | P0 回归（见 5.3） |

### 10.5 运行

```bash
python3 -m pytest tests/test_vectorize.py tests/test_vector_lowering.py \
                 tests/test_vector_encoder.py tests/test_backend.py -v
make test          # 全量回归
```

---

## 十一、实施顺序与验收标准

### 11.1 顺序（依赖串行）

```
P0 常量物化修复 + 回归            → P1 types/builder + 单测
→ P2 Vectorizer + IR 结构测试     → P3 expander + asm/仿真器对拍
→ P4 encoder 护栏 + 测试          → P5 compiler/main 接线 + CLI 测试
→ P6 全量回归 make test           → P7 文档同步（本目录两份文档）
```

### 11.2 验收标准（全部满足才可声明完成）

- [ ] `python3 -m pytest tests/ -v` 全绿（含新增 4 个测试文件），无 skip 新增。
- [ ] `--vectorize` 默认关闭时，同一输入产出的 asm 与基线逐字节相同（回归测试或人工 diff）。
- [ ] `Vectorizer` 对设计文档 2.2.1 的 C1–C7 逐条有测试覆盖；拒绝原因字符串与 1.2 完全一致。
- [ ] 向量化程序经 `InstructionSelector`（greedy 路径）→ `assemble_to_binary` → `RV32Emulator` 执行成功，输出与标量基线逐 i32 位相等（10.2）。
- [ ] 向量化产物的 asm 与 bin 中不存在任何 `v*` 助记符；编码器对 RVV 样例抛 `VectorEncodingError`（10.3）。
- [ ] `--vector-isa v|p` 返回 `CompileResult(success=False)`，错误文本含 `phase 2`（10.4）。
- [ ] CLI 三参数在 parser 与 `args_to_config` 双侧接通（10.4）。
- [ ] `scratchv/simulator/rv32_emulator.py`、`scratchv/standalone/benchmark.py` 的 `git diff` 为空。
- [ ] 文档中所有性能表述均带“未测量”标注；不出现任何测量口径的 SIMD 收益声明。
- [ ] commit message 英文（仓库规范），正文注明“phase 1: IR + strip-mining + scalar lowering; RVV deferred”。

### 11.3 建议提交切分（便于审查与回退）

1. `fix: materialize constant operands in instruction selection (prerequisite)`（P0）
2. `feat(ir): add phase-1 vector opcodes and builder APIs (topic 29)`
3. `feat(opt): add FOR strip-mining vectorizer (topic 29)`
4. `feat(backend): lower vector ops to scalar machine code (topic 29)`
5. `feat(encoder): reject vector mnemonics explicitly (topic 29)`
6. `feat(cli): wire --vectorize/--vector-width/--vector-isa (topic 29)`
7. `test(topic29): vector IR, lowering differential, encoder rejection`
8. `docs(topic29): design and development docs`

---

## 十二、风险与回退

| # | 风险 | 触发条件 | 缓解 | 回退 |
|---|---|---|---|---|
| R1 | P0 修复改变既有 asm，触发快照/计数类测试失败 | 有测试锁定 `mul ..., 4` 形态 | 先跑 `make test`；只改 R 型消费者；快照类测试若断言的是错误形态，修正期望并注明 | 单独 revert P0 commit；向量化样例改用无常量 IR（不推荐） |
| R2 | 向量模式匹配过窄，真实程序零命中 | 前端暂无数组/索引，IR 少有此形态 | 一期以手写 IR 测试覆盖；报告如实显示 `vectorized 0/N`；前端数组支持另立课题 | 关闭 `--vectorize`，零影响 |
| R3 | 匹配过宽导致语义错误 | 别名/依赖判定漏洞 | C7 收缩（读 base 集合与写 base 必须相容）；对拍用负数与大值覆盖回绕 | 拒绝策略收紧为“无别名才向量化” |
| R4 | lane 展开造成寄存器压力上升（W×活跃值） | 长表达式+W=4 | 默认 W=4，提供 `--vector-width 2`；线性扫描可溢出到栈（既有机制） | W=2 或关闭 |
| R5 | 向量 op 漏进后端/编码器 | 接线遗漏或未来改动 | `is_vector()` 分派 + `VectorLoweringError` + 编码器 `VectorEncodingError` 双层护栏；仿真器未知 opcode 静默跳过是必须避免的来源 | 恢复编码器拒绝测试 |
| R6 | 误把一期 loop 摊销当 SIMD 收益对外表述 | 文档/汇报 | 设计文档 5.2 与验收标准强制“未测量”标注 | 修正文案，不改代码 |
| R7 | `reg_alloc="linear"` 路径缺标签导致 `IndexError` | 用户显式 `--reg-alloc linear` 且含循环 | 向量化路径文档化要求 greedy；可选加 warning 回退 | 修复线性路径标签发射（另立 task） |
| R8 | 与课题 28 在 `types.py` 冲突 | 两课题同改枚举 | 尾部注释分区 + append-only 约定；实施前 rebase 后复核分区顺序 | 手工合并，保留双方分区 |

**总回退开关**：`--vectorize` 默认 `False`；删除 `optimizer/vectorize.py` 与 `backend/vector_scalar.py` 后，IR/builder 的新增枚举与 API 无调用方，不影响既有管线（最坏情况保留死代码，语义零影响）。

---

## 十三、二期 RVV 依赖清单

| # | 依赖 | 现状 | 二期需要 | 备注 |
|---|---|---|---|---|
| D1 | RVV 汇编/编码 | `riscv_encoder.py` 仅 RV32IM，且一期只加拒绝 | 文本发射不需要编码器；二进制路径需要 binutils ≥ 2.40 或 LLVM MC ≥ 14 的 `rv32gcv` 支持 | 零依赖约束 → 外部工具显式 opt-in |
| D2 | RVV 仿真器 | `RV32Emulator`/TinyFive 仅 RV32IM，无向量寄存器堆/CSR | Spike `--isa=rv32gcv` 或 QEMU `rv32,v=true` | 一期禁止预埋死代码 |
| D3 | Benchmark 周期模型 | `benchmark.py` 分类仅 RV32IM，无 VLEN 概念 | 新增向量类别与 VLEN 感知周期模型 | 未扩展前不得产出“RVV 实测”数字 |
| D4 | `vsetvli/vtype` 配置 | 无 | 固定 `SEW=32, LMUL=1`，`VLEN=128` 假设显式写入发射器参数并记录 provenance | 与 2.4 映射表一致 |
| D5 | 数据对齐策略 | 一期只需 4B 对齐 | `vle32.v` 非对齐行为依赖实现；不支持的系统需 peel 到 16B | 需在发射器里可配置 |
| D6 | 端到端测试链路 | 无 RVV 执行能力 | `tests/test_rvv_emit.py`：只断言文本，不执行；执行测试加 `pytest.mark.skipif(工具链缺失)` | 不允许用“文本正确”冒充“行为正确” |
| D7 | P-extension | 无 | **不做**：i32 lane 无法打包，需先量化到 i16（模型/前端改造） | 若未来做，需独立调研 |

---

## 附录 A：现有代码锚点（HEAD `d146515`）

| 锚点 | 位置 | 用途 |
|---|---|---|
| `OpCode` 枚举 | `scratchv/ir/types.py:14-72` | 追加向量分区 |
| `is_control_flow()` | `scratchv/ir/types.py:69` | `is_vector()` 放其后 |
| `IRBuilder._emit` | `scratchv/ir/builder.py:29-41` | 新增 v* API 复用 |
| `IRBuilder.reshape` | `scratchv/ir/builder.py:206-209` | 追加点 |
| `InstructionSelector._select_function` | `scratchv/backend/instruction_select.py:38-45` | `begin_function()` |
| `_select_instruction` | `:47-52` | `is_vector()` 分派 |
| `_op` | `:63-69` | P0 修复对照 |
| add/sub/mul/div 处理器 | `:88-102` | P0 替换点 |
| `_select_gelu` DIV | `:140` | P0 替换点 |
| `_select_endfor` | `:210-223` | +1 递增语义（strip 设计依据） |
| `_select_load/_select_store` | `:159-163` | `LW/SW` 形态 |
| `RISCVAEncoder._reg_num` | `scratchv/backend/riscv_encoder.py:103-111` | 静默返回 0 的根源 |
| `_encode_line` | `:326` | 护栏插入点 |
| `compile()` 步骤 | `scratchv/compiler.py:224-321` | 集成插入点 |
| `_run_optimizations` | `:364-382` | warnings 透传对照 |
| `_generate_code` | `:386-395` | DAG 回退点 |
| `build_arg_parser` | `scratchv/main.py:24-128` | CLI 追加 |
| `args_to_config` | `:135-156` | 接线追加 |
| `CompilerPass`/`PassResult` | `scratchv/pass_interface.py:34-88` | pass 基类契约 |
| `RV32Emulator._execute` | `scratchv/simulator/rv32_emulator.py:246-368` | 无向量状态、未知 opcode 静默 |
| benchmark 分类 | `scratchv/standalone/benchmark.py:48-65` | 无向量类别 |

## 附录 B：变更清单（实施后核对用）

```
新增：
  scratchv/optimizer/vectorize.py
  scratchv/backend/vector_scalar.py
  tests/test_vectorize.py
  tests/test_vector_lowering.py
  tests/test_vector_encoder.py
修改：
  scratchv/ir/types.py                 (+8 enum, +is_vector, +_VECTOR_OPS)
  scratchv/ir/builder.py               (+8 API, +_shape_width)
  scratchv/backend/instruction_select.py (+_op_reg/P0, +is_vector 分派, +begin_function)
  scratchv/backend/riscv_encoder.py    (+VectorEncodingError, +VECTOR_MNEMONIC_RE, +拒绝)
  scratchv/compiler.py                 (+3 config 字段, +_run_vectorizer, +ISA 拒绝, +warnings)
  scratchv/main.py                     (+3 CLI, +args_to_config)
  tests/test_backend.py                (+CLI 接线/driver 集成/P0 回归)
禁止改动（diff 必须为空）：
  scratchv/simulator/rv32_emulator.py
  scratchv/standalone/benchmark.py
  scratchv/standalone/onnx_to_riscv_standalone.py
```

---

## 实现结果（2026-09-14 集成）

> **集成 commit**：`02305c0`（`feat(topic29): add vector IR ops and strip-mining vectorizer with scalar lowering`）
> **集成位置**：`Seven_big_summary` 上第 11 个 topic commit（最后一个）
> **集成后全量**：`PYTHONPATH=. python3.11 -m pytest tests/ -q` → **1011 passed / 13 xfailed / 20 xpassed / 0 failed**

### 实现文件与要点

| 文件 | 要点 |
|------|------|
| `scratchv/ir/types.py` | `[Topic 29]` 分区 8 个向量 OpCode（排在 Topic 28 之后）+ `is_vector()` |
| `scratchv/ir/builder.py` | 8 个 `v*` API + `_shape_width` |
| `scratchv/optimizer/vectorize.py`（新增） | strip-mining 向量化器：C1–C7 判定、余数深克隆 |
| `scratchv/backend/vector_scalar.py`（新增） | 逐 lane 标量降级 |
| `scratchv/backend/instruction_select.py` | `is_vector` 分派 + P0 常量操作数修复 |
| `scratchv/backend/riscv_encoder.py` | 向量助记符拒绝护栏 |
| `scratchv/compiler.py`、`scratchv/main.py` | `--vectorize` / `--vector-width` / `--vector-isa` 双侧接线 |
| `tests/test_vectorize.py`、`tests/test_vector_lowering.py`、`tests/test_vector_encoder.py`、`tests/test_backend.py`（追加） | 新增 49 用例 |

### 测试数字

| 口径 | 结果 |
|------|------|
| 定向（4 个测试文件新增部分） | 49 用例 |
| 分支全量（cherry-pick 前） | 614 passed |
| 集成后全量 | 1011 passed / 13 xfailed / 20 xpassed / 0 failed |

可执行对拍证据：选择器 → greedy → AsmEmitter → `RV32Emulator` 三条链路逐 i32 位相等。

### 与本文档的偏差 / 未完成项

- 内存操作数改为 `ADDI a1, addr, 4k` + `lw/sw a1(0)`，规避既有编码器静默落 x0 的缺陷。
- VRELU 使用 zero 寄存器。
- NEG 按 unsupported-op 显式拒绝。
- 余数循环不再复查（不做二次向量化检查）。
- 二进制 op 对拍使用 W=2（受既有分配器 >19 vreg 误编译与仿真器 SW 缺陷限制）。

### 已知限制

- Phase 1 仅标量展开（`--vector-isa p/v` 显式拒绝）。
- 无任何性能声明（不把 loop 摊销当 SIMD 收益）。

---

## 评审修复（2026-09-14，分支 `impl/topic29`）

> 对应评审：`GaoMD/ScratchV/Review/分支评审-2026-09-14/topic29-review.md`（评审对象 `88d9eed`）

| 评审 ID | 修复 |
|---|---|
| F1（P0） | `_rewrite` 返回替换后 strip `ENDFOR` 的真实下标，`_clone_remainder` 以此插入余数块，`_skip_rewritten` 改为按块内顶层 `ENDFOR` 扫描定位；消除“长体嵌套 / 短体死代码”两种错位 |
| F2（P0） | 新增 C6 live-out 检查：区域内定义（含原 `iv`）在 `region.body` 之外被使用 → `region-value-escapes` 拒绝，消除悬空 SSA |
| F3（P1） | C7 收紧为同 base（唯一 LOAD+STORE、lane 一致）或异 base 常量绝对地址区间可证不重叠；其余异名可重叠 base → `aliasing-store` |
| F4（P1） | 新增余数错位结构回归（N<B / N>B × W=2/4）、IV/区域值 live-out、异名 base 别名、宽度矩阵对拍（W×{整除,余数}×{map,广播,原地}，其中 W=4+余数的 map/广播受既有分配器 >19 vreg 缺陷限制，仅做结构覆盖） |
| F5（P2） | `vectorize + backend=llvm` 在 `compile()` 前置拒绝（`success=False`） |
| F6（P2） | `validate_vector_width()`：`width < 2` 或非 int 抛 `ValueError`；`compile()` 前置校验返回 `success=False`（不再 `ZeroDivisionError`） |
| F7（P2） | `vectorize + reg_alloc=linear` 追加 warning 并回退 greedy（R7 落地） |

修复后全量：`PYTHONPATH=. python3.11 -m pytest tests/ -q` → **753 passed / 0 failed**（修复前 720 passed）。
