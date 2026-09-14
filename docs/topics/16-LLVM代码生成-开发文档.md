# 课题16 LLVM 代码生成后端（库路径）开发文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/backend/llvm_codegen.py`、`tests/test_llvm_codegen.py`、`tests/test_llvm_codegen_llvm_tools.py`（新增）  
> 配套文档：《设计文档.md》（同目录）——本文件是其实现指南，二者若冲突以设计文档为准  
> 前置环境：Python 3.11+（项目 venv）、`/usr/bin/llvm-as`（LLVM 10，已确认可用）、可选 `/usr/bin/lli`、`/usr/bin/opt`  

---

## 一、实施范围与接口契约

### 1.1 范围

**做**：

- 修复 `llvm_codegen.py` 的 SSA 唯一性、类型/常量合法性、`_emit_for/_emit_endfor` CFG、`_emit_br_if` 条件类型；
- 为 conv/gemm/matmul/dot/maxpool/softmax 实现真实循环 + GEP + MAC；gelu/sigmoid 改为单定义展开；
- 目标 triple 可配置（构造函数参数，默认省略）；
- 扩充/新增测试，含 `llvm-as`、`lli` 集成（缺失时 skip）。

**不做**：

- 不动 `scratchv/standalone/onnx_to_llvm_standalone.py` 及任何 standalone 文件；
- 不动 `scratchv/compiler.py`、`scratchv/main.py`、`scratchv/ir/*`；
- 不引入 llvmlite 或任何新依赖；
- 不做 mem2reg/循环展开/向量化等 IR→IR 优化（只要求正确可汇编）。

### 1.2 接口契约（精确名称）

#### 1.2.1 模块级常量与函数（`scratchv/backend/llvm_codegen.py`）

| 名称 | 签名 | 说明 |
|------|------|------|
| `_TYPE_MAP` | `dict[DataType, str]` | dtype→LLVM 基类型（保持） |
| `_LLVM_FLOAT` / `_LLVM_DOUBLE` / `_LLVM_I32` / `_LLVM_I64` | `str` 常量 | `"float"`/`"double"`/`"i32"`/`"i64"`（保持） |
| `_llvm_type` | `(dtype: DataType) -> str` | dtype→基类型（保持） |
| `_float_to_llvm_hex` | `(value: float) -> str` | **新增**，复制 standalone 算法：float32 舍入→double 位型→`0x%016X` |
| `_float_literal` | `(value: float) -> str` | **新增**，`0.0/1.0/-1.0` 短写，否则 `_float_to_llvm_hex` |
| `_llvm_const_val` | `(value: float \| int, ty: str) -> str` | 重写：浮点走 `_float_literal`，整型十进制 |
| `_llvm_const` | `(val: Value) -> str` | 按 `val.dtype` 调用 `_llvm_const_val`（保持签名） |
| `_is_pointer_value` | `(val: Value) -> bool` | **新增**，`bool(val.shape)` |

#### 1.2.2 异常

```python
class LLVMCodegenError(Exception):
    """Raised for unlowerable IR (missing operand, unmatched endfor, ...)."""
```

#### 1.2.3 `SSANamer`（新增类）

```python
class SSANamer:
    @staticmethod
    def sanitize(name: str) -> str: ...
    def fresh(self, hint: str = "r") -> str: ...          # -> "%hint_<n>"
    def fresh_label(self, hint: str = "bb") -> str: ...   # -> "hint_<n>"
    def register_definition(self, reg: str) -> None: ...  # 重复注册即抛错
    def registered(self, reg: str) -> bool: ...
```

#### 1.2.4 `LoopContext`（新增 dataclass）

```python
@dataclass
class LoopContext:
    ir_name: str | None   # IR 循环变量名（DSL for）；张量算子内部循环为 None
    ptr: str       # "%<iv>_ptr_<n>": alloca i32*
    value: str     # "%<iv>_ld_<n>": header 中 load 出的 i32 值
    header: str    # 标签名（无 %）
    body: str
    exit: str
    limit: int     # 循环上界（i32）
    step: int = 1  # 步长（i32）
```

#### 1.2.5 `LLVMCodegen` 公开 API（兼容保持不变）

```python
class LLVMCodegen:
    def __init__(self, program: Program, target_triple: str | None = None) -> None: ...
    def emit(self) -> str: ...
    def save(self, path: str) -> None: ...
```

- 位置参数 `program` 不变，`LLVMCodegen(program)` 全项目兼容（`compiler.py:390`、examples、benchmarks）；
- `target_triple=None` ⇒ 不输出 `target triple` 行；传字符串则原样输出。

#### 1.2.6 `LLVMCodegen` 内部 helper（实现契约，供 review/测试引用）

| 名称 | 签名 | 职责 |
|------|------|------|
| `_fresh` | `(hint: str = "r") -> str` | 委托 `SSANamer.fresh` |
| `_fresh_label` | `(hint: str = "bb") -> str` | 委托 `SSANamer.fresh_label` |
| `_value_ref` | `(val: Value) -> str` | 常量内联；否则查/建 SSA 引用 |
| `_value_type` | `(val: Value) -> str` | 标量/指针的 LLVM 类型 |
| `_bind` | `(name: str, ref: str, llvm_ty: str) -> None` | 写 `_named_values` + `_ref_types` |
| `_dest` | `(instr: Instruction) -> str` | 幂等分配 dst 引用 |
| `_dest_buffer` | `(instr: Instruction, count: int, elem_ty: str) -> str` | dst 缓冲：ALLOCA 复用 / 按 count 分配 |
| `_op` | `(instr: Instruction, idx: int) -> str` | 第 idx 操作数引用；缺失抛 `LLVMCodegenError` |
| `_ptr_of` | `(instr: Instruction, idx: int, ty: str = "float") -> str` | 操作数当指针用；标量 spill 到 `alloca` |
| `_alloc_slot` | `(elem_ty: str, count: int = 1, hint: str = "slot") -> str` | 入口 prologue alloca，返回指针 SSA 名 |
| `_materialize_const` | `(value: float \| int, ty: str, hint: str) -> str` | 常量实体化为 SSA 值 |
| `_coerce_operand` | `(ref: str, from_ty: str, to_ty: str, hint: str) -> str` | 混型算术转换：`sitofp`/`fptosi`；同型原样返回 |
| `_emit_binary` | `(instr, op: str) -> None` | add/sub/mul/div/fadd... 统一发射（内部先 `_coerce_operand`） |
| `_start_block` | `(label: str) -> None` | 结束当前块（必要时补 `br`）并打开新块 |
| `_terminate` | `(line: str) -> None` | 发射终止指令并置位 |
| `_ensure_terminator` | `() -> None` | 当前块无终止符时补 `br` 到合成续块 |
| `_loop_open` | `(limit: int, ir_name: str \| None, hint: str, start: int = 0, step: int = 1) -> LoopContext` | 循环规范形的前半；`ir_name` 非空时绑定 IR 循环变量 |
| `_loop_close` | `(ctx: LoopContext) -> None` | 循环规范形的后半 |
| `_dim_of` | `(instr: Instruction, keys: tuple[str, ...], default: int = 1, operand: int \| None = None, axis: int \| None = None) -> int` | 从 attrs/shape 取维度，键名兼容 |

#### 1.2.7 CLI（不变）

```bash
scratchv model.onnx --backend llvm -o out.ll     # 既有入口，行为=默认目标无关 IR
python -c "from scratchv.backend.llvm_codegen import LLVMCodegen; \
           open('o.ll','w').write(LLVMCodegen(p, 'riscv64-unknown-elf').emit())"  # 显式 triple
```

不新增命令行参数；triple 覆盖只走 Python API（范围约束）。

---

## 二、通用机制实现方案

### 2.1 张量表示与 slot 分配

**判定**：`_is_pointer_value(val)` 为真 ⇔ `val.shape` 非空；此外 `ALLOCA` 指令的 dest 显式绑定指针类型。

**指针类型表**（`_value_type`）：

```python
def _value_type(self, val) -> str:
    base = _llvm_type(val.dtype)
    if _is_pointer_value(val):
        return base + "*"
    ref = self._named_values.get(val.name)
    if ref is not None and self._ref_types.get(ref, "").endswith("*"):
        return self._ref_types[ref]
    return base
```

**入口 prologue**：`_alloc_slot()` 把 `%p = alloca <ty>, i32 <count>` 追加到 `self._prologue: list[str]`；`_emit_function` 在 `define ... {` 之后、第一个 `_emit_block` 之前输出 prologue（这些指令自动属于 entry 块）。这样循环内的 alloca 不会随迭代增长栈帧（`lli` 数值测试必需）。

**标量 spill**（`_ptr_of`）：

```python
def _ptr_of(self, instr, idx, ty="float"):
    val = instr.operands[idx]
    ref = self._op(instr, idx)
    if self._ref_types.get(ref, "").endswith("*"):
        return ref
    p = self._alloc_slot(ty, 1, "spin")        # 退化 1 元素张量
    self._p(f"  store {ty} {ref}, {ty}* {p}")
    return p
```

**结果缓冲**（`_dest_buffer`）：`ALLOCA` dest 直接返回其指针；`dest.shape` 非空返回 `_alloc_slot(elem_ty, prod(shape))` 并把 dest 名绑定该指针；否则分配 1 元素缓冲，算子执行完由调用方 `load` 出标量（见 3.2 通用尾巴）。

### 2.2 SSA 命名器

```python
class SSANamer:
    def __init__(self):
        self._reg_n = 0
        self._label_n = 0
        self._defs: set[str] = set()

    @staticmethod
    def sanitize(name: str) -> str:
        s = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        if not s or s[0].isdigit():
            s = "v_" + s
        return s

    def fresh(self, hint: str = "r") -> str:
        self._reg_n += 1
        return f"%{self.sanitize(hint)}_{self._reg_n}"

    def fresh_label(self, hint: str = "bb") -> str:
        self._label_n += 1
        return f"{self.sanitize(hint)}_{self._label_n}"

    def register_definition(self, reg: str) -> None:
        if reg in self._defs:
            raise LLVMCodegenError(f"duplicate SSA definition: {reg}")
        self._defs.add(reg)

    def registered(self, reg: str) -> bool:
        return reg in self._defs
```

要点：

- `_dest()` 幂等：若 `instr.dest.name` 已在 `_named_values` 中，直接返回旧引用，**不**调用 `_fresh`；
- 所有 `= ...` 发射点（含 `_materialize_const`、`_emit_*` 内部中间值）必须用 `_fresh` 并通过 `register_definition` 登记；
- 参数名不登记（参数定义不在函数体内），使用前先 `_bind`；
- `gelu/sigmoid` 等展开算子禁止 `_dest` 复用作为多行左值。

### 2.3 类型处理与常量

**分派规则**：

| 指令 | 类型来源 | 生成 |
|------|----------|------|
| `fadd/fsub/fmul/fdiv/fneg/fcmp` | `_infer_type(instr)`（dst 优先，其次首操作数） | 浮点指令 |
| `load` | `dest.dtype`（含指针判定） | `%d = load <ty>, <ty>* %p` |
| `store` | `operands[1].dtype`（值类型） | `store <ty> %v, <ty>* %p` |
| `load_const` | `dest.dtype` | 浮点：`fadd <ty> <hex>, 0.0`；整型：`add <ty> 0, <imm>` |
| `alloca` | `dest.dtype`，`attrs["size"]` 默认 4 | `%d = alloca <ty>, i32 <size>` |
| `for` | 计数固定 `i32` | 见第五章 |
| `return` | `_ref_types[ref]` 优先，其次操作数类型 | `ret float* %buf` 等 |
| 函数返回类型 | 首个 `RETURN` 操作数的 `_value_type` | 与 `ret` 一致 |

**常量编码**（P5/P6 修复）：

```python
def _float_to_llvm_hex(value: float) -> str:
    f32 = struct.unpack("<f", struct.pack("<f", value))[0]
    bits = struct.unpack("<Q", struct.pack("<d", float(f32)))[0]
    return f"0x{bits:016X}"

def _float_literal(value: float) -> str:
    if value == 0.0:  return "0.0"
    if value == 1.0:  return "1.0"
    if value == -1.0: return "-1.0"
    return _float_to_llvm_hex(value)

def _llvm_const_val(value, ty):
    if ty in ("float", "double"):
        assert isinstance(value, (int, float))
        return _float_literal(float(value))
    assert isinstance(value, int) or float(value).is_integer()
    return str(int(value))
```

**不变量**：浮点指令的参数串中不出现十进制指数形式；整型指令的参数串中不出现小数点。建议在 `_emit_binary`/`_emit_load_const` 后加开发期断言（`if ty in ("float","double"): assert "." in lit or "0x" in lit`）。

**混型算术协调**（DSL `for` 循环体常见 `s = add(s, i)`，`s: float`、`i: i32`）：

```python
def _coerce_operand(self, ref: str, from_ty: str, to_ty: str, hint: str) -> str:
    if from_ty == to_ty:
        return ref
    r = self._fresh(hint)
    self.namer.register_definition(r)
    if from_ty == "i32" and to_ty in ("float", "double"):
        self._p(f"  {r} = sitofp i32 {ref} to {to_ty}")
    elif from_ty in ("float", "double") and to_ty == "i32":
        self._p(f"  {r} = fptosi {from_ty} {ref} to i32")
    else:
        raise LLVMCodegenError(f"no coercion {from_ty} -> {to_ty}")
    return r

def _emit_binary(self, instr, fop: str, iop: str | None = None):
    ty = self._infer_type(instr)                     # 结果类型
    lhs = self._coerce_operand(self._op(instr, 0),
                               self._type_of_operand(instr, 0), ty, "cvt")
    rhs = self._coerce_operand(self._op(instr, 1),
                               self._type_of_operand(instr, 1), ty, "cvt")
    dst = self._dest(instr)
    op = fop if ty in ("float", "double") else iop
    self._p(f"  {dst} = {op} {ty} {lhs}, {rhs}")
```

`_type_of_operand` 返回操作数引用的 LLVM 类型（常量按 `val.dtype`）；`ty` 优先取 `dest.dtype`，dest 缺失时取浮点操作数类型。

### 2.4 控制流状态机

新增字段：`_terminated: bool`、`_defined_labels: set[str]`、`_prologue: list[str]`、`_loop_stack: list[LoopContext]`。

```python
def _start_block(self, label: str) -> None:
    if label in self._defined_labels:
        label = self._fresh_label(label)          # 绝不重复
    if not self._terminated:
        self._p(f"  br label %{label}")           # 显式落空转跳转
    self._p(f"{label}:")
    self._defined_labels.add(label)
    self._terminated = False

def _terminate(self, line: str) -> None:
    if self._terminated:
        raise LLVMCodegenError("terminator emitted twice in one block")
    self._p(f"  {line}")
    self._terminated = True

def _ensure_terminator(self) -> None:
    if not self._terminated:
        self._p(f"  br label %{self._fresh_label('cont')}")
        self._terminated = True
```

- `_emit_block` 改为：首块不打印标签（entry 隐式），其余块 `_start_block(block.name)`；块注释保留；
- `_emit_instruction` 若在 `_terminated` 状态下收到非终止指令，先 `_start_block(self._fresh_label("dead"))`，保证不产生“终止符后接指令”的非法文本；
- `_emit_br/_emit_br_if/_emit_return` 全部改调 `_terminate`。

### 2.5 `_emit_for` / `_emit_endfor` 修复方案

**修复目标**：preheader 跳 header 而非 body；循环上下文栈化；IV 有定义；嵌套标签不重复不悬空（修复 P3/P4）。

```python
def _emit_for(self, instr):
    self._dest(instr)                                  # 占位登记，值随后绑定
    limit = int(instr.attrs.get("end", 0))
    start = int(instr.attrs.get("start", 0))
    step = int(instr.attrs.get("step", 1))
    ctx = self._loop_open(limit, instr.dest.name, "loop_i", start, step)
    self._loop_stack.append(ctx)

def _loop_open(self, limit, ir_name, hint, start=0, step=1):
    ptr = self._alloc_slot("i32", 1, "iv_ptr")
    self._p(f"  store i32 {start}, i32* {ptr}")
    header = self._fresh_label(f"{hint}_hdr")
    body = self._fresh_label(f"{hint}_bdy")
    exit_ = self._fresh_label(f"{hint}_ext")
    self._start_block(header)                          # 自动补 br label %header
    iv = self._fresh(f"{hint}_ld")
    self._p(f"  {iv} = load i32, i32* {ptr}")
    self.namer.register_definition(iv)
    cond = self._fresh(f"{hint}_cond")
    self._p(f"  {cond} = icmp slt i32 {iv}, {limit}")
    self._terminate(f"br i1 {cond}, label %{body}, label %{exit_}")
    self._start_block(body)
    if ir_name is not None:                            # 仅 DSL for 需要绑定 IR 循环变量
        self._bind(ir_name, iv, "i32")                 # header load 支配 body+exit
    return LoopContext(ir_name, ptr, iv, header, body, exit_, limit, step)

def _loop_close(self, ctx):
    if not self._loop_stack or self._loop_stack[-1] is not ctx:
        raise LLVMCodegenError("endfor without matching for")
    self._loop_stack.pop()
    if not self._terminated:                           # body 以 ret 结束时为死块，跳过回边
        cur = self._fresh("iv_cur")
        self._p(f"  {cur} = load i32, i32* {ctx.ptr}")
        self.namer.register_definition(cur)
        nxt = self._fresh("iv_nxt")
        self._p(f"  {nxt} = add i32 {cur}, {ctx.step}")
        self.namer.register_definition(nxt)
        self._p(f"  store i32 {nxt}, i32* {ctx.ptr}")
        self._terminate(f"br label %{ctx.header}")
    self._start_block(ctx.exit)

def _emit_endfor(self, instr):
    if not self._loop_stack:
        raise LLVMCodegenError("endfor without matching for")
    self._loop_close(self._loop_stack[-1])
```

**边界**：

- 循环变量在 `endfor` 之后的引用：绑定的是 header 的 load，仍支配 exit，值等于终止时的 IV（合法且语义可解释）；
- 空循环体：body 与 header 同名结构仍合法；`_ensure_terminator` 保证 body 有回边；
- `for` 出现在已终止块之后：`_start_block(header)` 会先补 `br`，但这时其实应已由 `_emit_instruction` 开了 `dead` 合成块，无需额外处理。

### 2.6 `_emit_br_if` 修复方案（P7）

```python
_CMP_PRED = {"==": ("oeq", "eq"), "!=": ("one", "ne"),
             "<": ("olt", "slt"), "<=": ("ole", "sle"),
             ">": ("ogt", "sgt"), ">=": ("oge", "sge")}

def _emit_br_if(self, instr):
    targets = (instr.target or ",").split(",")
    true_t = targets[0].strip()
    false_t = targets[1].strip() if len(targets) > 1 else true_t
    cmp_op = instr.attrs.get("cmp_op")
    if cmp_op and len(instr.operands) >= 2:
        lhs, rhs = self._op(instr, 0), self._op(instr, 1)
        ty = self._infer_type(instr)
        fpred, ipred = self._CMP_PRED[str(cmp_op)]
        pred = fpred if ty in ("float", "double") else ipred
        kind = "fcmp" if ty in ("float", "double") else "icmp"
        cond = self._fresh("brc")
        self._p(f"  {cond} = {kind} {pred} {ty} {lhs}, {rhs}")
        self.namer.register_definition(cond)
    else:
        cond = self._op(instr, 0)                      # 已是 i1
    self._terminate(f"br i1 {cond}, label %{true_t}, label %{false_t}")
```

映射覆盖 `ExtendedDSLParser._parse_condition` 的六种运算符；`IRBuilder.br_if` 的 `operands=[cond]` 形态不受影响。

---

## 三、逐算子实现方案

统一约定：`acc`/`max`/`sum` 等所有 alloca 走 `_alloc_slot`（入口 prologue）；循环走 `_loop_open/_loop_close`；`hint` 见设计文档 2.3 表；结果写入 `_dest_buffer` 得到的缓冲，标量 dest 在算子末尾 `load` 首元素。

### 3.1 dot

```python
def _emit_dot(self, instr):
    ty = self._infer_type(instr)                       # "float"/"double"
    a = self._ptr_of(instr, 0, ty)
    b = self._ptr_of(instr, 1, ty)
    n = int(instr.attrs.get("length", instr.attrs.get("len", 1)))
    acc = self._alloc_slot(ty, 1, "dot_acc")
    self._p(f"  store {ty} 0.0, {ty}* {acc}")
    ctx = self._loop_open(n, None, "dot_i")
    ap = self._gep(ty, a, ctx.value, "dot_ap")
    av = self._load(ty, ap, "dot_av")
    bp = self._gep(ty, b, ctx.value, "dot_bp")
    bv = self._load(ty, bp, "dot_bv")
    pr = self._bin("fmul", ty, av, bv, "dot_pr")
    old = self._load(ty, acc, "dot_old")
    nw = self._bin("fadd", ty, old, pr, "dot_new")
    self._p(f"  store {ty} {nw}, {ty}* {acc}")
    self._loop_close(ctx)
    self._finish_scalar_result(instr, acc, ty)         # 见 3.2
```

需要的微 helper（也供其余算子复用）：

```python
def _gep(self, ty, base, idx, hint):     # %r = getelementptr ty, ty* base, i32 idx
def _load(self, ty, ptr, hint):          # %r = load ty, ty* ptr
def _bin(self, op, ty, lhs, rhs, hint):  # %r = op ty lhs, rhs
```

### 3.2 结果收尾与维度解析

```python
def _finish_scalar_result(self, instr, buf_ptr, ty):
    if _is_pointer_value(instr.dest):                  # 张量 dest：绑定缓冲指针
        self._bind(instr.dest.name, buf_ptr, ty + "*")
    else:                                              # 标量 dest：load 首元素
        r = self._load(ty, buf_ptr, "res")
        self._bind(instr.dest.name, r, ty)

def _dim_of(self, instr, keys, default=1, operand=None, axis=None):
    for k in keys:
        v = instr.attrs.get(k)
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
    if operand is not None and operand < len(instr.operands):
        shape = instr.operands[operand].shape
        if shape:
            idx = axis if axis is not None else 0
            if len(shape) > abs(idx):
                return int(shape[idx])
    return default
```

### 3.3 matmul

```python
def _emit_matmul(self, instr):
    ty = self._infer_type(instr)
    a = self._ptr_of(instr, 0, ty)
    b = self._ptr_of(instr, 1, ty)
    m = self._dim_of(instr, ("m", "rows"), 1, operand=0, axis=0)
    k = self._dim_of(instr, ("k", "inner"), 1, operand=0, axis=1)
    n = self._dim_of(instr, ("n", "cols"), 1, operand=1, axis=1)
    c = self._dest_buffer(instr, m * n, ty)
    acc = self._alloc_slot(ty, 1, "mm_acc")
    ci = self._loop_open(m, None, "mm_i")
    cj = self._loop_open(n, None, "mm_j")
    self._p(f"  store {ty} 0.0, {ty}* {acc}")
    ck = self._loop_open(k, None, "mm_k")
    # --- innermost body (i32 offsets) ---
    # aoff = add (mul ci.value, k), ck.value
    # av   = load(gep(a, aoff))
    # boff = add (mul ck.value, n), cj.value
    # bv   = load(gep(b, boff))
    # pr   = fmul av, bv ; acc = fadd load(acc), pr ; store acc
    self._loop_close(ck)
    # coff = add (mul ci.value, n), cj.value ; store load(acc) -> gep(c, coff)
    self._loop_close(cj)
    self._loop_close(ci)
    self._finish_scalar_result(instr, c, ty)
```

（文档中的缩进仅表嵌套层次；实现时按 `_loop_open/_loop_close` 顺序配对。）

### 3.4 gemm

维度：`M=_dim_of(attrs ("M",), A.shape[0])`、`K=A.shape[1]`；`trans_b = bool(attrs.get("trans_b", attrs.get("transB", False)))`；`N = W.shape[0] if trans_b else W.shape[1]`，退化 1。循环体：

```python
acc = bias[j]                               # load(gep(bias, cj.value))
for kk:
    av = A[i*K + kk]
    woff = j*K + kk if trans_b else kk*N + j
    wv = W[woff]
    acc += av*wv
C[i*N + j] = acc
```

`bias` 为标量时 `_ptr_of` spill；`attrs` 中 `trans_a` 暂不支持（如为真，抛 `LLVMCodegenError` 并注释说明）。

### 3.5 conv

维度解析优先级：`attrs` → `operand.shape` → 退化值。

```python
x = self._ptr_of(instr, 0, ty); w = self._ptr_of(instr, 1, ty); bias = self._ptr_of(instr, 2, ty)
xs = instr.operands[0].shape          # 取后三维 (C,H,W)
cin  = xs[-3] if len(xs) >= 3 else 1
h    = xs[-2] if len(xs) >= 3 else 1
ww   = xs[-1] if len(xs) >= 3 else 1
ws = instr.operands[1].shape
cout = ws[0] if len(ws) >= 4 else _dim_of(instr, ("out_channels",), 1)
k    = ws[2] if len(ws) >= 4 else _dim_of(instr, ("kernel_size", "kernel_shape"), 3)
s    = _dim_of(instr, ("stride", "strides"), 1)
p    = _dim_of(instr, ("padding", "pads"), 0)
ho = (h + 2*p - k)//s + 1; wo = (ww + 2*p - k)//s + 1
# 若 len(instr.operands) < 3（无 bias），bias 用 0.0 常量替代（_materialize_const），不调用 _ptr_of
```

循环体（6 层，oc/oh/ow/ic/kh/kw）：每层用 `_loop_open`；kh 内计算 `ih`、`ok_h`；kw 内计算 `iw`、`ok`，并用 `br i1 ok, label %mac, label %skip` 包住 MAC；`mac`/`skip` 用 `_fresh_label("conv_mac"/"conv_skip")` 并 `_start_block`。MAC 地址：输入 `ic*H*W + ih*W + iw`，权重 `oc*Cin*K*K + ic*K*K + kh*K + kw`；kw 退出后写 `out[oc*Ho*Wo + oh*Wo + ow] = acc`。

### 3.6 maxpool

`C/H/W` 来自 `operands[0].shape`（取后三维，退化 1）；`k=_dim_of(("kernel","kernel_shape"),2)`、`s=_dim_of(("stride","strides"),2)`；`ho=(h-k)//s+1`、`wo=(w-k)//s+1`。5 层循环；每 (c,oh,ow) 初始化 `m=-3.4e38`（`_float_literal`），内两层 `fcmp ogt + select` 更新，循环结束写回。

### 3.7 softmax

`n = _dim_of(instr, ("length", "n"), 1, operand=0, axis=-1)`；输出缓冲 `_dest_buffer(instr, n, ty)`。

- pass1（标签 `sm_max`）：`m=-3.4e38`；`m = select(fcmp ogt x[i], m, x[i], m)`；
- pass2（`sm_sum`）：`s += expf(x[i] - m)`；`fsub` 后 `call @expf`（f64 用 `@exp`）；
- pass3（`sm_div`）：`out[i] = expf(x[i] - m) / s`；
- 三趟各自独立 `_loop_open/_loop_close`；结果收尾同 3.2。

### 3.8 gelu

按设计文档 2.5.7 的九行展开实现，全部 `_fresh` + `register_definition`；常量 `0.044715`、`0.7978845608028654` 经 `_float_literal` 内联；`float` 调 `@tanhf`，`double` 调 `@tanh`；最后一行 `_dest` 只定义一次并 `_finish_scalar_result` 绑定。

### 3.9 sigmoid

按设计文档 2.5.8 的四行展开实现；`float` 调 `@expf`，`double` 调 `@exp`。

### 3.10 reshape / relu / exp / 算术

- `reshape`：同 dtype 直通，`_bind(dest.name, _value_ref(src), _value_type(src))`，不发指令（当前 `fadd x, 0.0` 保留也可，但要求整型安全：按类型选 `fadd`/`add`）；
- `relu`：`fcmp ogt + select`（保持）；
- `exp`：`call @expf/@exp`（保持）；
- `add/sub/mul/div/neg`：`_emit_binary` 统一按 `_infer_type` 分派浮点/整型指令（整型用 `add/sub/mul/sdiv`），避免 int 走 `fadd`。

---

## 四、测试文件与用例

### 4.1 文件清单

| 文件 | 内容 |
|------|------|
| `tests/test_llvm_codegen.py`（扩充） | 纯 Python 单元断言：SSA 唯一、标签唯一、常量写法、8 算子结构关键词 |
| `tests/test_llvm_codegen_llvm_tools.py`（新增） | `llvm-as` 可汇编测试 + `lli` 数值测试；工具缺失 skip |

### 4.2 `tests/test_llvm_codegen.py` 扩充用例（无外部工具依赖）

```python
def _name_defs(ir: str) -> list[str]:
    return re.findall(r"^\s*(%[A-Za-z0-9_.]+)\s*=", ir, re.M)

def test_no_duplicate_ssa_gelu_sigmoid(...):   # set(defs) 数量 == len(defs)
def test_for_labels_unique(...):               # 每个 label 形如 ^\s*(\w+):$ 唯一
def test_float_const_is_hex(...):              # 无 "e-0" 指数常量；含 0x
def test_int_const_not_float_op(...):          # 无 "fadd i32"
def test_all_ops_emit_loops(...):              # 8 算子每个含 "getelementptr"/"icmp slt"
```

### 4.3 `llvm-as` 集成测试（环境缺失 skip）

```python
import shutil, subprocess, pytest

LLVM_AS = shutil.which("llvm-as")
LLI = shutil.which("lli")
requires_asm = pytest.mark.skipif(LLVM_AS is None, reason="llvm-as not installed")
requires_lli = pytest.mark.skipif(LLI is None, reason="lli not installed")

def _assemble(ir: str, tmp_path):
    ll = tmp_path / "m.ll"; ll.write_text(ir)
    return subprocess.run([LLVM_AS, str(ll), "-o", str(tmp_path / "m.bc")],
                          capture_output=True, text=True)
```

用例矩阵（对应设计文档第三章）：

| 用例 | 输入 | 断言 |
|------|------|------|
| `test_asm_gelu_sigmoid` | IRBuilder 展开 | rc==0；无 `multiple definition`；定义次数唯一 |
| `test_asm_for_nested` | DSLParser 双重 for | rc==0；`loop_i_hdr` 先于 body；标签计数为 1 |
| `test_asm_tensor_ops` | dot/matmul/gemm/maxpool/conv 五程序 | rc==0；GEP/MAC/循环结构 |
| `test_asm_softmax` | shape=(2,) 与 (1,) | rc==0；三趟循环标签齐全 |
| `test_asm_onnx_cnn` | `models/graph/cnn.onnx`（存在则跑） | rc==0；无占位注释 |

### 4.4 `lli` 数值测试（harness 拼接法）

测试端构造程序：被测函数命名 `kernel`，参数用 `IRBuilder.make_value` 后设 `shape` 得到 `float*`；生成后拼接手写 `@main`：

```llvm
define i32 @main() {
  %a = alloca float, i32 4
  ; store 常量数组 ...
  %r = call float @kernel(float* %a, float* %b)
  %ten = sitofp i32 10 to float          ; 期望值 10.0，避免十进制浮点字面量
  %ok = fcmp oeq float %r, %ten
  %rc = select i1 %ok, i32 0, i32 1
  ret i32 %rc
}
```

然后用 `LLI m.bc` 执行并断言 `returncode == 0`。用例与期望值：

| 算子 | 输入 | 期望 |
|------|------|------|
| gelu→sigmoid | x=0.0 | 0.5 |
| 嵌套 for | i,j∈[0,3)，`s+=i` | 9.0 |
| dot | a=[1,2,3,4], b=[1,1,1,1] | 10.0 |
| matmul 1×1 | a=2, b=3 | 6.0 |
| gemm 1×1 | a=2, w=3, bias=0.5 | 6.5 |
| maxpool 2×2 | x=[1,2,3,4], K=2,S=1 | 4.0 |
| conv 1×1×1 | x=3, w=2, bias=1, K=1,S=1,P=0 | 7.0 |
| softmax N=2 | x=[0,0] | out=[0.5,0.5] |
| softmax N=1 | x=[7] | out=[1.0] |

注意：这些期望值全部可被 float32 精确表示，用 `fcmp oeq` 判定不引入容差问题。若 `lli` 存在但符号解析失败（`@expf` 未注册等），把对应激活类用例标记为 `xfail(strict=False)` 并在测试注释中说明；张量类用例不依赖 libm，必须真实执行。

### 4.5 回归清单

```bash
python -m pytest tests/test_llvm_codegen.py tests/test_llvm_codegen_llvm_tools.py -v
python -m pytest tests/ -q                       # 全量，无新增失败
llvm-as /tmp/out.ll -o /dev/null                 # 手工冒烟（cnn.onnx 输出）
```

---

## 五、验收标准

1. **可汇编（硬门槛）**：8 个算子的样例 IR、DSL `for/if/while` 样例、`models/graph/cnn.onnx` 经 `LLVMCodegen.emit()` 后 `llvm-as` 全部退出码 0；输出中不存在 `; UNSUPPORTED`、`; placeholder`、`; passthrough`。
2. **结构断言**：
   - 每个函数内 SSA 定义名唯一（正则计数 == set 大小）；
   - 每个标签唯一定义，且 preheader 分支目标为 header；
   - 每个基本块以 `br`/`ret` 结尾；
   - 张量算子至少含 `getelementptr <ty>, <ty>*`、`fmul`+`fadd`、`icmp slt i32`、`br i1`；循环层数符合：conv 6、maxpool 5、gemm/matmul 3、dot 1、softmax 3 趟。
3. **数值断言**：4.4 表格全部通过（`lli` 环境存在时必须真实执行；缺失则 skip，不得以 skip 充当通过）。
4. **类型断言**：浮点常量全部为 `0x` 十六进制或 `0.0/1.0/-1.0`；整型常量不出现在浮点指令中。
5. **兼容性**：`tests/test_llvm_codegen.py` 原有用例全绿；`benchmarks/test_benchmark.py::test_codegen_llvm` 通过；`LLVMCodegen(program)` 单参数调用可用。
6. **范围纪律**：`git diff --stat` 仅包含 `scratchv/backend/llvm_codegen.py` 与两个测试文件。

---

## 六、风险与回退

| 风险 | 概率/影响 | 缓释 | 回退 |
|------|-----------|------|------|
| 标量操作数进入张量算子所需元素数 > 1（如 `dot(a,b,len:8)` 中 a/b 是标量） | 中/中 | 需求元素数 > 1 时抛 `LLVMCodegenError`（`_require_elements`），不再 spill 成 1 元素缓冲后越界读；needed==1 的退化仍支持 | 如评审要求完整向量语义，需扩 IR（超范围，另立课题） |
| DSL 循环/分支变量为静态 SSA（无 phi）：累加不生效、while 条件不可变、if/else 合流未定义 | 高/高 | 文档 §5.7 锁定实际行为并补 `lli`/结构回归；while 不可能退出时 fail-loud；正确累加用 IRBuilder alloca/load/store | 前端变量降级为 alloca/load/store 属独立课题（评审 §4.1） |
| ONNX 路径参数变 `float*`、返回 `float*` 改变模块形态 | 中/中 | `benchmarks` 仅断言非空；无其他测试依赖具体签名 | 保持标量签名 + 内部 spill，放弃指针优化（改回点在 `_value_type`） |
| `_start_block` 自动补 `br` 改变 IR 文本，既有结构断言用例误判 | 中/低 | 同步更新 `tests/test_llvm_codegen.py` 中 `": " in ir` 类弱断言 | 保留旧的 `_emit_block` 分支作为开关（不推荐） |
| `lli` 数值测试受 libm 符号/平台影响 | 中/低 | 张量用例零 libm；激活用例 `xfail(strict=False)` | 数值验证降级为“结构 + 常量折叠冒烟” |
| LLVM 版本差异（十进制浮点、attr 语法） | 低/高 | 全部浮点常量走十六进制；外部声明只用 Llvm 10 已有语法 | 以 `/usr/bin/llvm-as` 实测为准，修正编码 |
| 重构范围蔓延到 optimizer/backend 其他文件 | 低/高 | 实施顺序按第七节，单文件提交；scope 检查用 `git diff --stat` | 按文件粒度 `git checkout --` 回退非目标文件 |

整体回退策略：本课题所有改动集中在 `llvm_codegen.py` 与测试；如集成失败，`git revert` 对应提交即可恢复旧行为（占位实现仍可汇编出错误数值的 IR，不影响 RISC-V 主路径）。

---

## 七、实施顺序（建议提交粒度）

1. **命名器 + 类型/常量**：`SSANamer`、`_float_literal`、`_llvm_const_val`、`_emit_load_const`、`_value_type`/`_ref_types`；跑通 `gelu/sigmoid` 的 `llvm-as`（修 P1/P2/P5/P6）。
2. **控制流状态机**：`_start_block`/`_terminate`/`_loop_open`/`_loop_close`、`_emit_br_if` cmp_op；跑通 `for`（含嵌套）与 `if/while` 的 `llvm-as`（修 P3/P4/P7）。
3. **基础设施**：`_alloc_slot` prologue、`_ptr_of`、`_dest_buffer`、`_gep/_load/_bin`。
4. **张量算子**：dot → matmul → gemm → maxpool → conv → softmax，逐个补 `llvm-as` 与 `lli` 用例。
5. **triple 可选化** + 文档注释清理（删除 placeholder 文本）。
6. **测试收口**：全量 pytest、`llvm-as` ONNX 冒烟、验收清单逐条勾选。

每步结束运行：`python -m pytest tests/test_llvm_codegen.py -q && llvm-as <样例> -o /dev/null`；完成后按项目 Harness 规则执行 self-review 与验证（本任务仅编写文档，不改仓库、不做 git 操作）。

---

## 实现结果（2026-09-14 集成）

> **集成 commit**：`bd0db49`（`feat(topic16): implement real NN op lowering and fix invalid LLVM IR`）
> **集成位置**：`Seven_big_summary` 上第 4 个 topic commit（顺序 06 → 07 → 09 → **16** → 17 → …）
> **集成后全量**：`PYTHONPATH=. python3.11 -m pytest tests/ -q` → **1011 passed / 13 xfailed / 20 xpassed / 0 failed**

### 实现文件与要点

| 文件 | 要点 |
|------|------|
| `scratchv/backend/llvm_codegen.py` | 重写（+1121 / −321）：SSA 命名、类型/常量、控制流状态机、8 个 NN 算子真实 lowering |
| `tests/test_llvm_codegen_topic16.py` | 47 个新测试（含 `lli` 数值断言；外部工具缺失时 skip） |

### 测试数字

| 口径 | 结果 |
|------|------|
| 定向（`tests/test_llvm_codegen_topic16.py`） | 47 passed（含 lli 数值） |
| 分支全量（cherry-pick 前） | 612 passed |
| 集成后全量 | 1011 passed / 13 xfailed / 20 xpassed / 0 failed |

### 与本文档的偏差 / 未完成项

- 标量变量统一使用 alloca + load/store（不再区分寄存器直出形态）。
- DSL 嵌套 `for` 的数值断言在修复轮（2026-09-14 阶段 2）已补齐 `lli`：静态 SSA 语义下实际值为 2.0，设计文档 §5.7 与 `test_lli_dsl_nested_for_static_ssa_value` 锁定。
- DSL `while` 条件变量重赋值被 `_check_unbounded_loops` fail-loud 拒绝（原实现会生成死循环）；`if/else` 合流读最后解析分支的行为由结构测试锁定。
- 多元素 initializer 在无 `shape` 信息时退化为 1 元素。
- `gemm trans_a=True` 明确抛错（不支持）。

### 修复轮补丁要点（2026-09-14 阶段 2，评审 F1–F5/F7/F9）

| 评审项 | 修复方式 | 代码位置 |
|--------|----------|----------|
| F1（P0） | 设计文档期望值修正 + 已知限制 §5.7；while 不可退出 fail-loud；nested-for/loop-carried/if-else-merge 回归锁定 | `_check_unbounded_loops` |
| F2（P1） | shaped dest 的元素级算子改走逐元素 `_emit_map`（含标量/单元素广播），`_dest` 拒绝张量目标 | `_emit_map`、`_dest` |
| F3（P1） | 张量算子按缓冲真实元素数校验，标量供多元素即抛错 | `_require_elements` |
| F4（P1） | softmax 按 `prod(shape[:-1])` 行循环、缓冲 `prod(shape)`；仅支持 axis=-1 | `_emit_softmax` |
| F5（P1） | 混合 `ret <value>`/`ret void` 与不一致返回类型 fail-loud | `_validate_return_types` |
| F7（P2） | 无 handler 的 opcode（transpose/concat）抛 `LLVMCodegenError`，删除零值伪计算 | `_emit_unsupported` |
| F9（P2） | 补 conv padding/stride、gemm trans_b 非方阵、matmul 非方阵、maxpool stride2、softmax 3 元素容差、for step=2、trans_a 负例等 `lli` 用例 | `tests/test_llvm_codegen_topic16.py` |

### 已知限制

- `llvm-as` / `lli` 环境缺失时相关用例 skip，不得以 skip 充当通过。
- ONNX 路径 `float*` 签名与返回类型变化仍按文档风险表处理；无其他调用方依赖具体签名。
- DSL 循环/分支变量语义仍为静态 SSA（根因在 `dsl_parser.py:114` 的 `_vars` 前端，评审 §4.1 列为既有仓库缺陷）；本分支仅做守卫/降级与文档锁定，不实现变量身份。
