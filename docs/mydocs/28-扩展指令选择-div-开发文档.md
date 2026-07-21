# 课题28 — div 功能开发文档

> **版本**: v1.0 | **日期**: 2026-07-21 | **状态**: 待实现
> **关联**: [设计文档](28-扩展指令选择-div-设计文档.md) | [课题28](28-扩展指令选择.md)

---

## 1. 修改清单总览

| 序号 | 文件 | 改动类型 | 说明 |
|------|------|---------|------|
| 1 | `scratchv/ir/types.py` | 新增 | `OpCode` 增加 `IDIV`, `REM`, `MOD`；`is_arith()` 扩展 |
| 2 | `scratchv/ir/builder.py` | 新增 | `idiv()`, `rem()`, `mod()` 方法 |
| 3 | `scratchv/backend/inst_select_ext.py` | 修改 | `_select_div` 增加 f32→FDIV_S 分支 |
| 4 | `scratchv/optimizer/constant_folding.py` | 修改 | `_try_fold` / `_compute` 增加 `IDIV`, `REM`, `MOD` |
| 5 | `scratchv/frontend/dsl_parser.py` | 新增 | `rem` handler |
| 6 | `tests/test_inst_select_ext.py` | 新增 | div/rem 测试用例 |

---

## 2. 详细修改步骤

### 2.1 `scratchv/ir/types.py` — OpCode 扩展

**位置**: [types.py:14-50](scratchv/ir/types.py#L14-L50)

**修改内容**:

```python
class OpCode(enum.Enum):
    """All supported IR operation codes."""

    # Arithmetic (binary)
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    # 新增 ↓
    IDIV = "idiv"    # 整数除法 (truncated toward zero)
    REM = "rem"      # 整数取余
    MOD = "mod"      # 取余别名
    # 新增 ↑
    # Arithmetic (unary)
    NEG = "neg"
    EXP = "exp"
    # ... 其余保持不变 ...
```

同时修改 `is_arith()` 方法 [types.py:52-53](scratchv/ir/types.py#L52-L53):

```python
def is_arith(self) -> bool:
    return self in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV,
                    OpCode.IDIV, OpCode.REM, OpCode.MOD)
```

---

### 2.2 `scratchv/ir/builder.py` — IRBuilder 新增方法

**位置**: [builder.py:94-97](scratchv/ir/builder.py#L94-L97) 之后

**新增代码**:

```python
def idiv(self, lhs: Value, rhs: Value) -> Value:
    """整数除法 (truncated toward zero)."""
    dest = self.make_value(dtype=DataType.INT32)
    self._emit(OpCode.IDIV, dest, [lhs, rhs])
    return dest

def rem(self, lhs: Value, rhs: Value) -> Value:
    """整数取余 (结果符号与被除数相同)."""
    dest = self.make_value(dtype=DataType.INT32)
    self._emit(OpCode.REM, dest, [lhs, rhs])
    return dest

def mod(self, lhs: Value, rhs: Value) -> Value:
    """取余别名，同 rem."""
    return self.rem(lhs, rhs)
```

---

### 2.3 `scratchv/backend/inst_select_ext.py` — 修复 f32 div 路径

**问题**: 当前 `_select_div` [inst_select_ext.py:312-318](scratchv/backend/inst_select_ext.py#L312-L318) 对 f32 走基类 `DIV`（整数除），未使用 `FDIV_S`。

**修改内容**:

```python
def _select_div(self, instr: Instruction) -> None:
    if self._is_fp64(instr):
        self._select_fdiv_d(instr)
    elif instr.dest and instr.dest.dtype == DataType.INT32:
        self._select_idiv(instr)
    elif instr.dest and instr.dest.dtype == DataType.FLOAT32:
        # f32 浮点除法 → FDIV_S
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FDIV_S, dst, a, b, comment="fdiv.s")
    else:
        super()._select_div(instr)
```

---

### 2.4 `scratchv/optimizer/constant_folding.py` — 常量折叠扩展

**位置**: [_try_fold at line 41-46](scratchv/optimizer/constant_folding.py#L41-L46) 和 [_compute at line 73-81](scratchv/optimizer/constant_folding.py#L73-L81)

**_try_fold 修改**:

```python
def _try_fold(self, instr: Instruction) -> Instruction | None:
    """Try to fold an instruction. Returns a replacement or None."""
    if instr.opcode not in (
            OpCode.ADD, OpCode.SUB,
            OpCode.MUL, OpCode.DIV,
            OpCode.IDIV, OpCode.REM, OpCode.MOD):  # 新增
        return None
    # ... 其余不变 ...
```

**_compute 修改**:

```python
@staticmethod
def _compute(opcode: OpCode, a: float, b: float) -> float | None:
    mapping = {
        OpCode.ADD: a + b,
        OpCode.SUB: a - b,
        OpCode.MUL: a * b,
        OpCode.DIV: a / b if b != 0 else None,
        # 新增 ↓
        OpCode.IDIV: int(a) // int(b) if b != 0 else None,
        OpCode.REM: int(a) % int(b) if b != 0 else None,
        OpCode.MOD: int(a) % int(b) if b != 0 else None,
        # 新增 ↑
    }
    return mapping.get(opcode)
```

---

### 2.5 `scratchv/frontend/dsl_parser.py` — DSL 支持 rem

**位置**: [_dispatch_op at line 133-168](scratchv/frontend/dsl_parser.py#L133-L168)

**在 `handlers` dict 中新增**:

```python
handlers = {
    "add": lambda: self.builder.add(resolved[0], resolved[1]),
    "sub": lambda: self.builder.sub(resolved[0], resolved[1]),
    "mul": lambda: self.builder.mul(resolved[0], resolved[1]),
    "div": lambda: self.builder.div(resolved[0], resolved[1]),
    "rem": lambda: self.builder.rem(resolved[0], resolved[1]),   # 新增
    "mod": lambda: self.builder.mod(resolved[0], resolved[1]),   # 新增
    # ... 其余不变 ...
}
```

---

### 2.6 `tests/test_inst_select_ext.py` — 测试用例

**新增测试类**:

```python
class TestDivRem:
    """Tests for div, rem, mod instruction selection."""

    def test_div_int32(self):
        """整数除法 → MachineOp.DIV"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.div(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.DIV in ops

    def test_div_float32(self):
        """f32 除法 → MachineOp.FDIV_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        b = builder.make_value(name="b", dtype=DataType.FLOAT32)
        c = builder.div(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FDIV_S in ops

    def test_div_float64(self):
        """f64 除法 → MachineOp.FDIV_D"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        b = builder.make_value(name="b", dtype=DataType.FLOAT64)
        c = builder.div(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FDIV_D in ops

    def test_rem_int32(self):
        """整数取余 → MachineOp.REM"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.rem(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.REM in ops

    def test_div_by_constant(self):
        """常量除法 → 常量折叠"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_const(10, dtype=DataType.INT32)
        b = builder.make_const(2, dtype=DataType.INT32)
        c = builder.div(a, b)
        builder.ret(c)

        # 常量折叠后应为 LOAD_CONST
        from scratchv.optimizer.constant_folding import ConstantFolder
        folder = ConstantFolder(builder.program)
        num_folded = folder.run()
        assert num_folded == 1

    def test_rem_by_constant(self):
        """常量取余 → 常量折叠"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_const(10, dtype=DataType.INT32)
        b = builder.make_const(3, dtype=DataType.INT32)
        c = builder.rem(a, b)
        builder.ret(c)

        from scratchv.optimizer.constant_folding import ConstantFolder
        folder = ConstantFolder(builder.program)
        num_folded = folder.run()
        assert num_folded == 1

    def test_div_by_zero_constant(self):
        """除零常量 → 不折叠（安全跳过）"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_const(10, dtype=DataType.INT32)
        b = builder.make_const(0, dtype=DataType.INT32)
        c = builder.div(a, b)
        builder.ret(c)

        from scratchv.optimizer.constant_folding import ConstantFolder
        folder = ConstantFolder(builder.program)
        num_folded = folder.run()
        assert num_folded == 0  # 除零不折叠

    def test_supported_ops_has_div_rem(self):
        """supported_ops 包含 div/rem/mod"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        ops = selector.supported_ops
        assert "div" in ops
        assert "idiv" in ops
        assert "rem" in ops
        assert "mod" in ops
```

---

## 3. 实施顺序

```
Step 1: types.py     — 新增 OpCode.IDIV, REM, MOD + is_arith()
Step 2: builder.py   — 新增 idiv(), rem(), mod() 方法
Step 3: inst_select_ext.py — 修复 _select_div f32→FDIV_S 路径
Step 4: constant_folding.py — 扩展 _try_fold 和 _compute
Step 5: dsl_parser.py — 新增 "rem", "mod" handler
Step 6: tests/       — 新增 TestDivRem 测试类
Step 7: 运行验证      — make test
```

每个 step 独立可验证，建议按顺序逐个完成并测试。

---

## 4. 验证方法

### 4.1 单元测试

```bash
make test
# 或
pytest tests/test_inst_select_ext.py::TestDivRem -v
```

### 4.2 端到端 DSL 测试

创建测试文件 `tests/dsl/test_div_rem.dsl`:

```dsl
# 整数除法测试
a = 10
b = 3
q = div(a, b)
r = rem(a, b)
return r
```

### 4.3 手动验证

```python
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector

# 测试 f32 除法
builder = IRBuilder()
builder.new_function("test_f32_div")
builder.new_block("entry")
a = builder.make_value(name="a", dtype=DataType.FLOAT32)
b = builder.make_value(name="b", dtype=DataType.FLOAT32)
c = builder.div(a, b)
builder.ret(c)

selector = ExtendedInstructionSelector(builder.program)
instrs = selector.run()
for i in instrs:
    print(f"  {i.op.value:12s} {i.dst} {i.src1} {i.src2}  # {i.comment}")
# 预期输出: fdiv.s  ...
```

---

## 5. 注意事项

| 注意点 | 说明 |
|--------|------|
| **OpCode 新增后 dispatch 自动生效** | 基类 `_select_instruction` 用 `getattr(self, f"_select_{opcode.value}")` 动态 dispatch，新增 OpCode 后只要 `_select_*` 方法存在即可自动路由 |
| **rem/mod 同方法** | `_select_mod` 直接委托给 `_select_rem`，两个 OpCode 共用同一 MachineOp |
| **f32 div 的 dtype 来源** | `builder.make_value()` 默认 dtype 是 `FLOAT32`，所以 `builder.div()` 产生的 dest 默认是 FLOAT32 |
| **基类 _select_div 不再对 f32 生效** | 扩展类的 `_select_div` 覆盖了基类，f32 走 `FDIV_S` 分支，不再回退到基类的整数 `DIV` |
| **常量折叠的 IDIV 语义** | 使用 Python `int(a) // int(b)`，向零取整，与 RISC-V `div` 语义一致 |
| **向后兼容** | 现有 `div` 的 OpCode 值不变，`builder.div()` 接口不变，仅扩展类中行为修正 |

---

## 6. 后续任务（留白）

本文档仅覆盖 div 相关功能。以下功能留待后续文档：

- **sqrt**: 硬件 `fsqrt.s` / 库调用 `sqrtf`
- **min/max**: 分支无整数 min/max，浮点 `fmin.s`/`fmax.s`
- **abs**: 分支无整数 abs，浮点 `fabs.s`
- **float64 全路径**: 加载/存储/算术/比较/转换
- **除零检测**: 可选 BEQZ 守卫
- **浮点寄存器分配**: 使用 F 寄存器 (f0-f31)