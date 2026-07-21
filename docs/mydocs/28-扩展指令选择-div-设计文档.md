# 课题28 — div 功能设计文档

> **版本**: v1.0 | **日期**: 2026-07-21 | **状态**: 设计中
> **关联**: [课题28 — 扩展指令选择](28-扩展指令选择.md)

---

## 1. 概述

### 1.1 目标

在 `ExtendedInstructionSelector` 中完善 **div 及相关运算** 的指令选择，支持所有数据类型的除法、取余运算，做到类型驱动的精确指令选择。

### 1.2 范围

| 操作 | 说明 | 本阶段 |
|------|------|--------|
| **div (浮点除)** | f32/f64 除法 | ✅ 实现 |
| **idiv (整数除)** | i32 有符号除法 | ✅ 实现 |
| **rem (整数取余)** | i32 有符号取余 | ✅ 实现 |
| **mod (取余别名)** | 同 rem | ✅ 实现 |
| sqrt | 平方根 | 留白 |
| min/max | 分支无 min/max | 留白 |
| abs | 分支无 abs | 留白 |
| float64 全路径 | f64 加载/存储/算术 | 留白 |

### 1.3 当前状态

经过代码分析，当前 div 指令选择存在以下问题：

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| **f32 除法无专用指令** | 高 | `_select_div` 对 f32 走基类 `DIV`（整数除），未使用 `FDIV_S` |
| **OpCode 缺失 rem/mod/idiv** | 高 | `OpCode` 枚举中没有 `REM`、`MOD`、`IDIV`，基类 dispatch 无法路由到扩展方法 |
| **MachineOp.FDIV_S 定义但未使用** | 中 | `machine_types.py` 中 `FDIV_S` 已定义，但 selector 中无消费代码 |
| **除零处理缺失** | 中 | 无除零检测或陷阱指令 |
| **前端不支持 rem/mod** | 中 | DSL/ONNX parser 未处理 `rem`/`mod`/`idiv` |

---

## 2. 架构设计

### 2.1 整体数据流

```
 DSL/ONNX 前端                IR (OpCode)              指令选择                     MachineOp
 ──────────────              ────────────             ────────────                ──────────
                                                                                  
 a / b  ──────►  OpCode.DIV ("div")  ──►  _select_div()  ──┬─ f64  ──► FDIV_D     
                                                            ├─ f32  ──► FDIV_S     
                                                            ├─ i32  ──► DIV        
                                                            └─ i64  ──► DIV        
                                                                                  
 a // b ──────►  OpCode.IDIV ("idiv") ──► _select_idiv() ──► DIV                   
                                                                                  
 a % b  ──────►  OpCode.REM ("rem")   ──► _select_rem()   ──► REM                 
                                                                                  
 a % b  ──────►  OpCode.MOD ("mod")   ──► _select_mod()   ──► REM  (别名)         
```

### 2.2 类型驱动的指令选择矩阵

```
                  OpCode.DIV                OpCode.IDIV          OpCode.REM/MOD
                  ─────────                ───────────          ──────────────
  dtype=f32  →    FDIV_S                   DIV                   REM
  dtype=f64  →    FDIV_D                   DIV                   REM
  dtype=i32  →    DIV                      DIV                   REM
  dtype=i64  →    DIV                      DIV                   REM
```

### 2.3 OpCode 扩展

在 `scratchv/ir/types.py` 的 `OpCode` 枚举中新增：

```python
class OpCode(enum.Enum):
    # ... 现有 opcode 保持不变 ...
    DIV = "div"        # 已有
    IDIV = "idiv"      # 新增：整数除法
    REM = "rem"        # 新增：整数取余
    MOD = "mod"        # 新增：取余（别名，DSL 中 % 运算符）
```

### 2.4 MachineOp 使用

| MachineOp | RISC-V 指令 | 用途 | 状态 |
|-----------|-----------|------|------|
| `DIV` | `div rd, rs1, rs2` | 有符号整数除法 (M 扩展) | 已有 |
| `REM` | `rem rd, rs1, rs2` | 有符号整数取余 (M 扩展) | 已有 |
| `FDIV_S` | `fdiv.s rd, rs1, rs2` | 单精度浮点除法 (F 扩展) | 已有，需启用 |
| `FDIV_D` | `fdiv.d rd, rs1, rs2` | 双精度浮点除法 (D 扩展) | 已有 |

### 2.5 类层次与 dispatch 设计

```
InstructionSelector (基类)
  ├── _select_instruction(instr)    # getattr(self, f"_select_{instr.opcode.value}")
  ├── _select_div(instr)            # 基类: 无条件 emit MachineOp.DIV
  └── ...
       │
       ▼ 继承
ExtendedInstructionSelector (扩展类)
  ├── _select_instruction(instr)    # 追踪 _current_dtype, 然后 super()
  ├── _select_div(instr)            # 重写: 按 dtype 三分支
  │     ├── f64 → _select_fdiv_d()
  │     ├── f32 → FDIV_S (新增)
  │     └── i32/i64 → DIV
  ├── _select_idiv(instr)           # 新增: i32 整数除 → DIV
  ├── _select_rem(instr)            # 新增: 整数取余 → REM
  ├── _select_mod(instr)            # 新增: 委托 _select_rem
  └── ...
```

### 2.6 除零策略

**本阶段选择: 不处理除零，依赖硬件行为。**

理由：
- RISC-V 规范规定 `div`/`rem` 除零时结果写入全 1，不触发异常
- 浮点 `fdiv` 除零产生 ±∞（符合 IEEE 754），不触发异常
- 后续版本可加入可选除零检测（`BEQZ` 检查除数 → 跳转 trap handler）

---

## 3. 接口设计

### 3.1 IR 层接口

```python
# IRBuilder 新增方法
class IRBuilder:
    def idiv(self, lhs: Value, rhs: Value) -> Value:
        """整数除法 (truncated toward zero)"""
        ...

    def rem(self, lhs: Value, rhs: Value) -> Value:
        """整数取余 (结果符号与被除数相同)"""
        ...

    def mod(self, lhs: Value, rhs: Value) -> Value:
        """取余别名，同 rem"""
        return self.rem(lhs, rhs)
```

### 3.2 前端接口

```python
# DSL 语法扩展
# 现有: a / b  → OpCode.DIV
# 新增: a % b  → OpCode.REM
# 新增: a // b → OpCode.IDIV (可选，远期)
```

### 3.3 指令选择器接口

```python
# ExtendedInstructionSelector 使用方式不变
selector = ExtendedInstructionSelector(program, enable_fp64=True)
machine_instrs = selector.run()
```

---

## 4. 测试策略

### 4.1 测试矩阵

| 测试用例 | 输入 | 预期 | 验证点 |
|---------|------|------|--------|
| float32 div | `10.0 / 4.0` | `2.5` | 生成 `fdiv.s` |
| int32 div | `10 / 4` | `2` | 生成 `div` |
| int32 rem | `10 % 4` | `2` | 生成 `rem` |
| neg int rem | `-10 % 4` | `-2` | 符号与被除数相同 |
| div by 1 | `x / 1` | `x` | 常量折叠优化 |
| div by 0 | `x / 0` | 不崩溃 | 硬件行为 |
| f32 div 链 | `a / b / c` | 正确 | 多指令序列 |
| mixed types | f32 / i32 | 类型错误 | 报错 |

### 4.2 测试文件

- `tests/test_inst_select_ext.py` — 扩展现有测试，新增 div 测试类
- 测试框架: `pytest`，使用 `make test` 运行

---

## 5. 风险与限制

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| M 扩展依赖 | 整数 div/rem 需要 RISC-V M 扩展 | 文档说明，在 `-march=rv32im` 下测试 |
| 寄存器分配不支持 F 寄存器 | fdiv.s 目标寄存器被当成整数寄存器 | 当前寄存器分配对所有寄存器一视同仁，功能正确但未利用 F 寄存器 |
| 常量折叠边界 | `div by 0` 常量折叠需跳过 | 已有 `b != 0` 守卫 |
| OpCode 枚举扩展 | 新增 opcode 可能影响下游 pass | 逐步添加，保持向后兼容 |

---

## 6. 后续扩展

本文档聚焦 div 功能。以下功能留待后续：

- **sqrt**: 硬件 `fsqrt.s` 或库调用 `sqrtf`
- **min/max**: 分支无整数 min/max，浮点 `fmin.s`/`fmax.s`
- **abs**: 分支无整数 abs，浮点 `fabs.s`
- **float64 全路径**: `fld`/`fsd`、`fadd.d` 等
- **除零检测**: 可选 BEQZ 守卫
- **浮点寄存器分配**: 使用 F 寄存器 (f0-f31) 而非整数寄存器