# 课题0：新增 ONNX 单算子支持

> **难度**：中 | **类型**：项目实战 | **源文件**：`scratchv/standalone/onnx_to_riscv_standalone.py`
> **状态**：⬜ 规划中

---

## 概述

ScratchV 的 Standalone 编译器将 ONNX 模型直接编译为 RISC-V 机器码。当前已支持 6 种算子（Conv、Gemm、MaxPool、ReLU、Sigmoid、Reshape）。本课题的目标是**新增一个 ONNX 算子的完整编译支持**——从 shape 推导、RISC-V 代码生成到单算子模型验证，走通全流程。

建议选择 `Softmax`、`Add`、`MatMul` 或 `GELU` 作为第一个新增算子（难度递进）。

---

## 理解背景

### 是什么？

ONNX 单算子支持的意思是：给定一个只包含**一种算子**的 `.onnx` 文件（比如只有一个 `Softmax` 节点），ScratchV 能把它编译成可执行的 RISC-V 机器码，并在 TinyFive 仿真器上跑出正确结果。

```
relu.onnx (只有 1 个 Relu 节点)
    │
    ▼
onnx_to_riscv_standalone.py   ← ScratchV Standalone 编译器
    │
    ▼
relu.bin (RISC-V 机器码)
    │
    ▼
tinyfive 仿真执行 → 验证结果正确
```

### 为什么？

- 理解编译器**后端**如何将一个抽象算子翻译为具体指令序列
- 走通 ScratchV 完整编译管线：ONNX 解析 → 权重定点化 → 内存规划 → 代码生成 → 仿真验证
- 当前 CNN 模型只有 6 种算子，新增算子能让 ScratchV 支持更复杂的模型
- 是贡献 ScratchV 项目最直接的入门方式

### 核心概念

#### 1. Standalone 编译管线

```
ONNX 文件 (protobuf 二进制)
    │  [1] ProtoReader: 手工 protobuf wire-format 解析
    ▼
ONNXModel (node 列表 + initializer 字典 + shape 信息)
    │  [2] 权重 Q16.16 定点化: float32 × 65536 → int32
    ▼
MemoryPlan (为每个 tensor 分配地址空间)
    │  [3] 代码段 + 数据段 + 工作区 三段布局
    ▼
CodeGen (_gen_conv, _gen_relu ...) 逐个节点生成 RISC-V 指令
    │  [4] 每条指令手动编码 32-bit RV32IM
    ▼
Flat Binary (.bin) → TinyFive 仿真 / Spike 仿真
```

#### 2. 新增一个算子需要改哪几个地方

新增算子 `Foo` 需要修改 `onnx_to_riscv_standalone.py` 中的 **3 个位置**：

| 位置 | 方法 | 作用 |
|------|------|------|
| 1. Shape 推导 | `_infer_foo_shape(node, shapes)` | 根据输入 shape 算出输出 shape |
| 2. 算子派发 | `_infer_output_shapes()` 中的 `if/elif` 链 | 把 `"Foo"` 路由到 shape 推导 |
| 3. 代码生成 | `_gen_foo(node)` | 生成 RISC-V 汇编指令序列 |

对于逐元素算子（如 ReLU），代码生成只需要：
1. 加载输入地址
2. 循环遍历每个元素
3. 做计算
4. 写回输出地址

#### 3. RISC-V 代码生成基础

Standalone 编译器使用 RV32IM 指令集（32 位整数 + 乘法扩展），手动编码每条指令：

```python
# 常用指令编码宏（在 onnx_to_riscv_standalone.py 中定义）
rv_add(rd, rs1, rs2)        # rd = rs1 + rs2
rv_sub(rd, rs1, rs2)        # rd = rs1 - rs2
rv_mul(rd, rs1, rs2)        # rd = rs1 * rs2  (低 32 位)
rv_lw(rd, rs1, offset)      # rd = mem[rs1 + offset]
rv_sw(rs2, rs1, offset)     # mem[rs1 + offset] = rs2
rv_addi(rd, rs1, imm)       # rd = rs1 + imm
rv_bne(rs1, rs2, offset)    # if rs1 != rs2: pc += offset
rv_slli(rd, rs1, shamt)     # rd = rs1 << shamt
rv_srai(rd, rs1, shamt)     # rd = rs1 >> shamt (算术右移)
rv_nop()                     # 空操作 (addi x0, x0, 0)
```

> 💡 **寄存器约定**：`a0`=输入指针，`a1`=输出指针，`gp`=数据段基址，`sp`=栈指针。临时寄存器用 `t0-t6`。

#### 4. Q16.16 定点算术

所有权重和中间结果都用 **Q16.16 定点格式**（1 个定点单位 = 1/65536 的浮点值）：

```python
# float → Q16.16
qval = int(fval * 65536)

# Q16.16 乘法（需要 64 位中间值防止溢出）
# 在 RISC-V 上: MULH 取高 32 位, SRAI 16 做截断
rv_mul(t0, a, b)        # t0 = (a*b) 低 32 位
rv_mulh(t1, a, b)       # t1 = (a*b) 高 32 位 (符号扩展)
rv_slli(t0, t0, 16)     # 低 32 位左移
rv_srli(t0, t0, 16)     # 逻辑右移 (取低 16 位)
rv_slli(t1, t1, 16)     # 高 32 位左移
rv_or(t0, t0, t1)       # 拼接 → 64 位乘积
rv_srai(t0, t0, 16)     # 算术右移 16 → Q16.16 截断结果
```

---

## 详细任务

1. **选择算子**：从 ONNX 标准算子中选择一个（建议 `Softmax` 或 `Add`）。
2. **理解 ONNX 算子规范**：阅读 ONNX 官方文档，确认输入/输出数量和类型、有无属性。
3. **创建单算子 ONNX 模型**：用 Python（`onnx` + `numpy`）生成只包含目标算子的 `.onnx` 文件，放入 `models/single_op/<op>/`。
4. **实现 shape 推导**：在 `_infer_output_shapes()` 中添加 `if op == "Softmax":` 分支，编写 `_infer_softmax_shape()` 方法。
5. **实现代码生成**：编写 `_gen_softmax(node)` 方法，按算子语义生成 RISC-V 指令。
6. **运行编译**：用 `onnx_to_riscv_standalone.py` 编译单算子模型，检查生成的汇编是否正确。
7. **仿真验证**：用 TinyFive 仿真执行编译出的 `.bin` 文件，对比 golden 输出（可用 numpy 算）。
8. **编写测试**：添加至少 1 个单算子测试用例。
9. **文档**：记录新增的算子、使用示例、遇到的坑。

### 新增算子的 checklist

```
□ 1. 创建 models/single_op/<op>/<op>_test.onnx
□ 2. 在 _infer_output_shapes() 中添加 shape 推导分支
□ 3. 编写 _infer_<op>_shape() 方法
□ 4. 编写 _gen_<op>() 代码生成方法
□ 5. 编译通过：python scratchv/standalone/onnx_to_riscv_standalone.py models/single_op/<op>/xxx.onnx -o /tmp/test.bin
□ 6. 汇编检查：cat /tmp/test.s 人工 review
□ 7. tinyfive 仿真：结果与 numpy golden 一致
□ 8. 提交 PR
```

---

## 交付产物

- `models/single_op/<op>/` — 至少 1 个单算子 ONNX 模型
- `scratchv/standalone/onnx_to_riscv_standalone.py` — 新增的 `_gen_<op>` 和 `_infer_<op>_shape` 方法
- 至少 1 个测试用例
- 文档：算子使用说明、验证方法

---

## 代码走读

### 以 ReLU 为例：最简单的算子

ReLU 是逐元素算子，只有一个输入和一个输出，没有属性。是理解代码生成流程的最佳起点。

**Step 1: Shape 推导**

```python
# 在 _infer_output_shapes() 中（约 516 行）
elif op == "Relu":
    if input_shapes and input_shapes[0]:
        self._set_output_shape(node, input_shapes[0])
```

ReLU 不改变 shape，输入多大输出就多大——直接透传。

**Step 2: 代码生成**（约 2051 行）

```python
def _gen_relu(self, node: NodeInfo) -> None:
    x_name = node.inputs[0]
    out_name = node.outputs[0]

    shape = self.model.get_shape(x_name)
    total_elements = 1
    for d in shape:
        total_elements *= d

    # 1. 加载输入/输出地址
    self._get_workspace_addr(x_name, self.T0)     # T0 = input ptr
    self._get_workspace_addr(out_name, self.T1)   # T1 = output ptr

    # 2. 循环计数器
    self.emit.emit_li32(self.T2, 0, "i = 0")      # T2 = loop index

    # 3. 循环体
    loop_label = self.emit.new_label("relu_loop")
    end_label = self.emit.new_label("relu_end")
    self.emit.label(loop_label)

    # for i in range(total_elements):
    self.emit.emit(rv_lw(self.T3, self.T0, 0),    # T3 = *input
                   f"load x[{i}]")
    # ReLU: if x > 0: x else 0
    self.emit.emit(rv_slt(self.T4, zero, self.T3),# T4 = (0 < x) ? 1 : 0
                   "x > 0?")
    # 条件选择：用 mask 实现（无分支版本）
    self.emit.emit(rv_sub(self.T4, zero, self.T4),# T4 = 0 或 -1 (mask)
                   "mask")
    self.emit.emit(rv_and(self.T3, self.T3, self.T4), # x & mask
                   "x if x>0 else 0")

    self.emit.emit(rv_sw(self.T3, self.T1, 0),    # *output = relu(x)
                   f"store y[{i}]")

    # 指针前进
    self.emit.emit(rv_addi(self.T0, self.T0, 4), "input++")
    self.emit.emit(rv_addi(self.T1, self.T1, 4), "output++")
    self.emit.emit(rv_addi(self.T2, self.T2, 1), "i++")

    # 循环条件判断
    self.emit.emit_li32(self.T4, total_elements)
    self.emit.emit(rv_bne(self.T2, self.T4, loop_label), "i < N?")
    self.emit.label(end_label)
```

**关键思路**：逐元素算子 = 一个循环，每次加载一个元素、做计算、存回去。大多数激活函数（ReLU、Sigmoid、Tanh）和算术算子（Add、Sub、Mul）都遵循这个模式。

### 以 Conv2D 为例：复杂的多级嵌套循环算子

```python
# 算子派发（约 1558 行）
handler = getattr(self, f"_gen_{op}", None)
if handler is None:
    raise ValueError(f"Unsupported op: {node.op_type}")
handler(node)
```

派发机制用 Python 的反射：`getattr(self, f"_gen_{node.op_type.lower()}")`。所以新增算子只需写一个 `_gen_<小写算子名>` 方法即可，**不需要**修改派发代码。

Conv2D 的代码规模（~400 行）远超 ReLU（~50 行），因为涉及：
- 6 层嵌套循环 (OC, OH, OW, IC, KH, KW)
- 边界检查（padding 时跳过越界元素）
- 指针步进优化（内层循环中只做 `lw + madd + addi`）
- 累加器与 bias 初始化
- Q16.16 乘法截断

### MemoryPlan 地址布局

```
0x00000000 ┌──────────────────────┐
           │  Code (.text)        │  ← 代码段：RISC-V 指令
           ├──────────────────────┤
           │  Data (.data)        │  ← 权重/偏置 (Q16.16)
           ├──────────────────────┤
  sp →     │  Workspace (栈上)    │  ← 中间 tensor 工作区
           └──────────────────────┘
```

---

## 动手练习

### 练习 1: 仿照 ReLU 添加 Tanh 算子

1. 用 Python 生成一个 `tanh_test.onnx`（1 个 Tanh 节点，输入 shape 任意）
2. 实现 `_gen_tanh`：用查表法或分段近似（参考已有的 Sigmoid 实现）
3. 用 TinyFive 仿真执行，验证结果

### 练习 2: 添加二元算子 Add

1. 生成 `add_test.onnx`（两个输入 tensor，逐元素相加）
2. 实现 `_gen_add`：注意 Add 有**两个输入**
3. 验证 shape 推导：两个输入 shape 不同时，ONNX 的 broadcasting 规则如何处理？

### 练习 3: 对比 Standalone 和库路径

1. 先用库路径（`onnx_parser.py`+后端）编译同一个模型
2. 对比生成的 RISC-V 汇编和 Standalone 路径的差异
3. 分析哪个版本的代码更高效？为什么？

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **shape 推导遗漏** | 忘记在 `_infer_output_shapes()` 中添加 `elif` 分支，导致后续代码生成时 tensor shape 未知 |
| **Q16.16 溢出** | `mul` 只取 32 位乘积的低 32 位；正确做法是用 `mulh`+移位组合取 64 位乘积并截断 |
| **地址加载错误** | 权重用 `_get_weight_addr()`（数据段），中间结果用 `_get_workspace_addr()`（栈区），不能搞混 |
| **循环边界** | 循环次数 = 元素总数，注意 NCHW 四维 tensor 的 flatten 顺序 |
| **operator name 大小写** | ONNX 算子名首字母大写（`"Relu"`, `"Conv"`），但派发时 `.lower()` 了——方法名必须全小写 |
| **单算子模型生成** | 用 `onnx.helper.make_node()` 而非导出真实模型，避免引入多余节点 |

---

## 进阶阅读

- [ONNX 算子规范](https://github.com/onnx/onnx/blob/main/docs/Operators.md) — 所有标准算子的输入/输出/属性定义
- [RISC-V RV32I 指令集参考](https://riscv.org/technical/specifications/) — 基础整数指令集
- [Q 格式定点数教程](https://en.wikipedia.org/wiki/Q_(number_format))
- 相关 topic: [课题2 — ONNX 解析器](02-ONNX解析器.md) | [课题8 — 指令选择](08-指令选择.md) | [课题19 — Standalone RISC-V 编译器](19-Standalone-RISC-V编译器.md) | [课题24 — Spike 仿真集成](24-Spike仿真.md)

---

## 12周每周目标

- **W1**：搭建环境，跑通 CNN 编译 `python scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx -o /tmp/cnn.bin --estimate`。理解 ONNX 模型结构。
- **W2**：学习 RISC-V RV32IM 指令集，理解每条指令的编码方式和语义。用 TinyFive 单步执行 `relu.bin`。
- **W3**：精读 `_gen_relu()` 方法（~50 行），逐行理解每条指令的作用。画出 ReLU 代码生成的数据流图。
- **W4**：选择要新增的算子（建议 Softmax 或 Add），阅读 ONNX 官方文档中的算子定义。用 Python 生成单算子 `.onnx` 文件。
- **W5**：编写 `_infer_<op>_shape()` 方法。手动构造几个测试用例，验证 shape 推导的正确性。
- **W6**：编写 `_gen_<op>()` 代码生成。先用**伪代码**写出算子算法，再逐句翻译为 RISC-V 指令序列。
- **W7**：实现循环框架（加载地址、循环计数器、条件跳转）。先写一个能跑的空循环（不做实际计算）。
- **W8**：实现核心计算逻辑（Q16.16 定点运算）。处理特殊操作（如 Softmax 的 exp + sum + div 三步）。
- **W9**：编译测试：用 `onnx_to_riscv_standalone.py --asm /tmp/test.s` 输出汇编，人工 review 逻辑是否正确。
- **W10**：仿真验证：用 TinyFive 执行编译出的 `.bin`，对比 numpy 计算的 golden 输出。修复 bug。
- **W11**：处理边界情况：不同 shape、不同输入值范围。确保所有 case 通过。
- **W12**：编写文档（使用说明、算法说明、验证方法），提交 PR。

---

> 🔗 相关文档：[00-环境搭建指南](../00-环境搭建指南.md) | [02-快速上手教程](../02-快速上手教程.md) | [04-故障排除FAQ](../04-故障排除FAQ.md)
