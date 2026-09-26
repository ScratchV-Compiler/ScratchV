# 课题2：ONNX 模型解析器

> **难度**：中 | **类型**：项目实战 | **源文件**：`scratchv/frontend/onnx_parser.py`
> **状态**：✅ 已完成

---

## 概述

ONNX 解析器将 `.onnx` 模型文件（protobuf 二进制格式）解析为 ScratchV 的内部 IR，支持 15 种 ONNX 算子（Conv、Gemm、MatMul、MaxPool、ReLU、Sigmoid、Softmax 等）。实现它需要理解 ONNX 模型结构、protobuf wire format、以及属性提取的多态处理。

---

## 理解背景

### 是什么？

ONNX 解析器将 `.onnx` 模型文件（protobuf 二进制格式）解析为 ScratchV 的内部 IR。

```
.onnx 文件 (protobuf 二进制)
        │
   ONNXParser (onnx.load)
        │
        ▼
   IR Program (Function → BasicBlock → Instruction)
```

### 为什么？

- ONNX 是 AI 模型的标准交换格式，PyTorch/TensorFlow/Keras 都能导出
- 解析器的存在让 ScratchV 能**编译真实的训练好的模型**，而不只是手写 DSL 程序
- 两条路径共享解析逻辑：库路径（用 `onnx` 包）和 Standalone 路径（手工 protobuf 解析）

### 核心概念

#### 1. ONNX 模型结构

```
ModelProto
  └── GraphProto
        ├── input[]      ← 模型输入（名称、形状、类型）
        ├── initializer[] ← 权重/偏置（常量 tensor）
        ├── output[]     ← 模型输出
        └── node[]       ← 计算节点列表
              ├── op_type      ← 算子类型: "Conv", "ReLU", ...
              ├── input[]      ← 输入名称: ["X", "W", "B"]
              ├── output[]     ← 输出名称: ["Y"]
              └── attribute[]  ← 属性: kernel_shape, strides, pads
```

#### 2. 15 种已支持的 ONNX 算子

| 类别 | 算子 | ONNX Op | IR OpCode |
|------|------|---------|-----------|
| 算术 | Add, Sub, Mul, Div | `Add`, `Sub`, `Mul`, `Div` | `ADD`, `SUB`, `MUL`, `DIV` |
| 激活 | ReLU, GELU, Sigmoid | `Relu`, `Gelu`, `Sigmoid` | `RELU`, `GELU`, `SIGMOID` |
| 归一化 | Softmax | `Softmax` | `SOFTMAX` |
| 池化 | MaxPool | `MaxPool` | `MAXPOOL` |
| 卷积 | Conv | `Conv` | `CONV` |
| 全连接 | Gemm, MatMul | `Gemm`, `MatMul` | `GEMM`, `MATMUL` |
| 形状 | Reshape | `Reshape` | `RESHAPE` |
| 一元 | Neg, Exp | `Neg`, `Exp` | `NEG`, `EXP` |

#### 3. 两条路径对比

| | 库路径 (`onnx_parser.py`) | Standalone 路径 |
|------|--------------------------|----------------|
| 依赖 | `onnx` 包 | 零依赖 (手工 protobuf) |
| 输出 | IR Program | ONNXModel → 直接机器码 |
| 精度 | 跟随 ONNX（float32 等） | Q16.16 定点 |
| 用途 | 主管线编译 | 零依赖独立编译 |

---

## 详细任务

1. 学习 ONNX 模型结构和 protobuf wire format 编码原理（varint, field, wire types）。
2. 实现 ProtoReader 基础解析（read_varint, read_field），能够读取 protobuf 二进制文件。
3. 实现 GraphProto 解析：提取 node 列表、initializer 列表、input/output 定义。
4. 实现 TensorProto 解析：raw_data 解码、shape 提取、dtype 识别。
5. 实现 Conv 算子的属性提取：kernel_shape, strides, pads, group 等多态属性处理。
6. 实现 Gemm/MatMul/MaxPool 算子的 node→IR 翻译。
7. 实现激活函数（ReLU, Sigmoid, Softmax, GELU）和算术算子的翻译。
8. 实现 Reshape/一元算子的翻译。
9. 构建完整的 ONNXModel.weights 字典（名称→numpy 数组）和 MemoryPlan。
10. 实现库路径版本（使用 `onnx` 包）作为 Standalone 版本的参考实现。
11. 测试所有 15 种算子的解析正确性（至少 3 个完整的 ONNX 模型）。
12. 对比库路径和 Standalone 路径的解析结果，验证一致性。

---

## 交付产物

- `scratchv/frontend/onnx_parser.py` — 库路径 ONNX 解析器
- Standalone 路径的 ONNXModel (在 `onnx_to_riscv_standalone.py` 中)
- 至少 3 个 ONNX 模型的解析测试
- 文档：支持的算子列表、属性类型说明、使用示例

---

## 代码走读

### ONNXParser 核心流程

```python
class ONNXParser:
    def parse(self, filepath: str) -> Program:
        model = onnx.load(filepath)
        graph = model.graph

        program = Program()
        func = Function(name="main")

        # 1. 注册所有权重/常量
        self._register_initializers(graph.initializer)

        # 2. 逐节点解析
        for node in graph.node:
            handler = self._HANDLERS.get(node.op_type)
            if handler is None:
                raise NotImplementedError(
                    f"Unsupported ONNX op: {node.op_type}"
                )
            instrs = handler(self, node)
            for instr in instrs:
                self.builder.add_instruction(instr)

        func.basic_blocks.append(self.builder.current_block)
        program.functions.append(func)
        return program
```

### 算子处理示例：Conv

```python
def _handle_conv(self, node):
    """Conv 节点 → CONV IR 指令"""
    # 提取属性
    kernel_shape = self._get_attr(node, "kernel_shape")
    strides = self._get_attr(node, "strides", [1, 1])
    pads = self._get_attr(node, "pads", [0, 0, 0, 0])

    # 创建 IR 指令
    instr = Instruction(
        dest=self._make_value(node.output[0]),
        op=OpCode.CONV,
        operands=[
            self._get_value(node.input[0]),   # 输入
            self._get_value(node.input[1]),   # 权重
            self._get_value(node.input[2]),   # 偏置
        ],
        attrs={
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
        },
    )
    return [instr]
```

### 关键细节：属性提取

ONNX 的属性有类型——可能是 `int`、`ints`（整数列表）、`float` 等：
```python
def _get_attr(self, node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
    return default
```

---

## 动手练习

### 练习 1: 添加新算子的支持

在 `onnx_parser.py` 中添加 `Tanh` 算子的处理。ONNX 定义 `Tanh` 输入一个 tensor，输出一个 tensor（逐元素 tanh）。

### 练习 2: 可视化 ONNX 模型图

用 [Netron](https://netron.app) 打开 `models/graph/cnn.onnx`，对照解析器的输出，理解模型的拓扑结构。

### 练习 3: 对比解析前后的权重

用 `onnxruntime` 直接推理模型，和 ScratchV 编译后的结果对比，验证解析的正确性。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **ONNX opset 版本** | 不同 opset 版本的算子属性可能不同，注意兼容性 |
| **属性类型多态** | ONNX 属性可能是 int/ints/float/floats/string，需要分别处理 |
| **initializer vs input** | 权重在 `initializer` 中（常量），输入在 `input` 中（变量），不要混淆 |
| **不支持的操作** | 遇到不支持的算子会抛 `NotImplementedError`，需要手动添加 handler |

---

## 进阶阅读

- [ONNX 算子规范](https://github.com/onnx/onnx/blob/main/docs/Operators.md) — 所有标准算子的定义
- [ARCHITECTURE.md](../../../reference/ARCHITECTURE.md) — Standalone 路径的 ONNXModel 手工 protobuf 解析原理
- 相关 topic: [课题1 — DSL 前端增强器](../../../topics/01-DSL前端增强器/01-DSL前端增强器.md) | [课题8 — 指令选择](../08-指令选择/08-指令选择.md) | [课题19 — Standalone RISC-V 编译器](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md)

---

## 12周每周目标

- **W1**：学习 ONNX 模型结构（ModelProto → GraphProto → NodeProto）。用 Netron 可视化 `models/graph/cnn.onnx`，理解 15 种算子的拓扑关系。
- **W2**：学习 protobuf wire format 编码原理（varint 编码、field tag 结构、wire type）。阅读 Google 的 [Protobuf Encoding 文档](https://protobuf.dev/programming-guides/encoding/)。
- **W3**：实现 ProtoReader 基础方法：`read_varint()` 和 `read_field()`。能够解析简单的 protobuf 消息（仅含 varint 和 length-delimited 字段）。
- **W4**：实现 GraphProto 完整解析：提取 node 列表（op_type, input, output, attribute）、initializer 列表（name, raw_data, dims, data_type）、input/output 定义。
- **W5**：实现 TensorProto 解析：raw_data 到 numpy 数组的解码、shape/dtype 的提取。重点处理 FLOAT 和 INT64 两种数据类型。
- **W6**：实现 Conv 算子的属性提取。处理 kernel_shape, strides, pads, group, dilations 等多态属性（可能是 int 或 ints）。
- **W7**：实现 Gemm, MatMul, MaxPool 三个算子的 node→IR 翻译。Gemm 需处理 transA/transB 属性，MaxPool 需处理 kernel_shape/strides/pads。
- **W8**：实现激活函数算子（ReLU, Sigmoid, Softmax, GELU）和算术算子（Add, Sub, Mul, Div）的翻译。
- **W9**：实现 Reshape, Neg, Exp 等一元算子的翻译。构建完整的 ONNXModel.weights 字典。
- **W10**：实现库路径版本（使用 `onnx` 包加载），作为 Standalone 版本的黄金参考。实现 `_register_initializers` 方法。
- **W11**：编写完整测试：至少 3 个 ONNX 模型（cnn.onnx + 2 个手写小模型），验证 15 种算子全部解析正确。对比库路径和 Standalone 路径的输出一致性。
- **W12**：撰写文档（支持的算子列表、属性类型说明、使用示例、添加新算子的指南），准备演示。
