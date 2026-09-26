# Topic: ONNX 模型解析器 (Frontend ONNX Parser)

> **源文件**: `scratchv/frontend/onnx_parser.py`
> **依赖**: `onnx` Python 包
> **输入**: ONNX protobuf 文件 (`.onnx`)
> **输出**: IR `Program` 对象

---

## 概述

ONNX 解析器将 ONNX 模型文件转换为 ScratchV 内部 IR。支持 15 种 ONNX 算子，从 protobuf 中提取计算图、权重、tensor 形状。

## 支持的操作列表

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

## 解析流程

```
ONNX .onnx 文件
    │
    ▼
onnx.load()  ← protobuf 解析
    │
    ▼
提取 graph.input     → 张量输入 (名称, 形状, 数据类型)
提取 graph.initializer → 权重/偏置 (常量 tensor)
提取 graph.output    → 张量输出
    │
    ▼
逐节点遍历 graph.node:
  1. 读取 op_type (如 "Conv")
  2. 调用对应的 _handle_*() 方法
  3. 每条处理函数: 从 node.input 读取操作数, 从 node.output 取目标,
     从 node.attribute 取属性 (如 Conv 的 kernel_shape, strides, pads)
  4. 生成 IR Instruction, 插入当前 BasicBlock
    │
    ▼
IR Program (包含 Function, BasicBlock, Instruction)
```

## 关键数据结构

**`ONNXModel` (在 standalone 版本中)**:
```python
@dataclass
class ONNXModel:
    inputs: list[NodeInfo]      # 输入张量
    outputs: list[NodeInfo]     # 输出张量
    nodes: list[OpNode]         # 计算节点列表
    weights: dict[str, np.ndarray]  # 权重数据 (numpy)
    shapes: dict[str, tuple]    # 形状信息
```

**`OpNode`**:
```python
@dataclass
class OpNode:
    op_type: str                # "Conv", "Gemm", "ReLU", ...
    inputs: list[str]           # 输入名称列表
    outputs: list[str]          # 输出名称列表
    attrs: dict                 # 属性 (kernel_shape, strides, pads, ...)
```

## 与 standalone 版本的关系

`scratchv/frontend/onnx_parser.py` 使用 `onnx` 包解析，返回 IR Program，用于主编译器管线。

`scratchv/standalone/onnx_to_riscv_standalone.py` 中的 `ONNXModel` 独立实现了 protobuf wire-format 解析器（零外部依赖），直接产出 `ONNXModel` 对象，跳过 IR 直接生成 RISC-V 机器码。

两条路径使用相同的 `ONNXModel`/`OpNode` 概念，但实现不同：
- **库路径**: onnx 包 → IR Program → 后端代码生成
- **Standalone 路径**: 手工 protobuf 解析 → ONNXModel → 直接机器码生成

## 相关 Topic

- Topic 1: DSL 前端增强器 (`frontend/dsl_extended.py`)
- Topic 9: DSL 错误提示美化器 (`frontend/dsl_errors.py`)
- [ARCHITECTURE.md](../../../reference/ARCHITECTURE.md) — 完整编译管线
