# 课题19：Standalone RISC-V 编译器

> **难度**：中 | **类型**：参考分析 | **源文件**：`scratchv/standalone/onnx_to_riscv_standalone.py` | **行数**：~2800
> **状态**：✅ 已完成

---

## 概述

Standalone RISC-V 编译器是 ScratchV 的核心产物——一个完全自包含的 ONNX→RISC-V 编译器，零外部依赖（只用 Python 标准库）。理解它的五大组件（ProtoReader、ONNXModel、MemoryPlan、CNNRISCVGenerator、RISCVEmitter）是理解 ScratchV 架构全貌的关键。

---

## 理解背景

### 是什么？

Standalone RISC-V 编译器手工实现了完整的 ONNX→RISC-V 编译管线：

- **Protobuf 解析器**（不需 `protobuf` 包）
- **ONNX 模型加载**（不需 `onnx` 包）
- **内存规划 + Q16.16 定点转换**
- **CNN 算子 RISC-V 代码生成**（内联循环）
- **RV32IM 指令编码**（两遍扫描 + label fixup）

### 为什么？

为什么要"重新发明轮子"而不是直接用 `onnx` 包 + `llvmlite`？

1. **零依赖 = 零门槛**：任何装了 Python 3 的机器都能跑
2. **完全可控**：每一行代码都是自己写的，可以精确理解每个细节
3. **教学价值**：从 protobuf 解析到机器码编码，全流程透明
4. **RV32IM 定点优化**：LLVM 不支持 Q16.16 定点运算，必须自己实现

### 核心概念

#### 1. 五大组件

```
onnx_to_riscv_standalone.py
├── ProtoReader          ← 手工 protobuf wire-format 解析器
├── ONNXModel            ← 提取计算图、权重、形状
├── MemoryPlan           ← 权重/偏置地址分配 + Q16.16 转换
├── CNNRISCVGenerator    ← 逐算子 RISC-V 机器码生成
└── RISCVEmitter         ← RV32IM 指令编码 + label fixup
```

#### 2. Q16.16 定点运算

```
32-bit 整数的高 16 位 = 整数部分
32-bit 整数的低 16 位 = 小数部分

float → Q16.16: int(value * 65536) & 0xFFFFFFFF
Q16.16 乘法: MUL a, b → 64-bit → SRAI 16 (截断回 32-bit)
Q16.16 加法: ADD a, b (直接加，小数点已对齐)
```

#### 3. Conv2D 代码生成

6 层嵌套循环的 RISC-V 内联代码：
```
for oh in 0..OH:           ← 输出高度
  for ow in 0..OW:         ← 输出宽度
    for oc in 0..OC:       ← 输出通道
      for kh in 0..KH:     ← 卷积核高度
        for kw in 0..KW:   ← 卷积核宽度
          for ic in 0..IC: ← 输入通道
            load input[ic][ih][iw]
            load weight[oc][ic][kh][kw]
            mul + srai 16  (Q16.16)
            add (累加)
```

---

## 理解要点

1. 掌握五大组件的职责和数据流向（ProtoReader → ONNXModel → MemoryPlan → CNNRISCVGenerator → RISCVEmitter）
2. 理解 Q16.16 定点运算原理：float→定点转换、乘法的 64 位中间结果 + 右移截断
3. 理解 Conv2D 内联循环生成的 6 层嵌套结构和指针步进优化
4. 理解两遍扫描的 label fixup 机制（第一遍记录位置，第二遍回填偏移）
5. 能够独立运行编译器并解读输出报告（--estimate, --report, --tinyfive）

---

## 交付产物

- 五大组件的数据流图（手绘或工具绘制）
- CNN 模型编译输出（output.bin + output.s + 估算报告）
- Conv2D 最内层循环的指令序列分析（标注每条指令作用）

---

## 代码走读

### ProtoReader — 手工 Protobuf 解析

```python
class ProtoReader:
    def read_varint(self):
        """读取 varint 编码的无符号整数"""
        value = 0
        shift = 0
        while True:
            byte = self.data[self.pos]
            value |= (byte & 0x7F) << shift
            self.pos += 1
            if not (byte & 0x80):  # 最高位 = 0 表示结束
                break
            shift += 7
        return value

    def read_field(self):
        """读取一个 protobuf field: (field_number, wire_type, value)"""
        tag = self.read_varint()
        field_number = tag >> 3
        wire_type = tag & 0x07
        # ... 根据 wire_type 读取不同格式的数据 ...
```

### CNNRISCVGenerator — Conv2D 代码生成

```python
class CNNRISCVGenerator:
    def _gen_conv(self, node, memory_plan):
        """生成 Conv2D 的 RISC-V 内联代码"""
        # 外三层: OH, OW, OC
        for oh in range(OH):
            for ow in range(OW):
                for oc in range(OC):
                    # 累加器清零
                    self.emit("addi", acc_reg, "x0", 0)
                    # 内三层: KH, KW, IC
                    for kh in range(KH):
                        for kw in range(KW):
                            # 指针步进优化（最强优化）
                            for ic in range(IC):
                                self.emit("lw", t0, in_ptr, 0)
                                self.emit("lw", t1, wt_ptr, 0)
                                self.emit("mul", t2, t0, t1)
                                self.emit("srai", t2, t2, 16)
                                self.emit("add", acc_reg, acc_reg, t2)
                    # + bias
                    self.emit("add", acc_reg, acc_reg, bias_reg)
                    self.emit("sw", acc_reg, out_ptr, 0)
```

### RISCVEmitter — 两遍扫描

```
第一遍：emit 所有指令，记录每个 label 的位置
第二遍：回填分支/跳转指令的偏移量（label 位置已知了）
```

---

## 动手练习

### 练习 1: 编译一个简单 ONNX 模型

用 `make bench-cnn` 编译 CNN 模型，阅读生成的报告和汇编。

### 练习 2: 分析生成的汇编

打开 `output.s`，找到 Conv2D 最内层循环的指令序列。数一数每个 MAC（乘加）需要多少条指令。

### 练习 3: 修改 Q16.16 精度

在 `MemoryPlan` 中找到 `float32_to_q16` 函数，尝试改用 Q8.8（乘以 256 而不是 65536），观察对动态指令数和精度的影响。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **Protobuf 解析兼容性** | 手工解析只支持有限的 wire type，不同 ONNX opset 版本可能用不同的编码方式 |
| **Q16.16 溢出** | 两个 Q16.16 值乘法结果是 Q32.32（需要 64 位），截断为 32 位需要 `srai 16`，但如果整数部分超过 16 位就会溢出 |
| **内存地址不重叠** | MemoryPlan 需要确保不同 tensor 的地址空间不重叠，否则会互相覆盖数据 |
| **TinyFive 速度** | TinyFive 仿真很慢（~1000 instr/s），大量指令可能需要设置 `--tinyfive-max-instr` 限制 |

---

## 进阶阅读

- [ARCHITECTURE.md](../../../reference/ARCHITECTURE.md) — 完整架构文档，包含双路径对比
- [03-指标解读指南](../../../guide/03-指标解读指南.md) — 如何解读动态指令数和缓存指标
- Protobuf wire format: [Encoding](https://protobuf.dev/programming-guides/encoding/)
- 相关 topic: [课题22 — Standalone LLVM 编译器](../22-Standalone-LLVM编译器/22-Standalone-LLVM编译器.md) | [课题3 — IR 系统](../03-IR系统/03-IR系统.md)

---

## 自学路线

- **第 1 周**：运行 `make bench-cnn`，阅读五大组件源码（从 ProtoReader 到 RISCVEmitter 逐组件阅读）。画出数据流图：每个组件的输入/输出分别是什么。
- **第 2 周**：深入 Conv2D 代码生成。打开 `output.s`，手动标注最内层循环的每条指令。理解 Q16.16 乘法的 mul→srai→add 三步模式。计算每个 MAC 需要多少条指令。
- **第 3 周**：对比 ScratchV Standalone 和 LLVM Standalone（课题 22）的代码生成策略差异。使用 `llvm_cache_compare.py` 和 `tinyfive_compare.py` 对比两者的静态/动态指令数和 Cache 行为。
- **第 4 周**：尝试修改 CNNRISCVGenerator，为一个新算子（如 AveragePool）添加代码生成支持。编写完整测试：ONNX 模型 → 编译 → 汇编 → TinyFive 仿真验证。
