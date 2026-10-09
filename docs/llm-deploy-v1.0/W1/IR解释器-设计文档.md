# ScratchV IR 解释器设计文档（第二版·精简版）

正式交付、CI 和他人复现使用 Ubuntu 24.04 x86_64 / Bash / Python 3.12，环境与工具准备见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。命令中的 `python` 指已激活的 `.venv-linux/bin/python`。

> 版本：v0.2，本轮实现已落地；实际支持规则见使用说明  
> 日期：2026-09-30

配套文档：[开发文档](IR解释器-开发文档.md)、[使用说明](IR解释器-使用说明.md)。

## 1. 目标与范围

实现一个接收 **Program 对象、输入数组和权重数组** 的解释器，按照 IR 描述的操作和控制流，使用 NumPy 计算并返回结果。

本轮交付包括：

1. 按 ONNX 张量算子的粒度补齐必要的 IR opcode，并定义操作数、属性和类型规则。
2. 实现这些操作的 NumPy 执行逻辑，以及 Program 的变量绑定、分支、循环和返回机制。
3. 提供直接构造 Program 的可运行示例和正确性测试。
4. 将测试与小规模 benchmark 加入 CI，生成结果与耗时报告。

前端本轮暂缓。示例通过 IRBuilder 或 IR 类型直接构造 Program；真实 ONNX 模型与 ONNX Runtime 的对照作为后续集成任务。

## 2. 当前基础

仓库已有 `IRTraceExecutor`，能接收 Program 和数组，执行部分 NumPy 操作；主要缺口是：

- 按列表顺序遍历代码，没有真正执行循环、分支和提前返回。
- 未知指令被跳过，缺失变量可能被替换为零。
- 部分操作存在改变语义的处理，例如 DIV 添加 epsilon、Reshape 失败后展平。

本轮建立独立、严格的解释器入口，审查并复用现有算子公式。复用现有 IRVerifier 检查 IR，复用共享 CFG 层的循环规范化规则。

## 3. 输入与接口

建议最小接口：

```python
result = IRInterpreter(program).run(
    inputs={"x": x_array},
    initializers={"W": weight_array},
    function_name="main",
    max_steps=1_000_000,
)

y = result.return_value
steps = result.executed_steps
```

| 参数或结果 | 含义 |
|---|---|
| `program` | ScratchV 的 Program 对象，包含函数、基本块、指令和全局值声明 |
| `inputs` | 入口参数名到 NumPy 数组的映射 |
| `initializers` | 全局值名到实际权重或常量数组的映射；无全局张量时可省略 |
| `function_name` | 选择入口函数；Program 只有一个函数时可省略 |
| `max_steps` | 实际执行指令数上限，避免无限循环 |
| `return_value` | 实际 RETURN 指令的结果；无操作数的 RETURN 返回 None |
| `executed_steps` | 实际执行的计算与控制指令数，包含循环规范化产生的指令 |

绑定规则：

- 输入名称与入口参数匹配；缺失、额外输入和未提供数据的全局张量报错。
- dtype 与 IR 声明一致；结果核对已声明的形状，不自动转换类型或展平。
- 当前 `Value.shape=()` 也可能表示未填写形状，不能一律判为标量；非空固定形状按声明校验。
- 标量字面量按 IR 的类型和值建立；`locals` 声明不等于已初始化。
- 每次运行创建独立状态，不修改 Program、调用者的输入和权重。
- RETURN 立即结束执行；不能用最后一个计算结果代替返回值。

## 4. IR 操作与 NumPy 实现

### 4.1 操作粒度

一个 opcode 表示一次张量操作。例如 MATMUL 直接执行矩阵乘法，ADD 执行逐元素加法，SOFTMAX 执行沿指定轴的归一化。

本轮采用基础操作组合表达 RMSNorm 等计算，优先覆盖下面的案例。每个 opcode 必须定义操作数、属性、dtype、广播及形状规则。使用 ONNX 的粒度不要求原样复制其数据结构；解释器执行已定义的 IR 语义。

### 4.2 已有 opcode 的重点支持

| 操作 | 执行要求 |
|---|---|
| ADD、SUB、MUL、DIV、NEG | 逐元素计算；二元操作支持合法广播；浮点 DIV 不额外添加 epsilon |
| LOAD_CONST | 按结果 dtype 创建常量 |
| MATMUL | 使用 `np.matmul` 的张量乘法规则，校验维度与相关属性 |
| RESHAPE | 元素总数不变；明确处理形状中的 0 和 -1；非法形状报错 |
| TRANSPOSE | 按 perm 排列轴；省略时逆转轴顺序 |
| CONCAT | 沿 axis 拼接，其他轴大小一致 |
| SIGMOID、SOFTMAX | 使用稳定计算公式；Softmax 明确 axis 和负轴规则 |

已有 RELU、EXP、GELU、DOT、GEMM、CONV、MAXPOOL 按各自登记的能力复用并补充测试，不因已有枚举就默认支持全部属性变体。GELU 的近似方式、卷积和池化的布局及属性限制应写入对应算子的契约。

### 4.3 需要补齐的 opcode

依据当前 W1 小 Transformer 的操作需求，新增以下基础操作：

| opcode | 操作数 | 属性 | 主要语义 |
|---|---|---|---|
| SQRT | 浮点数据 | 无 | 逐元素平方根 |
| REDUCE_MEAN | 浮点数据 | axes、keepdims | 沿指定轴求均值 |
| GATHER | 数据、整数索引 | axis | 沿指定轴取元素；结果类型与数据一致 |
| SLICE | 数据 | starts、ends、axes、steps | 按指定范围和步长切片 |
| UNSQUEEZE | 数据 | axes | 插入长度为 1 的轴 |
| EXPAND | 数据 | shape | 按目标形状进行合法广播 |

本轮 shape、axes 和切片参数使用静态属性，暂不扩展动态参数张量形式。属性的默认值、负轴及边界规则在算子实现前明确，并覆盖相应测试。

新增操作需要同步完成：

```text
OpCode → IRBuilder → IRVerifier 签名 → NumPy handler → 示例与测试
```

GATHER 的索引为 i32/i64，可与数据类型不同，验证器需要支持这种签名。打印器、优化 pass 等使用新指令时应保留其语义；不支持的处理路径明确报错。

## 5. 执行机制与错误处理

执行流程：

```text
验证 Program → 选择入口 → 检查支持能力与数据绑定
→ 建立执行计划 → 按控制流执行 → RETURN
```

- 从入口函数的首块开始，用程序计数器维护当前位置。
- BR 跳转到目标块；BR_IF 只执行实际选中的分支，条件必须是合法标量。
- FOR/ENDFOR 复用共享规范化规则，支持嵌套、零次循环和正步长；循环体按实际次数执行。
- 普通块之间的转移遵守 CFG，不能仅按块列表顺序执行。
- 循环规范化产生的指令关联原 IR 位置，内部变量与标签避开已有名字。
- 每条指令执行前检查 max_steps；没有合法返回、未定义值或能力不足时明确失败。

错误信息至少包含函数、原始块、指令位置、opcode 和原因。NumPy 的类型或形状错误包装后保留原始原因。未知 opcode、非法属性和缺失数据不能被跳过、回退或替换为零。

当前多函数调用、多值返回、LABEL 和 PHI 不作为本轮支持目标；依据现有 IR 与验证器限制明确拒绝。

## 6. 示例与正确性测试

示例直接构造 Program，附固定输入、权重和独立预期结果。示例脚本、测试和 benchmark 复用同一份案例定义。

| 案例 | 覆盖内容 |
|---|---|
| MatMul → Add → Softmax | 张量计算、广播、轴、返回值 |
| 基础操作组成的 RMSNorm | Mul、ReduceMean、Add、Sqrt、Div |
| Gather 读取 embedding | 整数索引、输出形状、越界错误 |
| Slice → Unsqueeze → Expand → Concat | 切片、形状变化、广播与拼接 |
| 条件分支 | 两条分支返回不同结果，未选分支不计算 |
| 循环 | 零次、多次、嵌套和正步长；验证实际返回值与执行计数 |

同时覆盖输入缺失、dtype/shape 错误、非法属性、未支持操作、执行超限、重复运行和失败后重试。

参考结果使用手算的小数组、独立实现或已验证的参考工具，不调用待测 handler 生成答案。先检查 shape 和 dtype，再检查数值：整数精确比较，浮点明确记录 atol/rtol。不得通过广播比较或截短数组掩盖错误。

## 7. CI 与 benchmark

### CI

- 在现有 CI 中增加解释器正确性测试和小规模 benchmark。
- 核心依赖为 Python、NumPy、pytest，直接构造 Program 的测试可独立运行。
- 数值不匹配、执行失败、超时或报告缺失时返回非零。
- 保存测试结果与 benchmark 的 JSON/Markdown 报告。

### Benchmark

- 使用固定案例、输入规模和随机种子；先检查正确性，再预热并重复运行。
- 计时覆盖一次 `run()`，包括验证、绑定、执行和返回；Program 构造、输入生成及参考答案计算放在计时之外。
- 报告包含案例、shape/dtype、环境与线程配置、执行计数、误差、重复次数、原始耗时和中位数。
- 每次重复仍校验结果；首轮 CI 以正确性和报告完整性为门槛，耗时用于观察。

接入 `benchmarks/` 与现有 CI 工作流，报告明确标识为 IR 解释器执行耗时。

## 8. 实施顺序与验收

1. 明确最小接口和各 opcode 契约，搭建值绑定与直线执行。
2. 补齐已有操作、新 opcode、Builder 和验证器签名。
3. 实现分支与循环，验证返回、计步和错误定位。
4. 提供示例和测试，接入 CI 与 benchmark。

本轮完成标准：约定案例均得到正确结果，错误输入明确失败，重复运行无状态残留，CI 通过并生成完整报告。前端暂缓不影响这些验收。

本轮已确认并实现两项 IR 契约：

- **整数算术**：ADD/SUB/MUL/NEG 固定宽度回绕；DIV 向零截断，除零及最小整数除以 -1 报错。
- **局部内存**：ALLOCA 的 size 为字节数并按 dtype 对齐；LOAD/STORE 读写首个同类型标量槽位，未初始化读取报错。

具体操作默认值、数值边界及命令见使用说明。旧 IRTraceExecutor 的调用迁移另行安排。

## 9. 代码与文档参考

- [IR 类型](../../../scratchv/ir/types.py)、[IRBuilder](../../../scratchv/ir/builder.py)。
- [IR 验证器](../../../scratchv/analysis/ir_verifier.py)、[共享 CFG 适配器](../../../scratchv/analysis/adapters.py)。
- [现有 IRTraceExecutor](../../../scratchv/simulator/rv32_emulator.py)。
- [CI 工作流](../../../.github/workflows/ci.yml)、[现有 benchmark](../../../benchmarks/run_benchmark.py)。
- [W1 执行计划](W1-执行计划.md)、[W1 接口草案](interfaces.md)。
