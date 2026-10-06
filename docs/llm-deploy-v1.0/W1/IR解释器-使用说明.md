# ScratchV IR 解释器使用文档

正式交付、CI 和他人复现使用 Ubuntu 24.04 x86_64 / Bash / Python 3.12，环境与工具准备见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。命令中的 `python` 指已激活的 `.venv-linux/bin/python`。

> 日期：2026-09-30  
> 输入：直接构造的 Program、输入数组及初始化数组。

本文面向调用解释器、检查结果和运行回归的使用者。配套文档：[设计文档](IR解释器-设计文档.md)、[开发文档](IR解释器-开发文档.md)。

解释器接收 `Program` 对象和 NumPy 数组，按 IR 操作与控制流执行，返回 NumPy 结果。本轮通过 IRBuilder 直接构造 Program；ONNX 前端的补齐和真实模型接入见设计文档的后续范围。

## 1. 安装与运行示例

建议使用 Python 3.12，与当前 CI 配置一致。在仓库根目录安装项目，然后运行示例；下文命令均在仓库根目录执行：

```bash
python -m pip install -e .
python -m examples.run_ir_interpreter --case matmul_add_softmax
python -m examples.run_ir_interpreter --case rmsnorm
python -m examples.run_ir_interpreter --case loop_sum
```

示例会检查预期结果，输出 PASS、执行计数和返回值。案例名称见 `benchmarks/ir_interpreter_cases.py`；包含张量计算、索引、形状操作、分支、零次及嵌套循环，共 11 个案例。

## 2. Python 接口与 Program 构造

### 2.1 最小例子：输入加常量

```python
import numpy as np
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.verification.ir_interpreter import IRInterpreter

x = Value("x", dtype=DataType.FLOAT32, shape=(2, 2))
builder = IRBuilder()
builder.new_function("main", params=[x])
builder.new_block("entry")
builder.ret(builder.add(x, builder.make_const(2.0)))

result = IRInterpreter(builder.program).run(
    {"x": np.array([[1, 2], [3, 4]], dtype=np.float32)},
    function_name="main",
    max_steps=1_000_000,
)
print(result.return_value)       # [[3, 4], [5, 6]]
print(result.executed_steps)     # 2：ADD 和 RETURN
```

### 2.2 带权重的张量计算

下面构造 `softmax(x @ W + bias)`。`x` 是函数参数；`W`、`bias` 是全局声明，由 `initializers` 提供数据：

```python
import numpy as np
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.verification.ir_interpreter import IRInterpreter

x = Value("x", shape=(2, 3))
w = Value("W", shape=(3, 2))
bias = Value("bias", shape=(2,))

b = IRBuilder()
b.new_function("main", params=[x])
b.new_block("entry")
b.program.global_values.extend([w, bias])
b.ret(b.softmax(b.add(b.matmul(x, w), bias)))

result = IRInterpreter(b.program).run(
    inputs={"x": np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")},
    initializers={
        "W": np.array([[1, -1], [2, 0], [0, 1]], dtype="float32"),
        "bias": np.array([1, -1], dtype="float32"),
    },
)
print(result.return_value)
# [[9.9330717e-01, 6.6928505e-03],
#  [9.9999917e-01, 8.3152798e-07]]
```

权重通过 `initializers={"W": weight_array}` 绑定到 `Program.global_values` 中的同名声明。Program 不保存完整权重数组，调用者需要提供实际数据。

### 2.3 循环与局部存储

下面直接构造一个把 0、1、2、3、4 累加的 Program：

```python
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType
from scratchv.verification.ir_interpreter import IRInterpreter

b = IRBuilder()
b.new_function("main")
b.new_block("entry")
slot = b.alloca(4, DataType.INT32)
b.store(slot, b.make_const(0, DataType.INT32))
i = b.for_loop(0, 5)
b.store(slot, b.add(b.load(slot), i))
b.endfor()
b.ret(b.load(slot))

result = IRInterpreter(b.program).run({})
print(result.return_value)    # 10，int32 标量
print(result.executed_steps)  # 37，含循环规范化后的控制指令
```

### 2.4 调用参数与返回值

入口为 `IRInterpreter(program).run(inputs, *, initializers=None, function_name=None, max_steps=1_000_000)`。

| 参数 | 用法 |
|---|---|
| `program` | 已构造的 `Program` 对象 |
| `inputs` | 参数名到 `np.ndarray` 的字典；无参数函数传 `{}` |
| `initializers` | 全局声明名到 `np.ndarray` 的字典，例如权重和偏置 |
| `function_name` | 入口函数名；仅有一个函数时可省略 |
| `max_steps` | 正整数执行步数上限；超限抛出 `StepLimitExceeded` |

返回 `ExecutionResult`：`return_value` 是返回数组或 `None`，`executed_steps` 是实际执行计数，`diagnostics` 保存静态验证诊断。

当前解释器未记录逐条指令的中间值日志。

- inputs 必须准确匹配入口参数，dtype 必须一致；非空固定 shape 按声明校验。
- `shape=()` 沿用当前 IR 的默认元数据约定，视为未声明完整形状；标量要求在控制流、存储等具体操作处检查。
- 仅选择一个入口函数；多函数 Program 必须指定名称。
- RETURN 立即返回；空 RETURN 的 return_value 为 None。
- 每次 run 建立独立状态，输入、权重和 Program 不被修改。
- `result.diagnostics` 保存静态验证诊断，包括不阻止执行的 WARNING。
- 执行失败抛出 `IRExecutionError`，含 code、函数、原始块、指令索引、opcode、阶段等信息。

### 2.5 错误定位

捕获 `IRExecutionError` 后，可读取具体错误类型和原始 IR 位置。例如沿用 2.1 中的 `builder.program`，传入错误类型的输入：

```python
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter

try:
    IRInterpreter(builder.program).run(
        {"x": np.array([[1, 2], [3, 4]], dtype="float64")}
    )
except IRExecutionError as error:
    print(error.code)  # DTypeError
    print(error.function_name, error.block_name, error.instruction_index)
    print(error)
```

| 常见错误 | 检查内容 |
|---|---|
| `BindingError` | 输入名、权重名是否与声明一致，是否缺少绑定 |
| `DTypeError` / `ShapeError` | 数据类型、输入/结果形状或广播是否满足要求 |
| `NumericError` | 除零、负数开方、浮点溢出、空归约等 |
| `IndexError` | GATHER 索引是否越界 |
| `MemoryError` | ALLOCA 尺寸、标量存储类型、未初始化读取等 |
| `InvalidProgram` | Program 的定义、类型和控制流是否通过静态验证 |
| `UnsupportedOpcode` / `UnsupportedAttribute` | 使用了未支持的操作、属性或变体 |
| `StepLimitExceeded` | 执行次数是否超限；必要时检查循环或调整上限 |

运行时指令错误的位置使用原始 block 和从 0 开始的指令索引。循环合成指令另外标记 `for-init`、`for-test` 或 `for-step` 阶段。输入绑定等发生在执行前的错误可能没有 block 或指令位置。

## 3. 支持的操作与属性

默认数据类型为 IR 声明的 f32/f64/i32/i64。下表中的“浮点”操作只登记 f32/f64，其余数值或形状操作支持四种类型。

| 操作 | 属性及限制 |
|---|---|
| ADD/SUB/MUL/DIV/NEG | 同类型；二元操作支持广播；除零报错 |
| LOAD_CONST | value；保持声明 dtype |
| MATMUL | 支持 NumPy 张量乘法；m/n/k 可一起省略；提供时校验二维形状，两个扁平缓冲按指定尺寸恢复 |
| DOT | length 必填，只支持等长一维向量 |
| RESHAPE | shape 必填；0 复制对应输入维度；至多一个 -1；不支持 allowzero 变体 |
| TRANSPOSE | perm 为非负轴的完整排列；省略时逆转维度顺序 |
| CONCAT | axis 必填，可为负轴 |
| SQRT | 浮点；负值报错 |
| REDUCE_MEAN | 浮点；axes 省略或为空时归约所有轴；keepdims 默认 True；重复轴及空归约维度报错 |
| GATHER | axis 默认 0；索引为 i32/i64，可为负索引；越界报错 |
| SLICE | starts/ends 必填；axes 默认顺序轴，steps 默认全 1；采用 Python 的截断边界语义，支持负步长，零步长报错 |
| UNSQUEEZE | axes 必填，按输出 rank 解释负轴；重复轴报错 |
| EXPAND | shape 必填且维度非负；输出为输入形状与该 shape 的广播结果 |
| RELU | 保持类型和形状 |
| EXP/SIGMOID/GELU/SOFTMAX | 浮点；GELU 使用 tanh 近似；Softmax 的 axis 默认 -1 |
| GEMM | 浮点二维 A/B；trans_a/trans_b 默认 False；alpha/beta 默认 1；bias 广播到乘法结果形状 |
| CONV | 浮点 NCHW 输入与 OIHW 权重，bias 为通道向量；方形核，kernel_size 默认 3，stride 默认 1，padding 默认 1；可提供 out_channels 进行核对；group/dilation 变体未支持 |
| MAXPOOL | 浮点 CHW/NCHW；kernel/stride 必填；无 padding、ceil_mode 或 indices 输出 |
| BR/BR_IF/RETURN/FOR/ENDFOR | 按真实路径执行；BR_IF 使用零维整数条件或同类型零维比较；FOR 使用正 i32 step |
| ALLOCA/LOAD/STORE | 受管局部标量存储，规则见下文 |

未知操作、属性和变体明确报错。循环规范化后的实际计算及控制指令计入 executed_steps，合成标签不计步；计数不能作为 RISC-V 指令数。

### 数值与局部存储规则

- i32/i64 的 ADD/SUB/MUL/NEG 固定宽度回绕。例如 i32 的 `2147483647 + 1` 得到 `-2147483648`。
- 整数 DIV 向零截断，例如 `-7 / 3 = -2`；除零和最小整数除以 -1 报错。
- FOR 归纳变量递增超过 i32 范围时报错，避免回绕形成意外循环。
- ALLOCA 的 size 为字节数，省略时沿用 4；必须为正、至少容纳一个元素且按 dtype 对齐。i64/f64 槽位至少为 8 字节。
- LOAD/STORE 只访问分配区的首个标量槽位，类型必须一致；未初始化读取报错。存储引用不能参与数值计算或作为返回值。
- 浮点绑定拒绝 NaN 和 +inf；允许 -inf 用于掩码。数值算子输出和正式返回值要求有限；Softmax 部分掩码有效，全掩码或空归一化轴报错。下溢到零允许。

## 4. 测试与 benchmark

### 4.1 测试目录和运行方式

自动化测试放在仓库根目录的 `tests/`：

| 文件 | 验证内容 |
|---|---|
| `tests/test_ir_interpreter.py` | 完整 Program、输入绑定、状态隔离、返回副本和错误定位 |
| `tests/test_ir_interpreter_ops.py` | 算子结果、dtype、广播、形状与数值/索引边界 |
| `tests/test_ir_interpreter_control_flow.py` | 分支、循环、步数上限、跨块执行和局部内存 |
| `tests/test_ir_interpreter_benchmark.py` | benchmark、示例 CLI、summary 和错误报告 |
| `tests/test_licm_safety.py` | 优化器外提前后的语义对照，属于 LICM 回归 |

```bash
python -m pip install pytest
python -m pytest tests/test_ir_interpreter*.py -q
python -m pytest tests/test_licm_safety.py -q
```

公共 Program、输入、权重和参考结果在 `benchmarks/ir_interpreter_cases.py`。`examples/` 提供手动运行脚本。相同案例可用于测试和 benchmark，具体错误和边界行为另由专用测试验证。

### 4.2 运行 benchmark

```bash
python -m benchmarks.bench_ir_interpreter --warmup 1 --repeats 3 --json-output benchmark_reports/ir_interpreter.json --markdown benchmark_reports/ir_interpreter.md
```

可用 `--case rmsnorm` 选择单个 benchmark，或重复指定 `--case`。省略时执行 6 个张量案例：MatMul+Bias+Softmax、RMSNorm、Gather embedding、形状操作链、32×32 和 128×128 MatMul。分支、空循环、步长和嵌套循环保留在执行测试中，也可通过 `--case` 显式运行。

计时覆盖一次完整 run，包含验证、执行计划、数据绑定副本、计算和返回副本。参考答案、Program 构造及数值比较在计时之外；预热和每次重复都校验结果。JSON 保存环境、线程配置、输入规格、操作覆盖、误差和原始耗时，Markdown 提供汇总。

数值错误、执行失败或报告写入失败返回非零。CI 已加入专用测试与小规模 benchmark，上传 JUnit XML、JSON/Markdown，并把 benchmark 汇总加入任务摘要。GitHub 上的实际运行结果需由 CI 确认。

### 4.3 阅读 summary

默认 6 个 benchmark 案例通过时，开头显示：

```text
总体状态：PASS
Summary: 6/6 PASS, 0 FAIL
```

`6/6` 对应本次选中的 benchmark 案例数量；pytest 的测试数量由测试收集结果给出。

summary 按四项组织：

1. **总体状态**：是否全部通过，以及通过/失败数量。
2. **案例结果**：操作、预期输出 shape/dtype、正确性和最大绝对误差。
3. **参考依据与容差**：独立参考来源；整数精确比较，浮点要求 `abs(actual - expected) <= atol + rtol * abs(expected)`，公共浮点案例默认 `atol=rtol=1e-6`。
4. **失败定位**：失败案例、错误原因，以及能够取得的函数、block、原始指令索引、opcode 和阶段。绑定或输出比较错误可能没有单条指令位置。

benchmark 的数值误差是校验过的执行结果与参考结果之差；没有成功比较时标为“未获得”。主机耗时受环境影响，执行计数还包含控制指令，阅读性能数据时应结合计时范围和 JSON 中的环境记录。

### LICM 安全外提

LICM 使用统一安全判断，取代六个新 opcode 的临时禁止外提规则。只有输入定义支配插入点（或已按依赖顺序外提），且能证明提前执行不会改变错误行为的指令才移动。判断适用于同一 IR block 内配对的 FOR/ENDFOR；先处理内层循环，再独立判断是否能移出外层。

- 整数 ADD/SUB/MUL 需要证明广播兼容；整数 NEG 按回绕规则可执行。浮点计算需要额外的数值安全证明。
- DIV 检查已知非零除数和整数最小值除以 -1；SQRT 检查有限、非负输入，可识别 RELU 输出的范围。
- GATHER 检查静态轴长和常量索引；SLICE/UNSQUEEZE/EXPAND/RESHAPE/TRANSPOSE/CONCAT 检查形状、轴、广播和结果形状。浮点数据还需证明有限。
- REDUCE_MEAN 检查非空归约维度及中间求和的溢出风险，可使用 SIGMOID 的 [0, 1] 范围；有限输入本身不足以排除求和溢出。
- 上述规则用于证明推测执行安全。对于固定边界满足 start < end 的循环，还可以外提必定执行的纯计算：输入须循环不变且在插入点可用，首轮到达该指令前不能存在保留的潜在失败、内存操作或控制流指令。多条外提指令保持原执行顺序，因此运行时数值或形状错误仍按原计算顺序发生。
- LOAD/STORE/ALLOCA 保留原位，并截断上述必定执行的前缀证明。属性不支持、既无推测安全证明也无必定执行证明的操作保留原位；无效 IR 不进行外提。

属性检查及标量常量的安全计算复用无状态 NumPy kernel，不读取运行时输入张量。未知 opcode 默认不外提，需补充证明规则及优化前后执行对照测试。零次循环不能使用必定执行证明；跨块条件分支循环仍保留原位，当前不插入动态保护分支。已知会失败的标量常量表达式保留原位置，便于诊断。

## 5. 当前边界

本轮不支持 CALL、多值返回、原始 LABEL/PHI、外部指针和动态参数张量形式的形状操作。未知维度的编译期推导和完整 ONNX 前端接入属于后续工作。

解释器运行原始 Program；新增六个操作尚无 LLVM/RISC-V 代码生成支持，这些后端会明确拒绝。完整优化管线的语义验收另行进行。

max_steps 约束解释器指令执行次数，无法中断一次大型 NumPy/BLAS 调用；CI 同时设置步骤超时。

## 6. 相关文档

- [设计文档](IR解释器-设计文档.md)。
- [开发文档](IR解释器-开发文档.md)。
- [解释器实现](../../../scratchv/verification/ir_interpreter.py)、[算子实现](../../../scratchv/verification/ir_numpy_ops.py)。
- [公共案例](../../../benchmarks/ir_interpreter_cases.py)、[benchmark 入口](../../../benchmarks/bench_ir_interpreter.py)。
- [手动运行示例](../../../examples/run_ir_interpreter.py)。
