# ScratchV IR 解释器开发文档

正式交付、CI 和他人复现使用 Ubuntu 24.04 x86_64 / Bash / Python 3.12，环境与工具准备见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。命令中的 `python` 指已激活的 `.venv-linux/bin/python`。

> 版本：v0.1，开发说明；本轮实现已落地  
> 日期：2026-09-30  
> 依据：[IR 解释器设计文档（第二版·精简版）](IR解释器-设计文档.md)

## 1. 开发目标

按设计文档实现 Program 的 NumPy 解释器，交付必要的 opcode 扩展、可运行示例、测试、CI 和 benchmark。本轮直接构造 Program，不依赖 ONNX 前端补齐。

本文记录修改位置、实施顺序和验证方法。相关接口和命令已实现，实际支持规则与运行方式见[使用说明](IR解释器-使用说明.md)。

## 2. 文件与职责

| 文件 | 改动 | 职责 |
|---|---|---|
| `scratchv/verification/ir_interpreter.py` | 新增 | 入口、运行状态、数据绑定、控制流、返回、计步、错误定位 |
| `scratchv/verification/ir_numpy_ops.py` | 新增 | opcode 到 NumPy handler 的登记、属性与运行时形状检查 |
| `scratchv/ir/types.py` | 修改 | 六个新 opcode 及相关分类 |
| `scratchv/ir/builder.py` | 修改 | 新操作构造方法、已有方法的类型处理 |
| `scratchv/analysis/ir_verifier.py` | 修改 | 新操作签名、GATHER 混合类型检查 |
| `scratchv/analysis/adapters.py` | 修改 | 提取可共享的循环执行计划及原位置映射 |
| `scratchv/optimizer/licm.py`、`scratchv/optimizer/hoist_safety.py` | 修改/新增 | 插入点定义可用性、安全外提证明、按依赖顺序移动 |
| `benchmarks/ir_interpreter_cases.py` | 新增 | 公共 Program 案例、输入、权重和独立预期结果 |
| `examples/run_ir_interpreter.py` | 新增 | 选择案例、调用解释器、展示返回值 |
| `benchmarks/bench_ir_interpreter.py` | 新增 | 正确性检查、计时、JSON/Markdown 报告、CLI |
| `benchmarks/ir_interpreter_summary.py` | 新增 | benchmark 与 CI 的状态、案例、参考依据/容差和失败定位汇总 |
| `tests/test_ir_interpreter*.py` | 新增 | 执行核心、算子、控制流和报告测试 |
| `tests/test_licm_safety.py` | 新增 | 安全/危险指令、零次循环、嵌套和连续循环、内存与定义顺序对照 |
| `tests/test_ir.py`、`tests/test_ir_verifier.py` | 修改 | Builder 与静态签名的回归测试 |
| `.github/workflows/ci.yml` | 修改 | 解释器测试与 benchmark 步骤、报告上传和摘要 |

同时审查 `scratchv/ir/printer.py`、`scratchv/analysis/usedef.py` 和相关优化 pass 对新指令的处理。保留既有字段和调用方式；本轮不迁移旧 IRTraceExecutor 的所有使用者。

## 3. 先确定操作契约

为每个支持的 opcode 在代码文档或登记表中写明：操作数数量、结果类型、属性及默认值、广播和形状规则、支持变体、错误条件。对应边界必须有测试。

### 新操作登记

| opcode | Builder 方法建议 | 静态签名 | NumPy 实现方向 |
|---|---|---|---|
| SQRT | `sqrt(x)` | 一个浮点输入，结果同类型 | `np.sqrt`，按契约处理负数 |
| REDUCE_MEAN | `reduce_mean(x, axes, keepdims)` | 一个浮点输入，结果同类型 | 归一化轴后使用 `np.mean` |
| GATHER | `gather(data, indices, axis)` | 数据 + i32/i64 索引，结果类型等于数据 | 检查索引及 axis 后使用 `np.take` |
| SLICE | `slice(x, starts, ends, axes, steps)` | 一个输入，结果同类型 | 按 IR 规则归一化参数，再构造切片 |
| UNSQUEEZE | `unsqueeze(x, axes)` | 一个输入，结果同类型 | 按输出 rank 归一化轴，再插入维度 |
| EXPAND | `expand(x, shape)` | 一个输入，结果同类型 | 校验广播规则，建立广播结果 |

shape、axes 和切片参数保存为静态属性。Builder 负责构造指令；能从 IR 判定的属性错误由验证器或能力预检报告，依赖实际数组的错误由运行时报告。

`OPCODE_SPECS` 当前的 T/F 家族要求输入与结果同类型，不能直接用于 GATHER。为它增加专门的签名检查：数据与结果类型相同，索引类型为 i32/i64；不放宽其他指令的类型检查。

### 已有 Builder 的配套修正

- 当前许多方法用默认 FP32 创建 dest。支持 FP64 或整数的操作应按合法输入类型建立结果；非法混用类型不能靠转换掩盖。
- 补充 TRANSPOSE、CONCAT 的公开构造方法。
- MATMUL 现有方法要求 m/n/k。张量案例需要兼容的构造方式；若允许省略这些属性，保留已有调用，校验提供属性时的一致性，不自动展平批量输入。
- 不强制增加通用编译期形状推导；固定形状可在案例中明确登记。

### 两项已确认的契约

1. **整数算术**：ADD/SUB/MUL/NEG 固定宽度回绕；DIV 向零截断，除零及最小整数除以 -1 报错。FOR 递增超过 i32 范围报错。
2. **局部内存**：ALLOCA 按字节分配并校验 dtype 对齐；LOAD/STORE 访问首个同类型标量槽位，未初始化读取报错。

上述规则已同步到实现、设计文档及测试。

## 4. 实现执行核心

### 4.1 最小公开接口

```python
@dataclass(frozen=True)
class ExecutionResult:
    return_value: Optional[np.ndarray]
    executed_steps: int

class IRInterpreter:
    def __init__(self, program: Program): ...

    def run(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        initializers: Optional[Mapping[str, np.ndarray]] = None,
        function_name: Optional[str] = None,
        max_steps: int = 1_000_000,
    ) -> ExecutionResult: ...
```

每次 run 建立新的运行状态，至少包含值表、当前块、指令位置和计步器。返回值采用独立数组快照，输入和权重建立运行时副本，handler 不原地修改操作数。

### 4.2 入口与绑定

1. 调用 `verify_ir(program, stage="before-execution")`；它返回 `(passed, issues)`，存在 ERROR 时终止，WARNING 保留诊断且不单独导致失败。
2. 选择入口函数，拒绝空 Program、多函数歧义和不存在的函数名。
3. 检查选定函数的操作及属性是否已支持，包括不可达分支中的能力检查；检查过程不计算分支数据。
4. 将 inputs 绑定到参数，将 initializers 绑定到声明的全局张量，检查名称冲突、缺失数据、dtype 和固定 shape。
5. 解析常量时遵守显式定义优先规则，LOAD_CONST 使用属性及结果类型；未定义值报错。

错误使用统一的 `IRExecutionError`，包含错误类别、函数、原始块、指令索引、opcode、值名与原因。可以按需要增加子类；不要求复杂的诊断框架。NumPy 异常使用异常链保留原原因。

### 4.3 指令派发

计算指令走 handler 注册表，控制指令由执行驱动处理。计算 handler 接收已解析的数组及属性，返回结果；驱动统一检查结果 dtype/shape 并写入 dest。

执行每条指令前检查 max_steps，实际执行后计数；RETURN 也计步。RETURN 立即返回，块结束时只能按执行计划中合法的续接关系前进，没有合法转移或返回则报错。

## 5. 算子实现与验证

先实现 LOAD_CONST、浮点算术、MATMUL 和 RETURN，形成可数值验证的直线图；再加入形状操作、激活函数和六个新 opcode。

- DIV 删除旧执行器额外的 epsilon；Reshape 删除展平回退。
- MATMUL 保留批量广播语义；RESHAPE 显式实现 0/-1 规则。
- Softmax 明确轴，减最大值后计算；Sigmoid 使用稳定公式。
- 浮点计算保持声明精度，系数和常量按计算 dtype 建立。
- 对 SQRT 负值、空归约、非有限值和全掩码 Softmax 等数值边界明确规则，写入契约和测试；不得由 NumPy warning 隐式决定是否成功。
- 对负轴、负索引、越界、重复轴、零切片步长及不可广播形状分别测试。

RELU、EXP、GELU、DOT、GEMM、CONV、MAXPOOL 的公式可从现有代码提取，但需要登记属性限制并验证实际结果。CONV/MAXPOOL 等使用小型独立循环参考，避免将待测公式复制到预期结果中。

每增加一个 opcode，完成 Builder、验证器、handler 和测试再进入下一个。登记表没有的 opcode 或属性变体明确报不支持。

## 6. 控制流与局部存储

现有 `IRCFGAdapter` 在构造时调用私有的 `_normalize_for_endfor` 和 `_partition_ir_stream`。将这段逻辑提取为共享执行计划接口，使 CFG 适配器与解释器使用同一规则。

执行计划至少提供：规范化块、分支目标、合法续接关系，以及规范化指令到原块/指令位置的映射。合成名称避开已有名称，提取过程不修改原 Program。

- BR 直接跳转；BR_IF 单条件要求零维整数，双操作数比较要求同类型零维值及合法 cmp_op。
- FOR 按 start/end/正 step 执行，ENDFOR 递增后回到条件判断；覆盖零次、嵌套和跨块循环。
- 循环内部的归纳变量更新属于执行计划，不作为原 Program 的重复定义，也不回写原 IR。
- 对原 Program 先做静态验证；不要用原 SSA 检查直接拒绝规范化后的内部循环更新。
- 合成标签不计步；合成计算和控制指令计步。错误映射到原 FOR/ENDFOR 位置。

为 ALLOCA/LOAD/STORE 建立独立存储引用和值状态；覆盖未初始化读取、错误类型和非法引用。不能为了绕过静态定义规则添加循环体中重复定义的普通 IR 值。

## 7. 案例与测试组织

公共案例包含名称、Program、入口、inputs、initializers、独立参考结果和浮点容差。每次调用案例构造器产生独立对象。

| 测试文件建议 | 主要检查 |
|---|---|
| `tests/test_ir_interpreter.py` | 入口与绑定、常量、返回、输入不变、连续运行、失败后重试、错误定位 |
| `tests/test_ir_interpreter_ops.py` | 重点已有操作、六个新操作、dtype/shape/属性边界 |
| `tests/test_ir_interpreter_control_flow.py` | 分支、零次及多次循环、嵌套、局部存储、计步和超限 |
| `tests/test_ir_interpreter_benchmark.py` | CLI、报告字段、计时样本、结果错误导致失败 |
| 现有 IR/CFG 测试 | 新签名、Builder 类型、共享规范化改动的兼容性 |

设计文档中的六类示例均加入公共案例。分支测试让未选中的合法分支包含运行时错误，证明它没有执行；循环检查返回值及计步，不能只检查运行成功。未支持操作使用当前未支持但可由 IR 表达的指令或变体测试，不伪造无法构造的枚举成员。

比较先验证准确 shape 和 dtype。整数元素精确一致；浮点容差写在案例中，失败时报告最大误差及位置。小数组尽量用手算预期值，组合图使用独立参考实现，不通过待测 handler 或旧执行器取得答案。

### 最小示例代码

以下代码可运行；先构造合法 Program，再调用解释器入口：

```python
import numpy as np
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.verification.ir_interpreter import IRInterpreter

x = Value("x", dtype=DataType.FLOAT32, shape=(2, 2))
builder = IRBuilder()
builder.new_function("main", params=[x])
builder.new_block("entry")
y = builder.add(x, builder.make_const(2.0))
builder.ret(y)

result = IRInterpreter(builder.program).run(
    {"x": np.array([[1, 2], [3, 4]], dtype=np.float32)},
)
# 预期 return_value：[[3, 4], [5, 6]]，dtype 为 float32。
```

## 8. Benchmark 与报告

新增 `python -m benchmarks.bench_ir_interpreter`，支持 `--case`、`--warmup`、`--repeats`、`--json-output` 和 `--markdown`。拒绝未知案例、负 warmup 和非正 repeats。

执行顺序：构造案例和参考结果 → 正确性检查 → 预热 → 重复执行 → 写报告。使用 `time.perf_counter()` 包围每次 run，数值比较在计时区间外，每次重复仍校验。计时包含验证、计划建立、绑定副本、计算和返回。

JSON 与 Markdown 使用同一份报告数据，至少记录：

- 环境：Python、NumPy、平台、BLAS 与线程配置、种子。
- 案例：名称、输入 shape/dtype、操作覆盖、容差。
- 结果：正确性、最大误差、executed_steps。
- 耗时：warmup/repeats、samples_s、median_s、min_s、max_s。

失败时尽量保存已有结果及错误信息，但退出码必须非零，失败案例不能标为通过。报告写入失败也返回非零。运行超时由 CI 的步骤超时限制；max_steps 不能中断一次大型 NumPy 调用。

## 9. 本地验证与 CI 接入

### 本地命令

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest tests/test_ir_interpreter*.py tests/test_licm_safety.py tests/test_ir.py tests/test_ir_verifier.py tests/test_cfg_builder.py tests/test_cfg.py -q
python -m examples.run_ir_interpreter --case matmul_add_softmax
python -m benchmarks.bench_ir_interpreter --warmup 1 --repeats 3 --json-output benchmark_reports/ir_interpreter.json --markdown benchmark_reports/ir_interpreter.md
```

`pip install -e .` 使用项目现有依赖；解释器核心不增加 ORT、QEMU 或 LLVM 依赖。

### CI 修改位置

现有 `ci.yml` 的 test 和 benchmark job 已配置 Python，并通过环境变量 PY 选定解释器，报告放在 `benchmark_reports/`。

1. test job 的依赖安装后、RISC-V 工具准备前加入解释器专用测试，输出 `ir_interpreter_tests.xml`。新增测试也会被后面的全量 `tests/` 步骤收集。
2. test-reports 上传列表增加该 XML。
3. benchmark job 的依赖安装后、RISC-V 工具准备前加入小规模解释器 benchmark。
4. 固定 CI 的 BLAS 线程配置并记录实际配置；JSON/Markdown 随已有 benchmark-reports 上传，Markdown 加入 job summary。
5. 新增步骤不使用吞错误的命令或 continue-on-error；为步骤设置合理 timeout-minutes。

建议步骤命令：

```bash
mkdir -p benchmark_reports
$PY -m pytest tests/test_ir_interpreter*.py tests/test_licm_safety.py -q --junit-xml=benchmark_reports/ir_interpreter_tests.xml
$PY -m benchmarks.bench_ir_interpreter --warmup 1 --repeats 3 --json-output benchmark_reports/ir_interpreter.json --markdown benchmark_reports/ir_interpreter.md
```

解释器步骤可在工具链准备前执行；现有整个 job 仍有其他工具链门槛。此轮按现有工作流接入，不把独立运行能力误写成整个 CI 已脱离工具链。

## 10. 开发顺序与完成检查

| 阶段 | 交付 | 验证 |
|---|---|---|
| 1. 执行核心 | 最小接口、绑定、常量、浮点计算、RETURN、异常 | 最小示例及绑定/状态负例通过 |
| 2. 算子配套 | 重点已有操作、六个新 opcode、Builder、Verifier | 单算子和组合图结果正确，静态检查与运行规则一致 |
| 3. 控制流 | 共享计划、分支、循环；局部存储 | 实际路径、循环结果、计步及超限正确，CFG 回归通过 |
| 4. 交付接入 | 示例、benchmark、CI 与报告 | 命令可运行，错误导致非零退出，CI 保存完整产物 |

完成前检查：

- 设计中的六类案例、重点已有操作和新增操作都有独立结果验证。
- 未支持能力、错误输入和数值错误按已登记契约失败；异常定位正确。
- 输入、权重与 Program 未改变，多次运行及失败后重试无残留状态。
- Builder、IRVerifier、共享 CFG 的相关回归通过。
- benchmark 每次重复检查正确性，报告字段和计时范围一致。
- CI 有正确性门槛与报告产物；示例、接口说明及支持清单同步更新。

整数算术和局部内存按已确认规则实现。真实 ONNX 前端接入与模型对照留给后续任务。
