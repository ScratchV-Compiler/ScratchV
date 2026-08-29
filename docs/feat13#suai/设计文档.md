# ScratchV 窥孔优化器 Benchmark 技术设计文档

> **文档版本**：v1.0  
> **编写日期**：2026-08-05  
> **作者**：suai  
> **课题编号**：13（窥孔优化器）  
> **需求依据**：docs/feat13#suai/开发文档.md  
> **涉及模块**：benchmarks/、scratchv/compiler.py、scratchv/backend/asm_peephole.py、scratchv/backend/inst_counter.py

---

## 一、功能介绍

### 1.1 功能概述

本设计用于建设 ScratchV 汇编级窥孔优化器的 Benchmark。重点不是扩展复杂优化算法，而是建立一套可重复运行、统一计数、对比开关并生成报告的评测工具。

系统包含两条测试路径：

1. **合成汇编微基准**：生成不同规模、不同可优化序列比例的 RISC-V 汇编，测量 AsmPeepholeOptimizer.optimize() 的独立耗时、规则命中次数和静态指令变化；
2. **真实 DSL 开关对比**：对同一个 DSL 文件使用两份除 peephole_asm 外完全相同的 CompilerConfig 编译，比较关闭和开启窥孔优化后的汇编，并在执行环境可用时比较两者的可观察结果。

结果同时支持终端、JSON 和 Markdown 输出，用于验收、后续分析和未来 CI 回归。

### 1.2 现有能力与复用接口

| 现有接口 | 用途 |
|----------|------|
| AsmPeepholeOptimizer.optimize(asm_text) | 返回优化后的汇编和总变更次数 |
| AsmPeepholeOptimizer.total_matches | 获取最近一次优化中各规则的命中次数 |
| count_instructions(asm_text) | 按类别统计静态指令 |
| CompilerConfig.peephole_asm | 控制编译流程是否运行汇编级窥孔优化 |
| CompilerDriver.compile() | 编译 DSL 并返回 CompileResult |
| CompileResult.output_text | 获取生成的 RISC-V 汇编 |
| RISCVAEncoder / assemble_to_binary() | 将受支持的汇编编码为 RV32 二进制 |
| RV32Emulator / TinyFive | 在环境可用且指令受支持时执行汇编 |

total_matches 已提供逐规则统计，因此初始实现不要求修改 asm_peephole.py。

### 1.3 设计目标

- **可比性**：开启和关闭 peephole 的两次编译除开关外配置完全一致；
- **准确性**：以静态指令数为主要指标，不用汇编文本行数代替；
- **可复现性**：固定随机种子，记录重复次数、Python 版本和 Git commit；
- **可诊断性**：记录逐规则命中次数、失败阶段和错误信息；
- **可扩展性**：测试用例、报告格式和验证后端可以独立扩展；
- **不中断性**：单个用例失败后继续运行其他用例；
- **兼容性**：不改变编译器和窥孔优化器的默认行为。

### 1.4 非设计目标

本阶段不实现 CFG、Basic Block、liveness/def-use 和 move coalescing，不以新增大量窥孔规则为主要目标，也不修改 DSL、AST 或 IR。

---

## 二、接口与数据设计

### 2.1 命令行接口

#### 合成汇编微基准

~~~bash
python benchmarks/bench_asm_peephole.py \
    --repeats 20 \
    --sizes 100 500 1000 2000 5000 \
    --fusion-ratios 0.0 0.1 0.3 0.5 \
    --seed 42 \
    --json-output reports/peephole_benchmark.json \
    --markdown-output reports/peephole_benchmark.md
~~~

| 参数 | 默认值 | 约束 |
|------|--------|------|
| --repeats | 20 | 大于等于 1 |
| --sizes | 100、500、1000、2000、5000 | 每个值大于 0 |
| --fusion-ratios | 0.0、0.1、0.3、0.5 | 每个值位于 0 到 1 |
| --seed | 42 | 相同参数生成相同输入 |
| --json-output | 无 | 指定时生成 JSON |
| --markdown-output | 无 | 指定时生成 Markdown |

#### 真实 DSL 开关对比

~~~bash
python benchmarks/compare_peephole.py \
    --cases benchmarks/cases/001_simple_add.dsl \
            benchmarks/cases/011_multi_op_chain.dsl \
            benchmarks/cases/012_nn_pipeline.dsl \
    --repeats 5 \
    --verify auto \
    --json-output reports/peephole_compare.json \
    --markdown-output reports/peephole_compare.md
~~~

--verify 计划支持：

- auto：后端可用时验证，否则标记 unavailable；
- required：无法执行或结果不一致时判定用例失败；
- off：跳过运行时验证并标记 skipped。

以上参数是设计接口，实现时允许根据 argparse 习惯小幅调整，但字段含义保持不变。

### 2.2 数据结构

~~~python
@dataclass
class TimingStats:
    repeats: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stdev_ms: float


@dataclass
class PeepholeCaseResult:
    case_name: str
    input_type: str
    instructions_before: int
    instructions_after: int
    instructions_saved: int
    reduction_percent: float
    category_counts_before: dict[str, int]
    category_counts_after: dict[str, int]
    total_changes: int
    rule_matches: dict[str, int]
    timing: TimingStats
    verification_status: str
    output_equal: bool | None
    error_stage: str | None = None
    error: str | None = None


@dataclass
class PeepholeBenchmarkReport:
    schema_version: str
    generated_at: str
    environment: dict[str, str]
    cases: list[PeepholeCaseResult]
~~~

verification_status 取 passed、failed、unavailable 或 skipped。报告汇总计算总用例数、失败数、优化前后总指令数、总体减少比例和各规则总命中次数。

如果两个脚本间的共享代码较少，类型和格式化函数先放在脚本内部；重复明显后再提取为 benchmarks/peephole_benchmark_common.py，避免过早抽象。

### 2.3 指令计数

复用 count_instructions()。该函数返回六类计数和 _detailed 明细，总数计算必须排除以下划线开头的辅助字段：

~~~python
def instruction_total(counts: dict[str, object]) -> int:
    return sum(
        value
        for key, value in counts.items()
        if not key.startswith("_") and isinstance(value, int)
    )
~~~

计算公式：

~~~text
instructions_saved = instructions_before - instructions_after

reduction_percent =
    instructions_saved / instructions_before × 100%
~~~

当 instructions_before 为 0 时，reduction_percent 定义为 0.0。优化后指令数允许不变；如果增加，则保存负的 instructions_saved 并添加警告，不篡改结果。

### 2.4 耗时测量

耗时范围只包含 optimizer.optimize()，不包含输入生成、文件读写、DSL 解析、代码生成、指令计数和报告序列化：

~~~python
start = time.perf_counter_ns()
optimized_asm, changes = optimizer.optimize(asm_text)
elapsed_ns = time.perf_counter_ns() - start
~~~

每次重复都新建 AsmPeepholeOptimizer，避免 total_matches 状态残留。保存全部样本并计算 min、max、mean、median 和 stdev，主要展示 median。

多次运行的优化结果、changes 和 total_matches 必须一致；不一致时将用例标记为失败，说明优化过程存在非确定性。

### 2.5 合成汇编流程

保留并扩展现有 _gen_synthetic_asm()：

~~~text
sizes × fusion_ratios
    → 使用固定 seed 生成汇编
    → 重复运行 optimizer
    → 检查每次输出一致
    → count_instructions(before/after)
    → 收集 total_matches
    → 生成 PeepholeCaseResult
~~~

每个参数组合形成独立 case。fusion_ratio=0.0 作为低命中对照，较高比例用于观察匹配数量和耗时变化。

### 2.6 真实 DSL 开关对比

两份配置只改变 peephole_asm：

~~~python
common = {
    "backend": "riscv",
    "optimize_level": "none",
    "reg_alloc": "linear",
    "beautify_asm": False,
    "const_merge": False,
    "schedule": False,
    "count_instr": False,
}

off_config = CompilerConfig(**common, peephole_asm=False)
on_config = CompilerConfig(**common, peephole_asm=True)
~~~

流程如下：

~~~text
同一个 DSL 文件
    ├─ CompilerDriver(off_config).compile() → asm_off
    └─ CompilerDriver(on_config).compile()  → asm_on

asm_off → AsmPeepholeOptimizer.optimize() → replay_asm_on
                                     ├─ 测量优化器独立耗时
                                     └─ 取得 total_matches

检查 normalize(replay_asm_on) == normalize(asm_on)
    → 统计两份汇编
    → 可选执行验证
    → 写入结果
~~~

直接优化回放用于隔离测量耗时、获取编译器内部没有暴露的规则统计，并核对编译开关链路。两份结果归一化后不一致时，记录 pipeline_mismatch。

两次编译写入 tempfile.TemporaryDirectory() 中的不同输出文件，不覆盖用户文件。

### 2.7 汇编归一化

归一化只用于比较直接优化回放与编译器 ON 输出，不用于指令计数和语义判定：

~~~python
def normalize_asm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
~~~

归一化不删除指令、标签或注释，也不重排指令，避免掩盖真实差异。

### 2.8 执行等价验证

验证分为三层：

1. **结构检查**：OFF 和 ON 编译均成功，直接回放结果与 ON 输出一致；
2. **可执行性检查**：汇编可编码，且模拟器可以运行结束；
3. **语义检查**：比较用例声明的可观察状态，例如返回寄存器 a0、指定寄存器或内存区域。

仅“模拟器没有报错”不能证明语义等价。只有两侧可观察状态一致时才能记为 passed。

| 状态 | 含义 |
|------|------|
| passed | 两侧成功执行且可观察状态一致 |
| failed | 编码失败、执行失败、超限或状态不一致 |
| unavailable | 没有执行后端或存在不支持的指令 |
| skipped | 用户主动关闭验证 |

初始阶段优先为人工构造的小型汇编定义明确的返回寄存器。真实 DSL 如果缺少稳定输入初始化或输出定位方式，应标记 unavailable，不能错误标记为通过。

### 2.9 报告输出

终端报告展示：

~~~text
Case | Type | Before | After | Saved | Reduction | Median(ms) | Verify
~~~

JSON 使用固定 schema_version，保存完整字段；Markdown 保存环境信息、汇总表、逐规则统计和失败详情。输出目录不存在时创建，报告内容完整生成后再写入，避免异常时留下半份报告。

---

## 三、测试设计

### 3.1 单元测试

计划新增 tests/test_asm_peephole_benchmark.py，至少覆盖：

1. instruction_total() 能排除 _detailed；
2. 空汇编的总指令数和减少比例；
3. 标签、注释、空行和汇编指示符不计为指令；
4. 优化前后减少数和百分比计算；
5. 指令数增加时保留负数结果；
6. 单次运行时 stdev 为 0；
7. 多次优化的输出、changes 和 matches 一致；
8. 规则命中次数来自 optimizer.total_matches；
9. 不合法的 repeats、size 和 fusion ratio 被拒绝；
10. JSON 能被重新读取；
11. Markdown 包含汇总表和失败详情；
12. 单个 case 异常不影响其他 case；
13. 相同 seed 生成相同合成汇编；
14. normalize_asm() 只忽略首尾空白和行尾空格。

### 3.2 合成汇编测试

#### 用例 1：高命中序列

~~~asm
.text
main:
    addi t0, t0, 1
    addi t0, t0, 2
    li t1, 5
    addi t1, t1, 3
    ret
~~~

验证点：

- 至少命中 addi+addi fusion 和 li+addi fusion；
- 优化后指令数小于优化前；
- 多次运行的输出、changes 和 total_matches 一致。

#### 用例 2：低命中对照

~~~asm
.text
main:
    add a0, a1, a2
    sub a3, a0, a4
    mul a5, a3, a6
    ret
~~~

验证点：

- 没有匹配时 changes 为 0；
- 优化前后静态指令数相同；
- Benchmark 正常完成，不把“无收益”当作错误。

#### 用例 3：空输入和非指令行

~~~asm
.text
.globl main
main:
    # comment only
~~~

验证点：

- 静态指令总数为 0；
- 减少比例为 0.0；
- 不出现除零异常。

### 3.3 真实 DSL 集成测试

首批选择仓库已有用例：

| 用例 | 作用 |
|------|------|
| benchmarks/cases/001_simple_add.dsl | 最小 DSL → IR → codegen → peephole 链路 |
| benchmarks/cases/011_multi_op_chain.dsl | 连续运算产生的较长汇编序列 |
| benchmarks/cases/012_nn_pipeline.dsl | 更接近真实模型流水线的组合场景 |

每个用例验证：

1. OFF 和 ON 两次编译均成功；
2. 两份配置只有 peephole_asm 不同；
3. replay_asm_on 与 asm_on 归一化后一致；
4. 指令统计字段齐全；
5. 没有优化收益时仍保留该用例；
6. 执行环境不可用时状态为 unavailable；
7. 单个用例失败时其他用例仍继续。

### 3.4 执行等价测试

对现有编码器和模拟器支持的小型汇编：

1. 分别编码优化前和优化后汇编；
2. 使用相同寄存器和内存初始状态；
3. 设置相同最大执行指令数；
4. 运行至 ret 或正常停止；
5. 比较 a0 和用例声明的其他可观察状态；
6. 任一侧超限、异常或状态不同均判定失败。

TinyFive 未安装、编码器不支持某条指令或真实 DSL 缺少稳定输出约定时，记录原因并标记 unavailable。不使用 DSL 解释器结果代替优化前后汇编的执行比较。

### 3.5 验收对应关系

| 开发文档验收项 | 设计中的验证方式 |
|----------------|------------------|
| 完善 bench_asm_peephole.py | 合成规模、冗余比例、重复测量和参数校验 |
| 新增 compare_peephole.py | 三个真实 DSL 用例的 OFF/ON 集成测试 |
| 指令数及减少比例 | 复用 count_instructions() 并测试公式和零输入 |
| 优化耗时 | perf_counter_ns() 多次测量并输出统计量 |
| 规则命中 | 读取 total_matches 并汇总 |
| 结果一致性 | 编译链路核对及可选模拟器状态比较 |
| JSON / Markdown | 序列化测试和字段完整性检查 |
| 不引入回归 | 运行现有 peephole、inst_counter 和项目测试 |

---

## 四、修改模块与实现步骤

### 4.1 benchmarks/bench_asm_peephole.py

保留现有 _gen_synthetic_asm() 和 bench_optimize() 的基本职责，进行以下调整：

1. 使用 count_instructions() 替代主要的行数统计；
2. 每次重复新建优化器并保存耗时、输出、changes 和 total_matches；
3. 检查重复结果是否确定；
4. 将秒统一转换为毫秒；
5. 支持 sizes、fusion ratios、seed 和报告路径参数；
6. 生成统一的 case 结果；
7. 保留原终端表格，使不带新参数的旧用法仍可运行。

建议函数：

~~~python
def validate_benchmark_args(...) -> None: ...
def instruction_total(counts: dict[str, object]) -> int: ...
def collect_timing(samples_ns: list[int]) -> TimingStats: ...
def benchmark_assembly(
    case_name: str,
    asm_text: str,
    repeats: int,
) -> PeepholeCaseResult: ...
def render_markdown(report: PeepholeBenchmarkReport) -> str: ...
def write_json(report: PeepholeBenchmarkReport, path: Path) -> None: ...
~~~

### 4.2 benchmarks/compare_peephole.py

新增脚本，负责真实 DSL 对比和批量执行。

~~~python
def make_compiler_config(peephole_enabled: bool) -> CompilerConfig:
    ...

def compile_dsl_pair(
    case_path: Path,
    temp_dir: Path,
) -> tuple[CompileResult, CompileResult]:
    ...

def compare_dsl_case(
    case_path: Path,
    repeats: int,
    verify_mode: str,
) -> PeepholeCaseResult:
    ...

def run_compare_suite(
    case_paths: list[Path],
    repeats: int,
    verify_mode: str,
) -> PeepholeBenchmarkReport:
    ...

def main(argv: list[str] | None = None) -> int:
    ...
~~~

实现要求：

- 使用 dataclasses.replace() 或显式复制配置，防止其他配置漂移；
- OFF 和 ON 使用不同临时输出文件；
- 检查 CompileResult.success 后再读取 output_text；
- 编译失败时记录 compile_off 或 compile_on；
- 对 baseline 重复直接优化，用于耗时和规则统计；
- 比较直接回放结果与 ON 输出，不一致时记录 pipeline_mismatch；
- 每个用例独立捕获异常，继续执行后续用例；
- main() 根据失败数返回非零状态，便于未来 CI 使用。

### 4.3 报告与共享逻辑

初始实现可将结果类型和报告函数保留在脚本中。如果重复明显，再新增 benchmarks/peephole_benchmark_common.py，仅包含：

- dataclass；
- 指令总数和百分比计算；
- 环境信息采集；
- JSON / Markdown 序列化；
- 验证状态常量。

编译器业务逻辑不放入共享模块。

### 4.4 scratchv/backend/asm_peephole.py

默认不修改。当前 optimize() 已返回总 changes，total_matches 已返回逐规则计数。

如确需扩展，只允许新增向后兼容的只读统计接口，不能改变 optimize(asm_text) 返回 tuple[str, int] 的结构。

### 4.5 scratchv/backend/inst_counter.py

默认不修改，直接复用 count_instructions()。如果测试发现计数错误，应先补充 inst_counter 的独立回归测试，再做最小修复，不能在 Benchmark 内维护另一套解析规则。

### 4.6 实现顺序

1. 实现参数校验、总指令计算和统一结果结构；
2. 将现有微基准扩展为确定性检查、规则命中和指令统计；
3. 实现终端、JSON 和 Markdown 报告；
4. 新增 compare_peephole.py，完成 OFF/ON 编译和静态对比；
5. 增加直接优化回放和 pipeline mismatch 检查；
6. 接入三个真实 DSL 用例；
7. 为小型汇编加入可观察状态比较；
8. 添加单元测试和集成测试；
9. 运行现有 peephole、inst_counter 和项目回归测试；
10. 保存一次可复现报告作为验收材料。

### 4.7 错误处理

| 阶段 | error_stage | 处理 |
|------|-------------|------|
| 参数校验 | arguments | CLI 输出原因并返回 2 |
| 读取输入 | read_input | 记录文件和异常，继续其他用例 |
| OFF 编译 | compile_off | 不执行后续比较 |
| ON 编译 | compile_on | 保留 OFF 信息，记录失败 |
| 优化回放 | optimize | 记录异常和 baseline 统计 |
| 链路核对 | pipeline_mismatch | 标记失败并保留两份结果 |
| 指令统计 | count | 标记失败，不输出虚假指标 |
| 执行验证 | verify | 按 verify 模式记 failed 或 unavailable |
| 报告写入 | report | CLI 返回非零，不覆盖已有有效报告 |

---

## 五、附录

### 5.1 JSON 示例

~~~json
{
  "schema_version": "1.0",
  "generated_at": "2026-08-05T20:00:00+08:00",
  "environment": {
    "python_version": "3.x",
    "platform": "Linux",
    "git_commit": "..."
  },
  "summary": {
    "total_cases": 3,
    "failed_cases": 0,
    "instructions_before": 240,
    "instructions_after": 220,
    "instructions_saved": 20,
    "reduction_percent": 8.3333
  },
  "cases": [
    {
      "case_name": "001_simple_add",
      "input_type": "dsl",
      "instructions_before": 20,
      "instructions_after": 18,
      "instructions_saved": 2,
      "reduction_percent": 10.0,
      "total_changes": 2,
      "rule_matches": {
        "addi+addi fusion": 2
      },
      "verification_status": "passed",
      "output_equal": true,
      "error_stage": null,
      "error": null
    }
  ]
}
~~~

示例数值仅说明数据结构，不代表实际 Benchmark 结果。

### 5.2 验收运行命令

~~~bash
python benchmarks/bench_asm_peephole.py --repeats 20
~~~

~~~bash
python benchmarks/compare_peephole.py \
    --cases benchmarks/cases/001_simple_add.dsl \
            benchmarks/cases/011_multi_op_chain.dsl \
            benchmarks/cases/012_nn_pipeline.dsl \
    --verify auto \
    --json-output reports/peephole_compare.json \
    --markdown-output reports/peephole_compare.md
~~~

计划测试命令：

~~~bash
python -m pytest tests/test_asm_peephole.py \
                  tests/test_inst_counter.py \
                  tests/test_asm_peephole_benchmark.py
~~~

### 5.3 关键设计决策

1. 使用静态指令数作为主要代码规模指标，行数只作辅助；
2. 真实 DSL 使用实际编译开关进行 OFF/ON 对比；
3. 直接优化 baseline 只用于隔离耗时、获取规则统计和核对链路；
4. 运行成功不等于语义等价，必须比较明确的可观察状态；
5. 执行环境不可用时如实标记 unavailable；
6. 默认不修改 optimizer 和 inst_counter 的公开接口；
7. 报告保留零收益和负收益用例，确保数据真实。

### 5.4 参考资料

- docs/feat13#suai/开发文档.md
- benchmarks/bench_asm_peephole.py
- benchmarks/bench_runner.py
- scratchv/compiler.py
- scratchv/backend/asm_peephole.py
- scratchv/backend/inst_counter.py
- scratchv/backend/riscv_encoder.py
- scratchv/simulator/rv32_emulator.py
- scratchv/simulator/tinyfive.py
- [ScratchV 课题 13：窥孔优化器](https://scratchv-compiler.github.io/ScratchV/docs/13-%E7%AA%A5%E5%AD%94%E4%BC%98%E5%8C%96%E5%99%A8.html)
- [ScratchV PR #39](https://github.com/ScratchV-Compiler/ScratchV/pull/39)
