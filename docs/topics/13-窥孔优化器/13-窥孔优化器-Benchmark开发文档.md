# ScratchV 窥孔优化器 Benchmark 功能开发文档

> **文档版本**：v1.0  
> **创建日期**：2026-08-05  
> **作者**：suai  
> **课题编号**：13（窥孔优化器）  
> **关联 Issue / PR**：待补充  
> **涉及模块**：`benchmarks/`、`scratchv/backend/asm_peephole.py`、`scratchv/backend/inst_counter.py`

---

## 1. 功能概述与目标

### 1.1 背景与动机

- **现状问题**：

  ScratchV 已具备基础的 RISC-V 汇编窥孔优化能力，可以通过匹配和替换局部低效指令序列减少冗余指令。现有 `benchmarks/bench_asm_peephole.py` 能够生成合成汇编，并初步测量优化耗时和代码行数变化，但目前还缺少统一、完整且可复现的 Benchmark，主要问题如下：

  1. 测试输入以合成汇编为主，缺少真实 DSL 程序编译得到的汇编；
  2. 缺少统一的 peephole 开启与关闭对比流程；
  3. 缺少明确的统计口径，例如优化前后静态指令数、减少指令数和减少比例；
  4. 缺少不同优化规则命中次数等诊断信息；
  5. 缺少便于人工查看和后续处理的结构化报告；
  6. 缺少优化前后程序执行结果一致性的验证；
  7. 当前结果不便作为后续优化开发和性能回归检查的基线。

- **应用场景**：

  1. 测量优化器处理不同规模汇编程序时的运行时间；
  2. 对比开启和关闭 peephole 时生成代码的静态指令数；
  3. 统计不同测试用例中的指令减少数量和减少比例；
  4. 观察不同窥孔优化规则的命中情况；
  5. 检查优化前后程序的执行结果是否一致；
  6. 对比合成汇编与真实 DSL 编译结果中的优化收益；
  7. 为后续优化器改进和 CI 性能回归测试提供数据基础。

### 1.2 功能描述

- **一句话定义**：建立一套可重复运行的 ScratchV 窥孔优化器 Benchmark，对比 peephole 开启和关闭时的代码规模、优化耗时、规则命中情况和程序执行结果。

- **核心价值**：

  1. 用量化数据说明窥孔优化器是否真正减少了生成的 RISC-V 指令；
  2. 评估优化器在不同规模输入下的运行开销；
  3. 验证优化不会改变原程序的执行结果；
  4. 为发现无效规则、异常优化和性能退化提供依据；
  5. 为后续优化器改进建立统一测试基线。

### 1.3 目标与非目标

| 类型 | 内容 |
|------|------|
| ✅ 包含范围 | 完善 `benchmarks/bench_asm_peephole.py`，支持不同规模和不同冗余比例的汇编测试输入 |
| ✅ 包含范围 | 新增 `benchmarks/compare_peephole.py`，统一比较 peephole 开启和关闭时的结果 |
| ✅ 包含范围 | 统计优化前后静态指令数、减少指令数和减少比例 |
| ✅ 包含范围 | 统计优化器运行时间，并支持多次运行后汇总结果 |
| ✅ 包含范围 | 在接口允许的情况下统计不同优化规则的命中次数 |
| ✅ 包含范围 | 覆盖人工构造汇编、自动生成汇编和真实 DSL 编译用例 |
| ✅ 包含范围 | 条件允许时，通过模拟器验证优化前后的执行结果一致 |
| ✅ 包含范围 | 支持输出 JSON 和 Markdown 格式的 Benchmark 报告 |
| ❌ 不包含范围 | 本阶段不实现 CFG、Basic Block、liveness 或 def-use 分析 |
| ❌ 不包含范围 | 本阶段不实现复杂的 move coalescing |
| ❌ 不包含范围 | 本阶段不以增加大量窥孔优化规则为主要目标 |
| ❌ 不包含范围 | 本阶段不修改 ScratchV DSL 语法、AST 或 IR 定义 |

---

## 2. 设计与规格说明

### 2.1 用户视角（外部接口）

本功能属于 Benchmark 工具，不新增或修改 DSL 语法。计划提供以下两个主要脚本。

#### 窥孔优化器性能基准脚本

```bash
python benchmarks/bench_asm_peephole.py
```

计划支持的参数示例：

```bash
python benchmarks/bench_asm_peephole.py \
    --repeats 5 \
    --json-output reports/peephole_benchmark.json \
    --markdown-output reports/peephole_benchmark.md
```

该脚本负责生成或加载不同规模的汇编，多次运行窥孔优化器，测量优化耗时并汇总优化前后的指令数量。

#### peephole 开关对比脚本

```bash
python benchmarks/compare_peephole.py \
    --cases benchmarks/cases \
    --json-output reports/peephole_compare.json \
    --markdown-output reports/peephole_compare.md
```

该脚本负责使用相同输入获得未优化和优化后的汇编，比较代码规模和优化收益，并在执行环境可用时验证两者运行结果。以上参数是初步规划，实际接口将在设计和实现阶段结合现有代码确定。

### 2.2 内部设计（核心逻辑）

- **数据结构变更**：

  本功能不修改 AST 和 IR 数据结构。单个 Benchmark 结果计划记录：

  ```python
  {
      "case_name": "001_simple_add",
      "input_type": "dsl",
      "instructions_before": 20,
      "instructions_after": 16,
      "instructions_saved": 4,
      "reduction_percent": 20.0,
      "optimization_time_ms": 0.52,
      "rule_matches": {},
      "verification_status": "passed",
      "output_equal": True,
      "error": None
  }
  ```

- **关键流程**：

  ```text
  读取 DSL、汇编文件或生成合成汇编
      → 获得未开启 peephole 时的汇编
      → 获得开启 peephole 后的汇编
      → 统计优化前后的静态指令数
      → 统计优化耗时和规则命中次数
      → 计算减少指令数和减少比例
      → 条件允许时分别执行两份汇编
      → 比较程序输出或最终状态
      → 汇总并输出报告
  ```

- **指标定义**：

  1. `instructions_before`：未开启 peephole 时的静态汇编指令数；
  2. `instructions_after`：开启 peephole 后的静态汇编指令数；
  3. `instructions_saved = instructions_before - instructions_after`；
  4. `reduction_percent = instructions_saved / instructions_before × 100%`；
  5. `optimization_time_ms`：仅统计窥孔优化器处理汇编代码所用的时间；
  6. `rule_matches`：不同优化规则的命中次数；
  7. `output_equal`：优化前后程序执行结果是否一致。

- **计数和重复运行规则**：

  静态指令计数应排除空行、注释、标签及 `.text`、`.data`、`.globl` 等汇编指示符。同一用例默认运行 5 次，记录中位数，并可同时记录最小值、最大值和标准差。自动生成测试数据时使用固定随机种子，报告中记录运行日期、Python 版本和 Git commit。

- **状态管理**：

  Benchmark 不新增全局编译状态。每个用例独立运行，测试配置、统计结果和错误信息由脚本内部对象保存，避免不同用例相互影响。

### 2.3 接口定义（模块间交互）

- **上游依赖**：

  1. DSL 编译入口能够接收相同的 DSL 输入；
  2. 后端代码生成流程能够控制是否启用 peephole；
  3. `asm_peephole.py` 能够接收汇编并返回优化后的汇编；
  4. `inst_counter.py` 或等价逻辑能够提供静态指令计数；
  5. `benchmarks/cases/` 提供可重复使用的真实 DSL 输入。

- **下游影响**：

  1. 不修改 DSL 解析和 IR 生成结果；
  2. 不改变窥孔优化器默认行为；
  3. Benchmark 不参与正常编译流程；
  4. 生成的报告可用于人工评审或后续 CI；
  5. 如果缺少规则命中统计接口，只进行不改变优化语义的小范围扩展。

---

## 3. 模块修改与实现步骤

### 3.1 涉及的文件清单

| 文件路径 | 修改类型 | 修改内容概述 |
|----------|----------|--------------|
| `benchmarks/bench_asm_peephole.py` | 修改 | 完善输入生成、重复测量、指令统计和结果汇总 |
| `benchmarks/compare_peephole.py` | 新增 | 对比 peephole 开关前后的代码规模、耗时和执行结果 |
| `tests/test_asm_peephole_benchmark.py` | 新增 | 测试指令计数、指标计算和报告生成逻辑 |
| `scratchv/backend/asm_peephole.py` | 可选修改 | 必要时增加规则命中统计或诊断接口 |
| `scratchv/backend/inst_counter.py` | 可选修改 | 复用或完善静态指令计数逻辑 |
| `benchmarks/cases/` | 复用或补充 | 提供真实 DSL Benchmark 输入 |
| `reports/` | 运行时生成 | 保存 JSON 和 Markdown 报告 |

### 3.2 分步实现计划

| 步骤 | 任务描述 | 预期产出 | 验证方式 |
|------|----------|----------|----------|
| 1 | 阅读现有 Benchmark、优化器、指令计数器和编译入口 | 明确可复用接口和当前不足 | 能说明现有流程和限制 |
| 2 | 定义统一指标和结果结构 | 确定指令数、耗时和减少比例等口径 | 使用小型汇编人工核对 |
| 3 | 完善静态指令计数逻辑 | 正确排除标签、注释和汇编指示符 | 单元测试 |
| 4 | 完善 `bench_asm_peephole.py` | 支持不同规模输入、多次运行和结果汇总 | 运行脚本检查输出 |
| 5 | 实现 `compare_peephole.py` | 使用相同输入比较 peephole 开关结果 | 检查两次运行配置 |
| 6 | 接入人工构造、自动生成和真实 DSL 用例 | 至少覆盖三类输入 | 检查各类用例均进入报告 |
| 7 | 增加结果一致性验证 | 能发现优化前后运行结果不一致 | 使用可用执行入口验证 |
| 8 | 增加 JSON 和 Markdown 报告 | 结果可保存和追溯 | 检查报告字段与汇总 |
| 9 | 添加单元测试和回归测试 | Benchmark 相关测试稳定通过 | 运行项目测试命令 |
| 10 | 整理使用说明和示例命令 | 其他开发者能够独立复现 | 按文档运行完整流程 |

### 3.3 异常处理与边界条件

- [ ] 空输入的指令数为 0，且不出现除零错误；
- [ ] 输入文件不存在时给出清晰错误信息；
- [ ] DSL 编译失败时记录失败用例和错误原因；
- [ ] 没有规则命中时正常输出 0；
- [ ] 优化后指令数不变或增加时如实记录并标记；
- [ ] 静态计数不把标签、注释、空行和汇编指示符计为指令；
- [ ] 重复次数小于 1 时拒绝运行并给出提示；
- [ ] 单个用例失败时不中断其他用例；
- [ ] 自动生成输入时固定随机种子；
- [ ] 模拟器不可用时将语义验证标记为“未验证”；
- [ ] 优化前后结果不一致时保存输入及两份汇编，便于定位问题。

---

## 4. 测试与验证方案

### 4.1 单元测试

- **计划测试文件位置**：`tests/test_asm_peephole_benchmark.py`
- **至少覆盖以下场景**：

  1. 空输入和普通 RISC-V 指令的计数；
  2. 排除空行、注释、标签和汇编指示符；
  3. 减少指令数及减少比例的计算；
  4. 优化前指令数为 0 时的处理；
  5. 没有规则命中的情况；
  6. 优化后指令数增加的异常情况；
  7. 多次运行后的耗时统计；
  8. JSON 和 Markdown 报告生成；
  9. 单个用例失败后继续执行其他用例；
  10. 固定随机种子后生成相同输入。

### 4.2 集成 / 基准测试

- **DSL 用例位置**：`benchmarks/cases/`
- **至少覆盖三类用例**：

  1. **高冗余人工汇编**：包含能够被现有规则匹配的指令序列，确认 Benchmark 可以测量到明确的指令减少；
  2. **无明显优化机会的对照汇编**：验证规则命中为 0 时仍能生成正确报告；
  3. **真实 DSL 编译用例**：选择 `001_simple_add.dsl`、`011_multi_op_chain.dsl`、`012_nn_pipeline.dsl` 等现有用例，比较 peephole 开关前后的生成结果。

- **结果表格示例**：

  | 用例 | 类型 | 优化前指令数 | 优化后指令数 | 减少数 | 减少比例 | 执行验证 |
  |------|------|-------------:|-------------:|-------:|---------:|----------|
  | synthetic-small | ASM | 100 | 85 | 15 | 15.00% | 一致 |
  | control-case | ASM | 50 | 50 | 0 | 0.00% | 一致 |
  | real-dsl-case | DSL | 240 | 225 | 15 | 6.25% | 一致 |

  表中数据仅用于说明报告格式，最终结果以实际运行数据为准。

### 4.3 验收标准（Definition of Done）

- [ ] `bench_asm_peephole.py` 和 `compare_peephole.py` 能独立运行；
- [ ] 能使用相同输入比较 peephole 开启和关闭的结果；
- [ ] 能统计优化前后静态指令数、减少指令数和减少比例；
- [ ] 能统计优化器运行时间；
- [ ] 在接口允许时能展示规则命中次数；
- [ ] 至少覆盖人工汇编、自动生成汇编和真实 DSL 三类输入；
- [ ] 至少包含一个没有优化收益的对照用例；
- [ ] 能输出 JSON 以及 Markdown 或终端表格报告；
- [ ] 条件允许时能验证优化前后执行结果一致；
- [ ] 无法进行执行验证时明确标记为“未验证”；
- [ ] 新增测试和项目现有测试通过，不引入回归；
- [ ] 报告记录运行环境和 Git commit，保证结果可追溯。

---

## 5. 风险评估与依赖

| 风险项 | 影响程度 | 缓解措施 |
|--------|----------|----------|
| 合成数据不能代表真实程序 | 中 | 同时加入真实 DSL 用例，分开展示两类结果 |
| 使用汇编行数代替指令数 | 高 | 排除标签、注释和指示符，主要报告静态指令数 |
| 开关前后的编译配置不一致 | 高 | 两次编译只改变 peephole 开关 |
| 优化后汇编语义发生变化 | 高 | 尽可能通过模拟器比较输出或最终状态 |
| 模拟器或工具链不可用 | 中 | 静态指标与执行验证分开，无法验证时明确标记 |
| 优化耗时受系统负载影响 | 中 | 多次运行并使用中位数 |
| 缺少规则命中统计接口 | 中 | 增加不改变优化语义的轻量统计接口 |
| 单个错误中断全部测试 | 中 | 每个用例独立捕获异常并继续运行 |
| 报告难以复现 | 中 | 固定随机种子并记录环境和 Git commit |

- **外部依赖**：

  基础统计和报告优先使用 Python 标准库及项目现有模块，不新增强制第三方依赖。真实汇编执行验证可能依赖项目现有模拟器、TinyFive 或 RISC-V 工具链；这些环境不可用时，不影响基础 Benchmark 运行。

- **对现有功能的兼容性**：

  不修改 DSL 语法、AST 和 IR，不改变窥孔优化器默认行为。新增统计接口应保持现有调用方式兼容，Benchmark 脚本不进入正常编译流程。

---

## 6. 开发进度跟踪

| 阶段 | 计划完成日期 | 状态 |
|------|--------------|------|
| 项目与现有代码阅读 | 2026-08-05 | 已完成 |
| 开发文档编写 | 2026-08-05 | 进行中 |
| 设计文档编写与评审 | 待确定 | 待开始 |
| 完善 `bench_asm_peephole.py` | 待确定 | 待开始 |
| 实现 `compare_peephole.py` | 待确定 | 待开始 |
| 添加测试和结果验证 | 待确定 | 待开始 |
| 生成 Benchmark 报告 | 待确定 | 待开始 |
| 代码审查与提交 PR | 待确定 | 待开始 |
| 合并主分支 | 待确定 | 待开始 |

> 具体日期根据课题安排和验收时间调整。文档经过评审后，再将“开发文档编写”更新为“已完成”。

---

## 7. 附录

### 7.1 参考资料

- [ScratchV 课题 13：窥孔优化器](https://scratchv-compiler.github.io/ScratchV/docs/13-%E7%AA%A5%E5%AD%94%E4%BC%98%E5%8C%96%E5%99%A8.html)
- [ScratchV PR #39](https://github.com/ScratchV-Compiler/ScratchV/pull/39)
- [LLVM Code Generator：Peephole Optimizations](https://llvm.org/docs/CodeGenerator.html#peephole-optimizations)
- [RISC-V Instruction Set Manual](https://riscv.org/technical/specifications/)
- `benchmarks/bench_asm_peephole.py`
- `scratchv/backend/asm_peephole.py`
- `scratchv/backend/inst_counter.py`
- `benchmarks/cases/` 中的现有 DSL 用例

### 7.2 测试用例详情

#### 用例 1：存在明显冗余的汇编

```asm
.text
main:
    mv a0, a0
    addi a1, a1, 0
    addi a2, a2, 1
    addi a2, a2, 2
    ret
```

预期能够统计优化前后的实际指令数；冗余指令在满足规则和安全条件时被删除或合并；优化后的指令数减少，且执行结果保持一致。

#### 用例 2：没有明显优化机会的对照汇编

```asm
.text
main:
    add a0, a1, a2
    sub a3, a0, a4
    mul a5, a3, a6
    ret
```

预期规则命中次数可以为 0，优化前后指令数可以相同，但 Benchmark 仍应生成完整报告。

#### 用例 3：真实 DSL 编译用例

从以下现有用例中选择能够完整编译的程序：

```text
benchmarks/cases/001_simple_add.dsl
benchmarks/cases/011_multi_op_chain.dsl
benchmarks/cases/012_nn_pipeline.dsl
```

测试流程：

```text
DSL 输入
  → 生成 IR
  → 分别生成关闭和开启 peephole 的 RISC-V 汇编
  → 对比静态指令数
  → 条件允许时分别在模拟器中执行
  → 对比程序输出或最终状态
```

如果没有获得指令减少，应如实记录；如果模拟器不可用，应将执行验证标记为“未验证”。

### 7.3 文档说明

本开发文档用于明确 ScratchV 窥孔优化器 Benchmark 专项需要解决的问题、计划交付的功能、测试方法和验收标准。具体函数拆分、命令行参数、数据对象实现方式和编译流程接入方式，将在后续设计文档中结合实际代码进一步确定。
