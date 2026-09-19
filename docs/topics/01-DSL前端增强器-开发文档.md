# 课题 01：DSL 前端增强器开发文档

> 状态：开发与验收计划，尚未实施。配套：[设计文档](01-DSL前端增强器-设计文档.md)。
> 基线：`99538fe5059599b337d4ab3da81be5406c29d8c4`，2026-09-18。

## 1. 开发原则与环境

先固定行为测试，再修改实现；每个阶段保持既有回归通过。正确性失败不转成 warning，不通过修改 expected、skip、xfail 或更换 Stub 掩盖。旧测试若依赖错误语义，保留原输入作为负例，另补正确初始化的正例并在 PR 中说明。

按当前 CI 使用 Linux + Python 3.12。项目元数据仍声明更早 Python 版本，本计划不更改其兼容性声明。以下命令在仓库根目录执行；`python` 须指向已激活的 Python 3.12 虚拟环境。

```bash
python -m venv .venv-topic01
source .venv-topic01/bin/activate
python -m pip install -e ".[all]" "pytest>=7,<10" markdown
python -m pytest tests/ -q --tb=short --ignore=tests/test_simulator.py
python -m pytest tests/test_simulator.py::TestStubProfiledMachine -q
python -m pytest benchmarks/test_benchmark.py -q
```

Stub 测试仅验证模拟器包装接口，不能验收控制流结果。开发分支从更新后的 upstream/main 建立；不要把课题 09 的旧分支作为本课题基线。阅读顺序：课题说明 → dsl_parser/dsl_extended → validator/errors → IR builder/types → instruction_select/llvm_codegen → 现有测试和 ci.yml。

## 2. 文件与职责

所有“拟新增”路径仅为后续实现计划，本次不创建空模块或占位测试。

| 文件 | 修改职责 |
| --- | --- |
| `scratchv/frontend/dsl_ast.py`（拟新增） | 携带源码范围的轻量语句节点 |
| `scratchv/frontend/dsl_lowering.py`（拟新增） | 变量分析、槽位、块终结及 AST 到 IR |
| `scratchv/frontend/dsl_extended.py` | 保留公开 API，协调结构解析和 lowering |
| `scratchv/frontend/dsl_validator.py` | 共享条件/块识别，保留诊断上限及位置 |
| `scratchv/frontend/dsl_parser.py` | 仅必要的公共算子构建复用，不重写基础 DSL |
| `scratchv/ir/builder.py` | 明确 br_compare 与标量槽位接口 |
| `scratchv/backend/instruction_select.py`、`llvm_codegen.py` | 比较双形式、标签和槽位降级契约 |
| `scratchv/compiler.py` | 必要的集成与不支持组合诊断，保持 CLI API |
| `tests/test_dsl_control_flow.py`（拟新增） | 结构、变量路径及后端契约回归 |
| `tests/test_dsl_control_flow_execution.py`（拟新增） | LLVM/真实 TinyFive 执行与超时 |
| `examples/topic01/`（拟新增） | 三个可终止、自包含的完整示例 |
| `benchmarks/bench_dsl_frontend.py`（拟新增） | 本课题专用正确性/规模报告 |
| `.github/workflows/ci.yml` | 仅在实现阶段接入原有 jobs |

涉及 IRVerifier、优化器或寄存器分配的修复须有最小失败用例，独立说明必要性；不借本课题重写其他负责人的模块。未合并的 PR 只用于协作参考，不作为已经可用的依赖。

## 3. 分阶段实施

### P0：锁定基线和失败证据

- [ ] 运行第 1 节命令，记录提交、环境、通过数及失败原因。
- [ ] 打印设计文档中三次递增 while 的 IR，确认循环头与 return 的值来源。
- [ ] 对 if 的 true/false 输入检查生成的比较和目标，不只统计块数。
- [ ] 检查 `tests/test_dsl_extended.py`、`test_parser.py`、`test_dsl_validator.py` 及 CFG/LLVM 测试，登记需要保留的兼容形式。

输出：基线记录及最小重现输入。不要将当前后端输出保存为正确性金标准，也不要先提交使 CI 持续失败的测试；失败证据与对应修复在同一实现提交中交付。

### P1：统一结构解析与 AST

- [ ] 为六种比较、空块、三层嵌套、省略冒号、CRLF/注释和结束符错配增加参数化测试。
- [ ] 创建位置节点；条件解析只接受两个原子操作数和一个比较符。
- [ ] 将结束关键字的消费交给所属块；AST 完成前不修改 IRBuilder。
- [ ] 让 validate 和 parse 使用同一结构判定，继续返回 ErrorCollector/抛出首个 DSLSyntaxError。
- [ ] 验证同一 parser 连续解析、失败后重试与新的 parser 结果一致。

验收：现有合法基础用例保留；非法条件不成为隐式变量；源码行列和课题 09 测试不回退。入口签名不变。本阶段的结构重构不声称修复执行语义。

### P2：先通过比较、标签和槽位契约测试

- [ ] 手工构造单条件与比较形式 BR_IF，检查六种比较的 true/false 两路。
- [ ] 增加 builder 的 br_compare；更新两种后端，拒绝错误操作数数量及未知 cmp_op。
- [ ] LLVM 生成结果必须经过 `llvmlite.binding.parse_assembly(...).verify()`；FLOAT32 不能直接用于 `br i1`。
- [ ] 统一 RISC-V 目标标签与定义，编码器不得留下未解析的符号。
- [ ] 建立两个及三个标量槽位的交错写入/读取用例，断言互不覆盖，并检查对齐和栈恢复。
- [ ] 明确旧 ALLOCA 与新标量槽位封装的大小单位，分别测试 LLVM 和 RISC-V；不能靠同一个数字参数猜测两端含义。

验收：LLVM 类型验证、整数值子集的真实 TinyFive 输出、旧 BR_IF 回归全部通过。不满足则停留在本阶段修契约，不能直接接入 AST lowering。FLOAT32 RISC-V 不支持时必须显式失败，并保留能力测试。

### P3：if/else 的变量与终结语义

- [ ] 先写 then/else 各自赋值、入口旧值、分支首次定义后使用以及块内 return 的失败测试。
- [ ] then 和 else 从同一入口环境分析；跨分支变量读写使用已验证的槽位接口。
- [ ] 单分支写入保留入口旧值；缺少全路径定义时输出诊断。
- [ ] RETURN 后不追加跳转；两条分支均终结时不生成可达的空汇合块。
- [ ] 测试中间 IR、LLVM verify、true/false 实际返回值及编译 CLI。

验收：不仅要求出现 BR_IF，还要求执行相反条件时确实走不同分支，输出不受源码中 else 解析顺序影响。

### P4：while、嵌套和旧 for 兼容

- [ ] 写零次、一次、五次循环及多个累加器用例，先观察失败。
- [ ] 为循环读写变量在入口初始化槽位；header 每次读取当前值，body 更新后回 header。
- [ ] 处理零次路径、循环体 return、if-in-while、while-in-if、while-in-while。
- [ ] 验证 for 的零次/多次执行及与 if/while 混合嵌套。必要的 FOR/ENDFOR 降级调整必须附真实结果测试。
- [ ] 对故意不更新条件的无限循环运行超时测试，确认报告 TIMEOUT 且退出非零。

验收：三个示例结果正确；所有支持的嵌套均可终止；标签唯一且所有目标存在；无循环中重复分配槽位和零次路径未定义读。

### P5：编译管线、benchmark 与 CI

- [ ] 用优化关闭与默认配置比较输出；对当前 naive/greedy/linear 寄存器分配路径逐一验证。
- [ ] DAG 指令选择单独验证；未支持组合须给出明确错误，不能静默改用另一条路径后记 PASS。
- [ ] 报告脚本提供 `--json`、`--json-output`、`--markdown`、`--html`，退出码区分成功和失败，HTML 转义源码。
- [ ] 将用例测试放进 tests 自动发现范围，benchmark 命令追加到原 benchmark job。
- [ ] Summary 展示支持范围、成功/失败数、性能采样；日志默认折叠，失败原因保持可见。
- [ ] 运行全部原有 CI 命令及新增执行测试；确认不改变其他课题步骤、artifact 名称和 deploy-pages 条件。

验收：新旧测试同时通过，报告里的真实执行状态可核实。测试超时、模拟器缺失或结果不一致均不能被计为通过。未达端到端门槛的阶段 PR 应明确仍未完成的范围。

## 4. 三个完整验收程序

以下是拟新增示例，不代表当前主分支已经能正确执行。输入全部在程序内初始化，避免将未初始化寄存器或内存当作输入。预期结果同时作为独立测试 oracle；不能从编译器本身反推 expected。

### if_else.dsl：期望 7

```text
a = add(3, 0)
b = add(4, 0)
if (a < b):
    result = add(a, b)
else:
    result = sub(a, b)
endif
return result
```

另将 a 改为 5，期望返回 1，确保 else 路径实际被执行。

### while_sum.dsl：期望 15

```text
i = add(1, 0)
total = add(0, 0)
while (i <= 5):
    total = add(total, i)
    i = add(i, 1)
endwhile
return total
```

将 i 初值改为 6，期望 0；改为 5，期望 5。注意仓库现有 `022_dsl_while_sum.dsl` 未更新条件变量 i，只能用作解析输入，不能直接充当可终止的执行基准。

### nested_loop.dsl：期望 6

```text
i = add(0, 0)
total = add(0, 0)
while (i < 3):
    j = add(0, 0)
    while (j < 2):
        if (i >= 0):
            total = add(total, 1)
        else:
            total = sub(total, 1)
        endif
        j = add(j, 1)
    endwhile
    i = add(i, 1)
endwhile
return total
```

额外断言内层 j 每轮重新初始化；比较所有标签的唯一性；在优化开启/关闭时结果相同。

## 5. 测试矩阵与执行约束

| 类别 | 输入变化 | 断言 |
| --- | --- | --- |
| 比较 | 六个运算符 × 小于/相等/大于；负数、0 | 每个运算符真/假均覆盖，左右操作数都影响结果 |
| 浮点能力 | 1.5 与 1.75、负数；后端级 NaN | LLVM 验证及结果；RISC-V 不可整数截断后返回成功 |
| 分支 | 无 else、有 else、两路赋值、入口已有值 | 分别运行两路，验证汇合值 |
| 未初始化 | 仅 then 定义后返回；仅循环体定义后返回 | 明确诊断，无有效编译产物 |
| 循环 | 0/1/5 次、多变量、反向条件、无限循环 | 正确结果或受控超时，零次路径无未定义值 |
| 终结 | then/else/while 内 return；return 后语句 | 无终结指令之后的可达指令，返回值正确 |
| 嵌套 | if/while/for 三类组合 | 块边界、标签、结果和终止均正确 |
| 诊断 | 缺括号、重复 else、结束符错配、非法条件、超过上限 | 原始行列、错误码和上限语义稳定 |
| 兼容 | 现有正确 DSL corpus、ONNX、CFG、基础算子 | 原有接口及正确程序结果不变 |

真实 TinyFive 必须确认 backend 可用且未走 Stub。使用独立进程执行，每个小用例最多 100,000 条指令和 10 秒墙钟时间；越界立即失败并保留输入、IR、汇编、后端及已执行步数。正常返回使用测试驱动设定的返回地址/终止位置，不能把任意提前停止视作成功。验证 ABI 规定的返回值及必要内存，不能用“有汇编输出”代替断言。

LLVM 执行除 verify 外，还应通过标量 JIT 或既有执行适配器核对返回值。参考 expected 使用上节明确常量及独立数学计算；DSLInterpreter 当前不执行分支，因此不作为控制流 oracle。随机测试使用固定种子和有界循环，失败记录可重放输入。

## 6. benchmark 协议

正确性与速度分开报告。固定正确直线 corpus 用于前后兼容与耗时对比；三个控制流示例及生成的有界嵌套用例用于语义验收。规模取 10/100/1000 条语句、嵌套深度 1/4/16，报告语句计数规则和生成种子。

两侧使用同一输入、同一驱动和独立进程，从明确 checkout 导入并记录实际模块路径，防止 editable install 测到同一份代码。每组先预热 5 次，再采样 20 次；计时包含解析及 IR 构建，不包含文件读取、进程启动、后端执行。报告 median、p95、最小/最大值及全部原始采样，执行时间另列。

基线无法正确执行的用例标为 baseline_unsupported 或 baseline_failed，不填零耗时、不计算加速比。每个当前支持用例必须 PASS；预先声明不支持的类型单列，不能扩大分母伪装通过率。首次交付不设未经测量的性能门槛，任何功能失败仍导致非零退出码。详细日志用 details/summary 默认折叠，汇总、失败原因和支持范围常显。

## 7. PR 验证与审查清单

后续每个实现 PR 运行：

```bash
python -m pytest tests/test_parser.py tests/test_dsl_extended.py tests/test_dsl_validator.py tests/test_dsl_errors.py tests/test_dsl_diagnostics_cli.py tests/test_dsl_diagnostics_benchmark.py -q
python -m pytest tests/ -q --tb=short --ignore=tests/test_simulator.py
python -m pytest tests/test_simulator.py::TestStubProfiledMachine -q
python -m pytest benchmarks/test_benchmark.py -q
git diff --check
```

新增控制流执行测试自动包含在 tests/ 内；应额外报告其独立结果。若工作区提供 `.Codex/harness/verify/run.py`，提交前还须运行 `python .Codex/harness/verify/run.py --level L2`。本次基线没有该文件，不能写成 L2 已通过。

自审项目：

- [ ] 课题 01 的变更范围和已支持类型写清，没有复制课题 09 或汇编美化器的 benchmark。
- [ ] 每个语义修复都有先失败后通过的测试证据；没有放松断言或静默回退。
- [ ] 文档中的 API、路径、示例与最终实现一致，未把计划能力写成已完成。
- [ ] 测试报告注明平台、Python、提交和失败/跳过原因；本地验证与 GitHub CI 状态分开说明。
- [ ] 仅暂存本课题文件，不提交虚拟环境、临时报告或本地配置；英文 commit message。
- [ ] PR 基于最新主分支，复用原 CI；若上游变化触及前端/IR/后端，重新执行受影响测试。

## 8. 本次文档 PR 的验证记录

本节只记录文档调研阶段实际验证结果；后续实现按上述门槛重新执行，不沿用本节数字宣称新功能通过。

- Windows / Python 3.13：`tests/`（按 CI 排除 test_simulator.py）加 `benchmarks/test_benchmark.py` 为 756 passed、1 failed；失败为 `TestCompareFiles.test_compare_two_files` 删除未关闭临时文件的 WinError 32，发生于未修改的主分支基线。另有 11 条既有 pytest 标记警告。
- Linux / Python 3.12：相同测试范围为 756 passed、1 skipped、0 failed；跳过的是 constant-merge 的真实 TinyFive 测试，原因是该环境未安装 tinyfive。另有 11 条既有 pytest 标记警告。
- Linux 单独执行 `tests/test_simulator.py`：8 passed、13 skipped，13 项均因缺少 tinyfive 跳过；未将这些结果作为真实控制流执行通过的证据。
- L2 harness 文件缺失，调用返回文件不存在；不计为通过。
