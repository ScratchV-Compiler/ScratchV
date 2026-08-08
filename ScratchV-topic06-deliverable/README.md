# ScratchV 课题 06 交付说明

本测试套件不是独立项目，需要放在 ScratchV 仓库根目录下运行，并依赖 ScratchV 原项目环境。
本目录是 ScratchV 课题 06 的编译正确性与性能回归测试套件，包含 23 个 DSL 测试用例、自动化测试脚本、依赖说明、CI 示例以及可重新生成的测试报告。

当前程序在原始自动编译、指令数统计、Benchmark、基线对比和报告功能上，进一步增加了真实 TinyFive 直通验证、元数据校验、编译与模拟超时、编译失败日志、统一 JSON schema、轻量/完整报告、可配置退化阈值和子集测试。测试套件不会在 TinyFive 不可用时回退到 stub，也不会使用测试套件内部的参考解释器代替真实编译结果。

真实验证路径为：

```text
DSL -> ScratchV 编译器 -> RISC-V 汇编 -> TinyFive -> a0/x10 返回值 -> 期望值对比
```

最近一次全量运行结果为 13 个通过、10 个失败。3 个 `branch` 用例因 TinyFive 模拟超时失败；另外 7 个 `dot`/`matmul` 用例涉及尚未完成的数组内存输入输出约定和编译器后端降低逻辑。测试套件负责真实暴露、隔离和记录这些问题，不负责实现 ScratchV 的分支、数组或矩阵代码生成。

## 运行依赖

该交付目录需要放在 ScratchV 项目环境中运行。`run_tests.py` 会导入
`scratchv` 包，并通过 ScratchV 编译器把每个 DSL 用例编译为 RISC-V 汇编。
调用真实 TinyFive 所需的 ScratchV 适配修改可参考：

- [ScratchV PR #15](https://github.com/ScratchV-Compiler/ScratchV/pull/15)
- [ScratchV PR #17](https://github.com/ScratchV-Compiler/ScratchV/pull/17)

安装基础测试依赖：

```powershell
pip install -r requirements-topic06.txt
```

基础模式需要以下依赖：

- `tinyfive`：用于模拟执行生成的 RISC-V 汇编
- `pytest`：用于测试支持

如需生成 HTML 和 PNG 完整报告，再安装可选依赖：

```powershell
pip install -r requirements-topic06-full.txt
```

其中 `jinja2` 用于生成 HTML，`matplotlib` 用于生成性能图表。

## 目录结构

```text
run_tests.py
requirements-topic06.txt
requirements-topic06-full.txt
tests_main/
  activation/
  branch/
  elementwise/
  loop/
  reduction/
  tensor/
reports/        # 测试报告，可重新生成
  failures/     # 编译失败和超时日志
.github/
  workflows/
    benchmark.yml
build/          # 编译输出的汇编文件，可重新生成
```

`tests_main/` 下共有 23 个 DSL 用例。每个用例由 `.dsl` 文件和对应的
`.meta.json` 文件组成，`.meta.json` 中定义输入、期望返回值和用例说明。

## 运行测试

在本目录下执行：

```powershell
python run_tests.py
```

测试脚本会自动完成以下步骤：

- 遍历 `tests_main/` 下的所有 `.dsl` 文件
- 调用 ScratchV 编译器生成 RISC-V 汇编
- 编译单个用例超过 30 秒时终止该编译进程，并继续运行后续用例
- 编译失败或超时时将命令、返回码、stdout 和 stderr 保存到 `reports/failures/*.compile.log`
- 通过 ScratchV 的 TinyFive 适配层验证生成的汇编
- 在独立子进程中执行 TinyFive；单次模拟超过 5 秒时终止该用例并继续后续测试
- TinyFive 未安装或执行失败时明确判定为 FAIL，不回退到 stub
- 使用 TinyFive 返回寄存器 `a0/x10` 与 `.meta.json` 中的期望值进行结果对比
- 使用 `time.perf_counter()` 记录编译、模拟和总耗时
- 统计 PASS/FAIL 和指令数
- 默认在 `reports/` 下生成 Markdown 和固定 schema 的 JSON 报告

## Benchmark 模式

重复运行每个用例并统计平均指令数：

```powershell
python run_tests.py --benchmark 3
```

如果首次模拟已经 timeout，该用例不会再执行 Benchmark 重复；如果重复过程中发生 timeout 或模拟失败，脚本会立即停止剩余次数。失败运行不会作为 `0` 加入平均值，报告会记录实际完成次数和 `benchmark_stopped_reason`。

更新性能基线：

```powershell
python run_tests.py --benchmark 3 --update-baseline
```

性能退化阈值默认是 5%，可以按场景调整：

```powershell
# 超过 2% 即判定为退化
python run_tests.py --benchmark 3 --regression-threshold 2

# 超过 10% 才判定为退化
python run_tests.py --benchmark 3 --regression-threshold 10
```

阈值必须大于或等于 0，并会写入 Markdown、HTML 和 JSON 报告。

## 子集测试

按类别运行：

```powershell
python run_tests.py --category activation
```

按用例名称进行大小写不敏感的包含匹配：

```powershell
python run_tests.py --filter matmul
```

组合筛选：

```powershell
python run_tests.py --category tensor --filter relu
```

没有匹配用例时脚本返回退出码 2，并保留已有报告。对子集使用 `--update-baseline` 时，只更新匹配用例的基线，不会删除其他用例的基线。

性能基线文件位于：

```text
reports/benchmark_baseline.json
```

## 测试报告

默认运行后生成轻量报告：

```text
reports/report.md
reports/report.json
```

需要 HTML 和 PNG 时运行：

```powershell
python run_tests.py --full-report
```

完整模式额外生成：

```text
reports/report.html
reports/course_report_instructions.png
```

`report.md` 适合提交或归档，`report.json` 供 CI 或其他程序稳定解析，`report.html` 适合在浏览器中查看。普通模式和 Benchmark 模式使用相同字段，Benchmark 专属字段在普通模式下为 `null`。JSON 顶层的 `selection` 会记录 `--category` 和 `--filter`，未筛选时两项均为 `null`。轻量模式不会加载 `jinja2` 或 `matplotlib`，也不会刷新已有的 HTML、PNG 文件。

编译失败时，报告中的 `compile_returncode`、`compile_timed_out`、`compile_error` 和 `compile_log` 会记录错误摘要及独立日志位置。成功用例的错误和日志字段为 `null`。

## 添加测试用例

新增用例时添加两个文件：

```text
tests_main/{category}/{name}.dsl
tests_main/{category}/{name}.meta.json
```

DSL 示例：

```text
result = add(a, b)
return result
```

元数据示例：

```json
{
  "description": "Simple scalar add.",
  "expected_output_type": "scalar",
  "inputs": {
    "a": 2,
    "b": 3
  },
  "expected_return": 5
}
```

矩阵输出需要声明类型、元素类型和形状：

```json
{
  "expected_output_type": "tensor",
  "output_dtype": "int32",
  "output_shape": [2, 2],
  "expected_return": [[19, 22], [43, 50]]
}
```

大型期望结果也可以使用 `expected_output_file` 引用同目录下的 JSON 文件。`expected_return` 与 `expected_output_file` 必须且只能填写一个。元数据中的类型、形状或元素值不合法时，仅当前用例标记为 FAIL，后续测试继续运行。

## CI 示例

GitHub Actions 示例位于：

```text
.github/workflows/benchmark.yml
```

示例工作流会在 push 和 pull request 时安装基础依赖、运行测试并上传 Markdown、JSON、性能基线和编译失败日志。CI 默认使用轻量报告，不需要安装 `jinja2` 和 `matplotlib`。

## 详细设计

程序模块、报告字段、需求演进、Mentor Review 处理状态和当前失败原因见 `课题6设计文档.md`。
