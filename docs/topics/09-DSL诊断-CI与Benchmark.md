# 课题 9：DSL 诊断 CI 与 benchmark

`.github/workflows/dsl-diagnostics.yml` 提供两个检查项：`Topic 9 tests` 和
`Topic 9 benchmark`。它们使用 GitHub 托管 Linux runner 和 Python 3.12，运行于
指向 main 的 PR、main 的 push，也支持手动触发。现有全项目 CI 继续保留。

## 专项测试

```powershell
python -m pip install -e . "pytest>=7,<10"
python -m pytest tests/test_dsl_errors.py tests/test_dsl_validator.py tests/test_dsl_diagnostics_cli.py tests/test_parser.py tests/test_dsl_extended.py tests/test_dsl_diagnostics_benchmark.py -v --tb=short
```

测试覆盖错误模型、位置、提示、块恢复、多错误、颜色、CLI 和正常解析回归，
以及 benchmark 的失败传播、基线导入隔离和报告产物。CI 上传 JUnit XML 到
`dsl-test-reports`。既有全量 CI 也会发现这些测试；专项检查有意提供独立可见的
课题 9 结果，不改变其他课题的测试范围。

## 本地 benchmark

先为基线建立独立 checkout。以下 `997d2aa` 是本 PR 加入诊断实现之前的历史版本；
rebase 后与最新上游比较时，应改用对应的 PR base SHA。

```powershell
git worktree add --detach ../ScratchV-dsl-baseline 997d2aa
python -m benchmarks.bench_dsl_diagnostics --baseline-root ../ScratchV-dsl-baseline --json-output benchmark_reports/dsl_diagnostics.json --markdown benchmark_reports/dsl_diagnostics.md --html benchmark_reports/dsl_diagnostics.html
```

`--json` 是布尔开关，仅向 stdout 输出 JSON；文件输出使用 `--json-output`。
HTML 使用标准库生成，不依赖外部样式、脚本或可视化库。报告同时包含成功与失败信息。

### 正确 DSL 解析性能

- 两个版本均使用**当前 checkout 的同一批** `benchmarks/cases/*.dsl`，包含基础语法、
  for、if/else、while。统一调用 `ExtendedDSLParser().parse(source)`，覆盖 CLI 使用的解析器。
- 同一脚本通过两个独立 Python 进程导入各自 checkout 的解析器，并验证模块路径。
  不用关闭当前 validator 的方式冒充旧版解析器。
- 两侧复用当前项目安装的依赖及同一个 Python。每侧预热 5 次，再测量 10 组；
  每组解析完整语料 100 次。计时包含创建解析器和解析，排除进程启动、导入、文件读取、
  IR 摘要计算和每组开始前的显式 GC；计时期间保留 Python 默认 GC。
- 比较每个文件的源码 SHA-256 和 `Program.dump()` SHA-256，任何解析异常、空语料、
  源码差异或 IR 差异均使检查失败，不跳过失败用例。
- JSON 记录原始组耗时、中位数、倍率、实际基线/当前 SHA、模块路径、Python、系统和依赖版本。
  两个 checkout 的提交信息不包含未提交修改，正式报告应使用干净 checkout。

### 错误输入诊断

12 个内置固定用例独立验收错误码和行列、源码、文件名、提示、无 ANSI 输出、CLI 退出码 1、
诊断不重复和不产生汇编文件。包含 CRLF/tab/Unicode、三个独立错误以及 25 个错误输入
触发 20 条诊断上限的情况。分别测量验证/收集和纯文本渲染；CLI 用于正确性验收，不计时。

错误样例不放入正常 `benchmarks/cases/`，避免通用 DSL runner 将预期错误当作编译失败。
旧版本不具备新诊断能力，不参与错误输入的等价性能比较。

## 基线与结果判定

- PR：基线为事件中的 `pull_request.base.sha`，当前为默认 checkout 的 PR 合并测试提交。
- main push：基线为事件中的 `before`，当前为该 push 提交。
- 手动：选择 workflow 的执行分支，并在 `baseline_ref` 指定基线，默认 main。
- 基线不存在、无法 checkout 或 worker 超时均失败，不回退到另一个版本。

**诊断验收、IR 一致性及报告执行错误是硬门禁。** 设计文档的解析倍率目标是 1.5x，
默认明确显示 `target_met` 和超标提示，但不因共享 runner 的计时波动阻止合并。
因此 CI 绿色不等于已经达到 1.5x 性能目标。需要严格执行该目标时添加
`--enforce-performance`，阈值由 `--max-parse-ratio` 指定，默认 1.5。

benchmark 结束后，Markdown 写入 Job Summary；JSON、Markdown 和 HTML 上传至
`dsl-benchmark-reports`。上传及汇总步骤使用 `always()`，功能失败仍保留诊断证据。
这里不计算常量合并次数、TinyFive 指令减少量或 LLVM 指令数收益。

## 本地验证与自审记录（2026-09-13）

环境为 Windows、Python 3.13.3；workflow 指定的 Linux/Python 3.12 运行结果需由 GitHub Actions 确认。

- 专项测试：117 passed，包含 12 项 benchmark 回归测试。
- 全量命令 `python -m pytest tests benchmarks/test_benchmark.py -q`：427 passed、3 failed。
  三项失败均在独立的 `997d2aa` 基线 checkout 复现：
  `TestCompareFiles.test_compare_two_files` 删除未关闭临时文件时遇到 Windows 文件锁；
  `TestVerifyAssembly` 的两个测试调用到未定义的 `load_asm`。相关源码与基线无差异，
  本次 CI 接入未修改这些模块，也未新增跳过规则。
- 23 个正确 DSL 用例全部可解析，基线与当前 IR 摘要一致；12 个固定诊断用例全部通过。
- 独立运行 benchmark、避免与 pytest 并行：基线组中位数 0.067008 秒，当前 0.117891 秒，
  倍率 **1.759x**，未达到设计文档的 1.5x 目标。基线为 `997d2aa`，当前编译器实现为
  `c9dd532`。这是当前机器的一次测量，不代表 Linux runner 的结果。
- Python 3.12 语法解析、compileall、workflow YAML 和全部六段 Bash 脚本语法检查通过。
- 已尝试 L2 命令，但本地 `.Codex/harness/verify/run.py` 不存在；不宣称 L2 通过。

自审结论：新增检查只验收 DSL 诊断和解析；基线导入隔离、IR 差异失败传播、失败报告保留、
HTML 源码转义及性能目标显式标记均有回归覆盖。原有全量测试失败和解析性能目标未达标
作为已知问题保留，不包装成“全量通过”或性能提升。
