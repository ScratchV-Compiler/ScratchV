# 课题 9：DSL 诊断 CI 与 benchmark

课题 9 接入原有 `.github/workflows/ci.yml`，不新增 workflow 或 job。
现有 `test` job 运行 DSL 测试，现有 `benchmark` job 增加
`Topic 9 DSL diagnostics benchmark` 步骤。沿用上游 runner 配置（PR 使用 ubuntu-latest，
push 使用 self-hosted）、Python 3.12
和触发条件：指向 main 的 PR，以及 main、wjy_dev、jzj_dev 的 push。

## 专项测试

```powershell
python -m pip install -e . "pytest>=7,<10"
python -m pytest tests/test_dsl_errors.py tests/test_dsl_validator.py tests/test_dsl_diagnostics_cli.py tests/test_parser.py tests/test_dsl_extended.py tests/test_dsl_diagnostics_benchmark.py -v --tb=short
```

测试覆盖错误模型、位置、提示、块恢复、多错误、颜色、CLI 和正常解析回归，
以及 benchmark 的失败传播、基线导入隔离和报告产物。原有 `pytest tests/` 会自动
发现这些文件，不重复执行专项测试。结果合并到 `benchmark_reports/test_results.xml`，
由原有 `test-reports` artifact 上传，PR 和失败运行也保留测试报告。

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

- PR：基线为事件中的 `pull_request.base.sha`，当前为事件的 PR 合并测试提交。
- push：基线为事件中的 `before`，当前为该 push 提交。新分支首次 push 的全零
  `before` 无法用于对比，该步骤会明确失败；打开指向 main 的 PR 后使用 PR base。
- 基线通过临时 Git worktree 准备，退出步骤时清理，不混入报告 artifact。
- 两个 job 都必须 checkout 到事件的 `GITHUB_SHA`；获取提交、准备基线或 worker
  执行失败均使相应步骤失败，不静默回退到 main。

**诊断验收、IR 一致性及报告执行错误是硬门禁。** 设计文档的解析倍率目标是 1.5x，
默认明确显示 `target_met` 和超标提示，但不因共享 runner 的计时波动阻止合并。
因此 CI 绿色不等于已经达到 1.5x 性能目标。需要严格执行该目标时添加
`--enforce-performance`，阈值由 `--max-parse-ratio` 指定，默认 1.5。

benchmark 结束后，Markdown 写入 Job Summary；JSON、Markdown 和 HTML 上传至
原有 `benchmark-reports`。上传及汇总步骤使用 `always()`，功能失败仍保留诊断证据。
Markdown 摘要和本地 HTML 中，每个用例的完整诊断日志默认折叠，点击用例名称展开。
汇总表、失败检查和性能提示直接可见；JSON 保留完整诊断内容。
这里不计算常量合并次数、TinyFive 指令减少量或 LLVM 指令数收益。
