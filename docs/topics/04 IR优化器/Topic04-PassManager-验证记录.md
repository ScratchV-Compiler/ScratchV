# Topic 04 PassManager 验证记录

日期：2026-09-26。环境：Windows，Python 3.13.5，仓库 `.venv`。
基线：PR #48 `f0e4e6d` + 上游 main `11a2c3c`。

## 自动化验证

```bash
python -m pytest tests/test_pass_manager.py tests/test_pass_registry.py -q
```

结果：66 passed。覆盖条件开关、真实常量折叠、名称选择、空/重复管线、
禁用不构造、工厂独立实例、功能性输出串接、IR 分析、失败即停及输出保护。

```bash
python -m pytest tests benchmarks -q --tb=short --ignore=tests/test_standalone_execution.py
```

结果：1067 passed、4 skipped、1 failed。
唯一失败为 `tests/test_inst_counter.py::TestCompareFiles::test_compare_two_files`，
原因是在 Windows 上删除仍打开的临时文件（WinError 32）。
此测试在独立的、未修改的主分支 `11a2c3c` 上也以相同错误失败。

未忽略模块的首次收集发现 `tests/test_standalone_execution.py` 顶层导入 Unix
`resource`，Windows 不提供该模块。在未修改主分支上也复现此收集错误。
不修改这些无关测试，不把受限平台测试报告为已通过；Linux CI 负责执行它们。

相关集成子集（PM、优化器、ONNX benchmark、寄存器分配指标）共 122 passed。

## Topic 04 CI（2026-09-27）

在现有 `.github/workflows/ci.yml` 的 `test` 作业中加入专项检查，沿用 PR 到
`main` 及现有主线分支 push 的触发规则。专项检查在 RISC-V 工具安装及全量测试前执行。

```bash
python -m pytest tests/test_pass_manager.py tests/test_pass_registry.py tests/test_optimizer.py tests/test_optimizer_advanced.py -v --tb=short --junit-xml=benchmark_reports/topic04_pass_manager.xml
```

本地同一测试集合结果：78 passed（Windows / Python 3.13.5）。线上使用已有
Linux / Python 3.12 配置；本地结果不代表线上已通过。
CI 通过 `pipefail` 保留测试失败状态，失败会阻止该作业通过。
结果状态写入 Actions 摘要，JUnit XML 与详细日志归入现有 `test-reports` artifact，
保留 30 天；测试失败时仍尝试上传报告。

## 静态检查

以下 5 个核心文件的 mypy 检查通过：`scratchv/pass_manager.py`、
`scratchv/assembly_passes.py`、`scratchv/pass_interface.py`、
`scratchv/compiler.py`、`scratchv/main.py`。
使用 `--ignore-missing-imports --follow-imports=silent`。

以上文件及两个 PM 测试文件的 Ruff、Black（target py312）、isort（black profile）
检查通过。`git diff --check` 和 `scripts/check_docs_links.py` 通过。

仓库指定的本地 `.claude/harness/verify/run.py` 未提供；未声称执行了 L2 harness。
仓库 pre-commit 配置固定 Black 使用 Python 3.12，而本机只提供 Python 3.13；
本次直接执行上述检查，不声称整套 pre-commit hooks 已通过。

## 自审结论

### 2026-09-27 同步主分支冲突处理

合入主分支 `f747aa1`（包含 #75 调度器与 #57 窥孔 benchmark 更新）。
CI 保留 Topic 04 专项检查及主分支新增 LLVM 工具安装；调度适配器改用
`schedule_assembly`，保留 strict、LLVM 路径、报告、诊断及失败时输出保护。
新增三项回归验证配置传递、报告与统计隔离、两类调度异常的输出保护。

PM、优化器、窥孔相关测试及调度禁用/无效参数测试共 135 passed，使用 Python
UTF-8 模式避免 Windows 默认 GBK 读取上游 HTML 测试产物失败。
修改的核心模块 mypy、Ruff、Black 检查通过，文档 139 个链接检查通过。
本机缺少 llvm-mca、RISC-V 执行工具；真实调度测试因工具缺失未通过，
CPU 执行测试跳过，交由保留完整工具安装步骤的 Linux CI 验证。

### 设计审查

- 保留最新主分支的 DSL 诊断、寄存器映射、ABI frame 和汇编优化行为。
- 常量折叠算法来自现有实现，变化为 #48 接口适配、调度与开关。
- 名称注册与默认启用分离；禁用在构造工厂对象前生效。
- `none`、显式空列表、重复 Pass 及禁用全部同名项有明确语义。
- 保留旧 PM 导入路径、IR 原地执行和统计契约；新配置字段追加，保留原位置参数顺序。
- 汇编与 IR 分阶段构建；真实只读指令统计通过 PM 执行。
- 新主分支中的 CNN benchmark 旧接口调用已迁移，并纳入回归。
- 不引入新的常量折叠数值语义、分析缓存或隐式不动点调度。
