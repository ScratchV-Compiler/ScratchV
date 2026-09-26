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

- 保留最新主分支的 DSL 诊断、寄存器映射、ABI frame 和汇编优化行为。
- 常量折叠算法来自现有实现，变化为 #48 接口适配、调度与开关。
- 名称注册与默认启用分离；禁用在构造工厂对象前生效。
- `none`、显式空列表、重复 Pass 及禁用全部同名项有明确语义。
- 保留旧 PM 导入路径、IR 原地执行和统计契约；新配置字段追加，保留原位置参数顺序。
- 汇编与 IR 分阶段构建；真实只读指令统计通过 PM 执行。
- 新主分支中的 CNN benchmark 旧接口调用已迁移，并纳入回归。
- 不引入新的常量折叠数值语义、分析缓存或隐式不动点调度。
