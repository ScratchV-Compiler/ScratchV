# ScratchV 课题 06 交付说明
本测试套件不是独立项目，需要放在 ScratchV 仓库根目录下运行，并依赖 ScratchV 原项目环境。
本目录是 ScratchV 课题 06 的性能基准与验证测试套件，包含 23 个 DSL测试用例、自动化测试脚本、依赖说明以及可选的测试报告。
其中3个branch中的三个if分支调用tinyfive时目前还不通过，是因为当前 ScratchV 的分支汇编和 tinyfive 适配层没有对齐。

## 运行依赖

该交付目录需要放在 ScratchV 项目环境中运行。`run_tests.py` 会导入
`scratchv` 包，并通过 ScratchV 编译器把每个 DSL 用例编译为 RISC-V 汇编。
若调用真实的tinyfive而非stub需要安装两个源码补丁：
https://github.com/ScratchV-Compiler/ScratchV/pull/15
https://github.com/ScratchV-Compiler/ScratchV/pull/17
安装额外依赖：

```powershell
pip install -r requirements-topic06.txt
```

至少需要以下依赖：

- `tinyfive`：用于模拟执行生成的 RISC-V 汇编
- `pytest`：用于测试支持
- `jinja2`：用于生成 HTML 报告
- `matplotlib`：用于生成性能图表

## 目录结构

```text
run_tests.py
requirements-topic06.txt
tests_main/
  activation/
  branch/
  elementwise/
  loop/
  reduction/
  tensor/
reports/        # 测试报告，可重新生成
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
- 通过 ScratchV 的 TinyFive 适配层验证生成的汇编
- 根据 `.meta.json` 中的期望值进行结果对比
- 增加时间测量（time.perf_counter），输出到报告
- 统计 PASS/FAIL 和指令数
- 在 `reports/` 下生成 Markdown 和 HTML 报告

## Benchmark 模式

重复运行每个用例并统计平均指令数：

```powershell
python run_tests.py --benchmark 3
```

更新性能基线：

```powershell
python run_tests.py --benchmark 3 --update-baseline
```

性能基线文件位于：

```text
reports/benchmark_baseline.json
```

## 测试报告

运行后会生成：

```text
reports/report.md
reports/report.html
reports/course_report_instructions.png
```

`report.md` 适合提交或归档，`report.html` 适合在浏览器中查看测试结果。

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
  "expected_output_type": "return_value",
  "inputs": {
    "a": 2,
    "b": 3
  },
  "expected_return": 5
}
```