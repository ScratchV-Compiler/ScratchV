# 课题 6：ScratchV 课程版性能测试套件使用说明

本文档只说明当前作业交付使用的课程版测试套件：`run_tests.py` 和 `tests_main/`。

## 1. 目录结构

```text
run_tests.py             # 自动化测试脚本

tests_main/
  activation/            # relu 等激活函数用例
  branch/                # if/else 分支用例
  elementwise/           # add 和链式 add 用例
  loop/                  # for/endfor 循环用例
  reduction/             # dot/reduction 用例
  tensor/                # matmul/tensor 用例

build/                   # 编译后生成的 RISC-V 汇编

reports/
  report.md              # Markdown 测试报告
  report.html            # HTML 测试报告
  course_report_instructions.png
  benchmark_baseline.json

.github/workflows/
  benchmark.yml          # CI 示例，运行课程版测试套件
```

当前 `tests_main/` 下有 23 个 DSL 用例，覆盖算术、神经网络算子、循环、if/else 分支、矩阵计算和组合场景。

## 2. 运行测试

在项目根目录运行：

```powershell
python run_tests.py
```

运行后会自动：

- 遍历 `tests_main/` 下的 `.dsl` 文件。
- 调用 ScratchV 编译器生成汇编。
- 调用 TinyFive 适配器模拟执行。
- 使用参考执行器计算实际返回值。
- 对比实际返回值和 `.meta.json` 中的预期返回值。
- 统计 PASS/FAIL 和指令数。
- 生成报告。

## 3. Benchmark 模式

重复运行 3 次并取平均：

```powershell
python run_tests.py --benchmark 3
```

报告会记录：

- 平均指令数
- 最小指令数
- 最大指令数
- 95% 置信区间
- 基线指令数
- 性能变化率
- 是否性能退化

## 4. 性能基线和退化判断

第一次生成基线：

```powershell
python run_tests.py --benchmark 3 --update-baseline
```

之后正常运行：

```powershell
python run_tests.py --benchmark 3
```

判断规则：

- 当前平均指令数比基线高出 5% 以上，判定为性能退化。
- 低于或等于 5% 的波动不算退化。
- 基线文件保存在 `reports/benchmark_baseline.json`。

## 5. 报告文件

运行后生成：

```text
reports/report.md
reports/report.html
reports/course_report_instructions.png
```

`report.md` 适合提交作业或放进文档。`report.html` 适合演示，包含表格和 `matplotlib` 生成的性能图表。

## 6. 添加新测试用例

每个课程版用例由两个文件组成：

```text
tests_main/{category}/{name}.dsl
tests_main/{category}/{name}.meta.json
```

`.dsl` 示例：

```text
# Simple add
result = add(a, b)
return result
```

`.meta.json` 示例：

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

添加后运行：

```powershell
python run_tests.py --benchmark 3
```

如果失败，优先检查 DSL 语法、输入变量名、预期输出和当前编译器是否支持该算子。

## 7. if/else 分支语法

当前支持简单分支：

```text
if flag
result = add(a, b)
return result
else
result = sub(a, b)
return result
endif
```

规则：

- `flag` 非 0 时走 `if` 分支。
- `flag` 为 0 时走 `else` 分支。
- 当前不支持复杂比较表达式，比如 `if a > b`。

## 8. CI

`.github/workflows/benchmark.yml` 会在 push 和 pull request 时运行：

- `python -m pytest -q`
- `python run_tests.py --benchmark 3`

CI 会上传课程版报告文件，方便查看每次修改后的正确性和性能变化。
