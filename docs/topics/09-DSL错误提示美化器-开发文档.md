# 课题9：DSL错误提示美化器开发文档

> **文档类型**：开发指南 | **状态**：草案 | **版本**：v0.1
> **调研基线**：`main@997d2aa` | **建议周期**：12 周
> **重要说明**：本文是后续开发指导，不表示文中功能已经在当前仓库实现

---

## 1. 课题目标

本课题在现有 `scratchv/frontend/dsl_errors.py` 基础上，把结构化错误诊断真正接入：

- `DSLParser`；
- `ExtendedDSLParser`；
- `CompilerDriver` 的 DSL 编译路径；
- ScratchV CLI 的错误输出。

最终效果是：用户提交错误 DSL 时，能够看到准确位置、源码高亮、稳定错误码和可信的修复建议；需要检查整个文件时，可以一次报告多个独立错误。

详细架构和接口约束见 [设计文档](09-DSL错误提示美化器-设计文档.md)。

---

## 2. 当前起点

### 2.1 已经存在的代码

| 文件 | 需要先理解的内容 |
|------|------------------|
| `scratchv/frontend/dsl_errors.py` | `DSLSyntaxError`、`format_error()`、`ErrorCollector`、建议表 |
| `scratchv/frontend/dsl_parser.py` | 基础 DSL 的逐行正则解析与 IRBuilder 调用 |
| `scratchv/frontend/dsl_extended.py` | `if/while` 块解析、块索引和嵌套逻辑 |
| `scratchv/compiler.py` | DSL/ONNX 分流、扩展解析器回退、`CompileResult.errors` |
| `scratchv/main.py` | CLI 参数、编译结果和 stderr 输出 |
| `tests/test_dsl_errors.py` | 现有错误模块的单元测试风格 |
| `tests/test_parser.py` | 基础 DSL 成功路径 |
| `tests/test_dsl_extended.py` | 扩展语法及 `DSLParseError` 兼容要求 |

### 2.2 当前能力边界

当前错误美化模块可以独立构造和打印错误：

```python
from scratchv.frontend.dsl_errors import make_error, format_error

error = make_error(
    line=2,
    col=10,
    message="unsupported operation 'ad'",
    source_line="result = ad(a, b)",
    filename="bad.dsl",
    error_code="E200",
)
print(format_error(error, use_color=False))
```

但是解析器并不会自动产生这个错误对象。`ErrorCollector` 也只是容器，当前解析流程没有错误恢复机制。

### 2.3 开发前必须复现的问题

开始改代码前，至少记录以下三类基线输出：

```text
1. 无法识别的普通语句
2. 不支持的算子
3. 缺失 endif/endwhile 的扩展 DSL
```

基线记录应包含：输入 DSL、调用入口、异常类型、异常文本和退出码。后续用相同输入证明错误质量确实改善。

---

## 3. 前置知识

开始课题前，建议掌握：

1. Python 异常继承、`dataclass`、`enum` 和类型注解；
2. 正则表达式的 `match`、`fullmatch` 和捕获组；
3. 编译器前端中的源码位置、错误恢复和级联错误；
4. ScratchV 的 `IRBuilder` 与 `Program`；
5. pytest 参数化、异常断言和文本快照；
6. ANSI 转义码、TTY 和 stderr；
7. Git 小步提交和回归测试。

不要求先实现完整 tokenizer、AST 或 LSP。

---

## 4. 推荐阅读顺序

### 第一步：跑通正确 DSL

阅读并运行：

```bash
python -m pytest tests/test_parser.py tests/test_dsl_extended.py -q
```

目标：理解正确程序如何从 DSL 进入 `IRBuilder`，不要先看错误模块就直接修改异常类型。

### 第二步：单独理解错误模块

```bash
python -m pytest tests/test_dsl_errors.py -q
```

重点回答：

- `DSLSyntaxError` 的字段哪些是 1-based？
- `format_error()` 如何计算 caret 长度？
- `ErrorCollector` 达到上限后怎样计数？
- 自动建议来自显式 `fix_hint` 还是启发式规则？

### 第三步：追踪 CLI 数据流

```text
scratchv/main.py
  → CompilerDriver.compile()
  → CompilerDriver._parse()
  → ExtendedDSLParser 或 DSLParser
  → CompileResult.diagnostics / diagnostic_limit_reached / errors
  → stderr
```

特别关注 `_parse()` 中捕获所有异常再回退的逻辑。它是错误信息被覆盖的主要风险点。

### 第四步：阅读设计约束

阅读 [设计文档](09-DSL错误提示美化器-设计文档.md) 的第 3、5、7、9、12 和 16 节，再开始编码。

---

## 5. 建议目录与改动范围

后续实现建议只修改与课题直接相关的文件：

```text
scratchv/frontend/
├── dsl_errors.py          # 扩展位置、渲染和收集语义
├── dsl_grammar.py         # 拟新增：Parser/Validator 共享语法与算子签名
├── dsl_validator.py       # 拟新增：无 IR 副作用的验证器
├── dsl_parser.py          # 接入 SourceBuffer/validator
├── dsl_extended.py        # 接入控制流块验证
└── __init__.py            # 必要时导出稳定公共接口

scratchv/
├── compiler.py            # 保留结构化诊断，收窄回退条件
└── main.py                # 终端颜色与 stderr 输出

tests/
├── test_dsl_errors.py
├── test_dsl_validator.py  # 拟新增
├── test_parser.py
├── test_dsl_extended.py
└── test_dsl_diagnostics_cli.py  # 拟新增端到端测试
```

不要在本课题中顺便重构 IR、优化器或后端。

---

## 6. 开发策略

采用测试先行和小步集成。每一步都应保持正确 DSL 可编译，不能等到最后一次性跑测试。

建议顺序：

```text
锁定输出契约
  → SourceBuffer
  → 错误继承兼容
  → 基础行级验证
  → 算子签名验证
  → 扩展块验证
  → Parser 集成
  → CompilerDriver/CLI 集成
  → 多错误与颜色策略
  → 全量回归
```

---

## 7. 阶段一：锁定诊断输出契约

### 7.1 先写失败测试

为以下输出建立无颜色精确测试：

```text
bad.dsl:5:1: error[E100]: cannot parse statement
  5 | retrun result
    | ^~~~~~
note: did you mean 'return'?
```

至少断言：

- 文件名；
- 1-based 行列；
- 错误码；
- 源码行；
- caret 起点和长度；
- 无 ANSI 转义码。

### 7.2 保持旧接口

已有调用仍应工作：

```python
DSLSyntaxError(1, 1, "message")
format_error(error, use_color=False)
ErrorCollector(filename="test.dsl", max_errors=20)
```

不要为了新设计删除现有参数或更改已有字段顺序。

### 7.3 完成标准

- 新输出契约测试失败的原因是功能尚未实现，而不是测试写错；
- 现有 `tests/test_dsl_errors.py` 仍通过；
- 文档中的示例与测试期望完全一致。

---

## 8. 阶段二：实现 SourceBuffer

### 8.1 目的

解析器当前过早执行 `strip()`，导致缩进和原始列信息丢失。`SourceBuffer` 应成为唯一的源码位置来源。

建议接口：

```python
@dataclass(frozen=True)
class SourceBuffer:
    text: str
    filename: str = "<dsl>"

    def line_text(self, line: int) -> str:
        ...
```

### 8.2 测试矩阵

| 输入 | 预期 |
|------|------|
| `a\nb` | 第 1 行 `a`，第 2 行 `b` |
| `a\r\nb` | 与 LF 行号一致 |
| 空字符串 | 不越界，不虚构源码 |
| 末尾换行 | 不产生错误的额外语句 |
| 中文标识符或注释 | 列号按 Python 字符索引计算 |
| 制表符 | 原始列稳定，渲染时 caret 对齐 |

### 8.3 常见错误

- 在保存原始行之前调用 `strip()`；
- 混用 0-based 内部索引和 1-based 用户位置；
- 用字节偏移计算 Unicode 列号；
- 只在扩展解析器保留空行，基础解析器仍丢失行号。

---

## 9. 阶段三：统一异常兼容关系

当前 `DSLParseError` 定义在 `dsl_parser.py`，`DSLSyntaxError` 独立继承 `Exception`。直接改为抛出新异常可能破坏捕获 `DSLParseError` 的调用方和测试。

建议目标：

```python
try:
    DSLParser().parse(bad_source)
except DSLParseError as error:
    assert isinstance(error, DSLSyntaxError)
```

实现时应避免 `dsl_parser.py` 与 `dsl_errors.py` 的循环导入。可选择：

1. 把兼容基类移到 `dsl_errors.py`，再从旧路径重新导出；
2. 把共享异常基类放进一个小型无依赖模块。

不建议通过捕获任意异常再用字符串包装的方式伪造兼容，因为它会丢失原始错误类别和位置。

---

## 10. 阶段四：基础 DSL 验证器

### 10.1 为什么不能在错误后继续构建 IR

一条语句可能已经调用了部分 `IRBuilder` 方法才失败。继续解析会让 builder 状态不可预测。因此验证器只检查源码，不创建 `Value`、Block 或 Program。

### 10.2 建议验证顺序

对每个物理行：

1. 跳过空行和注释；
2. 判断语句类别；
3. 检查外层结构；
4. 检查关键字和块栈；
5. 检查算子是否存在；
6. 检查位置参数与关键字参数；
7. 记录诊断或进入下一行。

### 10.3 使用 `fullmatch`

验证语法时优先使用 `fullmatch`，避免只匹配行首后忽略尾部垃圾。例如：

```text
x = add(a, b) unexpected
```

不能因为前半段匹配成功而被当成合法语句。

### 10.4 算子签名表

不要在多个 `if/elif` 中重复参数规则。建议集中描述：

```python
OP_SIGNATURES = {
    "add": OpSignature(positional=2),
    "relu": OpSignature(positional=1),
    "softmax": OpSignature(positional=1, optional_kwargs={"axis"}),
    "matmul": OpSignature(
        positional=2,
        optional_kwargs={"rows", "cols", "inner", "m", "n", "k"},
    ),
}
```

`OP_SIGNATURES`、语句正则和关键字集合必须放在 Parser 与 Validator 都导入的共享模块中，不能各自维护副本。Parser 的执行 handler 可以独立存在，但必须增加自动测试：`set(OP_SIGNATURES) == set(OP_HANDLERS)`。这样新增算子时，只要漏改一侧，测试会立即失败。

### 10.5 完成标准

- 不支持算子不再泄漏普通 `DSLParseError` 文本；
- 参数不足不再泄漏 `IndexError`；
- 一行结构错误不会产生多个级联错误；
- 正确基础 DSL 的 Program 与基线一致。

---

## 11. 阶段五：扩展 DSL 块验证

### 11.1 块栈

使用独立的验证栈，不复用 IR builder 的 `_loop_stack`：

```python
@dataclass
class BlockFrame:
    kind: Literal["if", "while", "for"]
    line: int
    col: int
    saw_else: bool = False
```

### 11.2 必测规则

- `endif` 只关闭 `if`；
- `endwhile` 只关闭 `while`；
- `endfor` 只关闭 `for`；
- `else` 必须位于 `if` 内；
- 同一个 `if` 只能有一个 `else`；
- 文件结束时每个未关闭块都报告开始位置；
- 嵌套块错误不能破坏后续独立语句的位置。

结束符类型不匹配统一使用 `E110`。例如 `while ... endif` 应在 `endif` 处报告“期望 `endwhile`”，随后把该 `endif` 作为 `while` 的恢复性结束符并弹栈，EOF 不重复报告 `E111`。对于 `if ... while ... endif`，报告一个 `E110` 后弹出内层 `while` 和匹配的外层 `if`。只有真正留到 EOF 的块才报告 `E111`。

### 11.3 不要静默接受文件结尾

当前扩展解析流程在找不到结束关键字时可能走到文件尾。验证器必须把这种情况转为 `E111`，并指向块开始处，而不是最后一行。

---

## 12. 阶段六：接入 Parser

### 12.1 成功路径

成功调用保持不变：

```python
program = DSLParser().parse(source)
program = ExtendedDSLParser().parse(source)
```

建议增加 keyword-only 文件名：

```python
program = DSLParser().parse(source, filename="model.dsl")
```

### 12.2 失败路径

默认 `parse()` 在验证失败时抛出第一个 `DSLSyntaxError`。它不返回 Program，也不进入 IR 生成阶段。

多错误调用使用显式入口：

```python
collector = ExtendedDSLParser().validate(
    source,
    filename="bad.dsl",
    max_errors=20,
)

if collector.has_errors:
    print(collector.report(), file=sys.stderr)
```

### 12.3 Parser 状态重置

每次 `parse()` 前确认 builder、变量表、循环栈和标签计数器处于干净状态。诊断集成不能让同一个 parser 实例第二次解析时继承上次失败状态。

---

## 13. 阶段七：接入 CompilerDriver 与 CLI

### 13.1 收窄解析器回退

当前 `_parse()` 会捕获扩展解析器的任意异常后尝试基础解析器。修改时应遵循：

- 用户语法错误直接返回，不回退；
- 由于扩展解析器继承基础语法，编译器驱动优先统一使用 `ExtendedDSLParser`；如需保留两种模式，使用显式配置选择，不扫描关键字猜测；
- 只有明确的“解析器不适用”状态允许回退；
- `_parse()` 不捕获 `IndexError`、`AssertionError` 等实现错误。

还要修改外层 `CompilerDriver.compile()`：只把 `DSLSyntaxError` 等已知用户错误转换为失败结果，意外异常继续抛出，让库测试直接失败。CLI 顶层负责在正常模式输出 `internal compiler error` 并返回 2；只有显式调试模式打印 traceback。

### 13.2 CompileResult

建议给 `CompileResult` 增加 `diagnostics: list[DSLSyntaxError]`、`diagnostic_limit_reached: bool` 和 `diagnostic_limit: int`。结构化诊断供 CLI 最终渲染，后两个字段把错误抑制状态从 collector 传到 renderer；`errors` 继续保存完整的无颜色文本，兼容现有调用方。`diagnostics` 与 `errors` 同时存在时，CLI 只渲染 `diagnostics`，避免重复输出，但仍根据 `diagnostic_limit_reached` 输出 footer。

不要再次添加模糊前缀：

```text
不建议：Parse error: bad.dsl:5:1: error...
建议：  bad.dsl:5:1: error[E100]: ...
```

避免出现 `Error: Parse error: ...` 的重复层级。

### 13.3 CLI

CLI 负责：

- 把诊断写到 stderr；
- 失败返回 1；
- 把 `sys.stderr` 传给 renderer，终端交互时允许颜色；
- 输出重定向或 `NO_COLOR` 存在时关闭颜色；
- 不打印 traceback，除非用户显式启用调试模式。

建议的手工验收命令：

```bash
scratchv bad.dsl -o output.s
scratchv bad.dsl -o output.s 2> error.txt
```

检查 `error.txt` 中没有 `\x1b[` ANSI 序列。

---

## 14. 阶段八：多错误收集

### 14.1 只恢复到可信边界

基础 DSL 的可信边界是下一物理行；扩展 DSL 还包括 `else`、`endif`、`endwhile` 和 `endfor`。

不要在参数列表中盲目寻找下一个逗号后继续，因为当前解析器不是 token 流，容易把后续字符误认为新语句。

### 14.2 错误上限

建议默认最多报告 20 个真实错误。达到上限后：

- 停止继续验证；
- 设置 `collector.limit_reached = True`；
- renderer 输出一条无源码位置的 `note: error limit ...` footer；
- 转换为 `CompileResult` 时同步复制 `diagnostic_limit_reached` 和 `diagnostic_limit`；
- 标题中的错误数量仍为 20；
- 不把抑制消息放进 `errors`，也不当作第 21 个源码错误。

### 14.3 排序和去重

报告按 `(line, col, error_code)` 排序。同一位置、同一错误码、同一消息只保留一次。

---

## 15. 阶段九：修复建议

### 15.1 建议来源

```text
显式上下文提示 > 精确拼写表 > 错误码固定提示 > 无提示
```

示例：

| 输入 | 错误 | 建议 |
|------|------|------|
| `retrun x` | 未识别关键字 | `did you mean 'return'?` |
| `x = ad(a, b)` | 不支持算子 | `did you mean 'add'?` |
| `if (a > b` | 缺少右括号 | `add the missing ')'` |
| 随机未知单词 | 未识别语句 | 不猜测 |

### 15.2 误报测试

不仅要测试“应该出现建议”，也要测试“这里不应出现建议”。例如变量名与关键字相似时，不应建议把变量改成关键字。

---

## 16. 测试计划

### 16.1 快速测试

开发过程中每个小步骤运行：

```bash
python -m pytest tests/test_dsl_errors.py tests/test_dsl_validator.py -q
```

### 16.2 解析器回归

```bash
python -m pytest \
  tests/test_parser.py \
  tests/test_dsl_extended.py \
  tests/test_dsl_errors.py \
  tests/test_dsl_validator.py \
  tests/test_dsl_diagnostics_cli.py -q
```

Windows PowerShell 可把路径放在同一行执行。

```powershell
python -m pytest tests/test_parser.py tests/test_dsl_extended.py tests/test_dsl_errors.py tests/test_dsl_validator.py tests/test_dsl_diagnostics_cli.py -q
```

### 16.3 项目验证

```bash
python .Codex/harness/verify/run.py --level L1
python .Codex/harness/verify/run.py --level L2
```

如果本地专属 harness 不存在，应明确记录环境缺失，并运行仓库可用的等价检查：

```bash
make test
python scripts/build_docs_html.py --output-dir benchmark_reports/docs
```

不能因为 harness 缺失就声称 L2 已通过。

### 16.4 测试用例清单

| 类别 | 最少用例 |
|------|----------|
| 正确基础 DSL | 算术、一元算子、matmul、for |
| 正确扩展 DSL | if/else、while、嵌套块 |
| 行列定位 | 首行、中间行、缩进、CRLF、Unicode、tab |
| 语句结构 | 缺赋值号、括号、逗号、冒号、尾部垃圾 |
| 块结构 | 多余结束符、错误类型结束符、缺失结束符、重复 else |
| 算子签名 | 未知算子、参数不足、参数过多、未知 kwarg、非法值 |
| 多错误 | 3 个独立错误、级联抑制、达到上限 |
| 输出 | ANSI 开关、`NO_COLOR`、stderr、退出码 |
| 兼容性 | `DSLParseError` 捕获、原 parse 调用、正确 IR 不变 |

---

## 17. 调试指南

### 17.1 caret 偏移一列

检查：

- 内部索引是否 0-based；
- 对外 `col` 是否 1-based；
- 行号前缀宽度是否计入了源码列；
- 原始行是否被 `strip()`；
- tab 是否经过显示宽度映射。

不要通过随意加减常数修复单个样例，应先写多个不同列位置的参数化测试。

### 17.2 错误行号总是 1

确认基础解析器没有使用 `text.strip().split("\n")` 后丢弃原始索引。应对原始物理行执行 `enumerate(..., start=1)`。

### 17.3 扩展错误变成普通解析错误

检查 `CompilerDriver._parse()` 是否仍然捕获所有异常并回退。结构化用户错误不应触发回退。

### 17.4 一处错误产生很多错误

验证器可能在外层结构失败后继续执行参数检查。为每行定义“主要结构错误后停止本行”的规则。

### 17.5 正确 DSL 的 IR 改变

错误诊断改动不应改变成功语义。比较改动前后的 IRPrinter 输出和指令结构，定位验证阶段是否误改了 parser 或 builder 状态。

---

## 18. 代码评审清单

### 正确性

- [ ] 所有行列均为 1-based，并有边界测试。
- [ ] 原始源码行在位置计算前未被破坏。
- [ ] 有错误时不生成或返回部分 IR。
- [ ] 扩展语法错误不会被回退逻辑覆盖。
- [ ] 用户错误不会泄漏 Python 内部异常。
- [ ] 错误码与设计文档一致。

### 兼容性

- [ ] `DSLParser().parse(text)` 成功调用保持不变。
- [ ] `DSLParseError` 旧捕获方式仍有效。
- [ ] `format_error()` 现有参数仍可使用。
- [ ] 正确 DSL 的 IR 与基线一致。

### 用户体验

- [ ] 消息说明问题而不是描述实现细节。
- [ ] caret 指向真正错误 token。
- [ ] 修复建议可信且没有明显误报。
- [ ] 非 TTY 输出不含 ANSI。
- [ ] 多错误报告没有明显级联噪音。

### 测试

- [ ] 单元、解析器集成、CompilerDriver 和 CLI 均有覆盖。
- [ ] LF、CRLF、Unicode、tab 和空文件有覆盖。
- [ ] 错误上限和去重有覆盖。
- [ ] L1、L2 或明确记录的等价验证已执行。

---

## 19. 12 周开发计划

| 周次 | 目标 | 可验收产物 |
|------|------|------------|
| W1 | 熟悉 DSL、IRBuilder 和当前错误路径 | 调研笔记、3 个基线错误样例 |
| W2 | 锁定纯文本输出和错误码 | 失败测试、输出规范 |
| W3 | 实现并测试 SourceBuffer | LF/CRLF/Unicode/tab 单元测试 |
| W4 | 统一异常继承和兼容导出 | `DSLParseError` 兼容测试 |
| W5 | 实现基础语句结构验证 | `E100`–`E103` 测试 |
| W6 | 实现算子签名验证 | `E200`–`E203` 测试 |
| W7 | 实现 if/while/for 块栈验证 | `E110`–`E112` 测试 |
| W8 | 接入基础和扩展解析器 | 正确 IR 回归、结构化首错 |
| W9 | 实现多错误恢复、去重和上限 | 3 错误样例、`limit_reached` footer 测试 |
| W10 | 接入 CompilerDriver 和 CLI | stderr、退出码、无 traceback 测试 |
| W11 | 完善颜色、建议和边界用例 | TTY/NO_COLOR、误报测试 |
| W12 | 全量验证、性能测量和文档收尾 | L2 结果、评审报告、演示样例 |

---

## 20. 交付产物

建议最终提交包含：

- 结构化错误与渲染改进；
- `SourceBuffer` 和无副作用 DSL validator；
- 基础、扩展解析器集成；
- CompilerDriver 和 CLI 集成；
- 单元、集成、端到端测试；
- 至少 10 个错误 DSL 示例及预期输出；
- 错误码参考表；
- L1/L2 或等价验证记录；
- 性能测量结果；
- 最终 self-review 报告。

---

## 21. 最终验收演示

准备一个包含三个独立错误的 `bad.dsl`：

```text
x = ad(a, b)
y = relu(x, 1)
endwhile
```

期望演示：

1. `validate()` 一次报告 3 个错误；
2. 每个错误的位置和错误码正确；
3. `ad` 获得可信的 `add` 拼写建议；
4. `relu` 参数数量错误不泄漏 `IndexError`；
5. `endwhile` 报告无匹配开始块；
6. 编译流程失败且不产生输出汇编；
7. 重定向到文件时没有 ANSI；
8. 修正三处错误后，同一程序正常生成 IR 和目标代码。

这组演示同时覆盖位置、建议、签名检查、块检查、多错误、CLI 和成功回归，是本课题最小但完整的验收闭环。
