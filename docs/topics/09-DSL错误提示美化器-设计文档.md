# 课题9：DSL错误提示美化器设计文档

> **文档类型**：技术设计 | **状态**：草案 | **版本**：v0.1
> **调研基线**：`main@997d2aa` | **调研日期**：2026-07-15
> **关联模块**：`scratchv/frontend/dsl_errors.py`、`dsl_parser.py`、`dsl_extended.py`、`scratchv/compiler.py`

---

## 1. 文档定位

本文描述 ScratchV DSL 错误提示美化器的**拟议迭代方案**，用于指导后续开发和评审。

本文不是已完成实现的说明。当前仓库已经具备独立的错误对象、文本格式化器和错误收集器，但尚未把这些能力完整接入基础 DSL 解析器、扩展 DSL 解析器和编译器驱动。文中标为“建议”“拟新增”的接口均属于设计提案。

配套文档：

- [课题9：DSL错误提示美化器](09-DSL错误提示美化器.md)：现有课程概览
- [课题9：DSL错误提示美化器开发文档](09-DSL错误提示美化器-开发文档.md)：建议开发顺序、测试方法和交付标准
- [课题1：DSL前端增强器](01-DSL前端增强器.md)：扩展 DSL 的控制流语法

---

## 2. 背景与问题

ScratchV 提供两套 DSL 解析器：

- `DSLParser`：解析逐行表达式、`return` 和 `for/endfor`。
- `ExtendedDSLParser`：在基础语法上增加 `if/else/endif` 和 `while/endwhile`。

当前解析失败时，用户通常只能得到类似下面的消息：

```text
Error: Parse error: Cannot parse line: retrun result
```

这条消息没有文件名、行号、列号和修复建议。更重要的是，编译器驱动会先尝试扩展解析器，并用宽泛的 `except Exception` 回退到基础解析器。扩展解析器产生的原始错误可能被覆盖，最终报告与真正失败位置不一致。

目标体验如下：

```text
bad.dsl:5:1: error[E100]: cannot parse statement
  5 | retrun result
    | ^~~~~~
note: did you mean 'return'?
```

好的诊断应回答四个问题：

1. 哪个文件、哪一行、哪一列出错？
2. 编译器实际发现了什么问题？
3. 哪段源码与问题直接相关？
4. 用户下一步应该怎样修复？

---

## 3. 当前实现审计

### 3.1 已有能力

| 位置 | 当前能力 | 结论 |
|------|----------|------|
| `dsl_errors.py` | `DSLSyntaxError` 保存行、列、消息、源码行、文件名、提示和错误码 | 可复用 |
| `dsl_errors.py` | `format_error()` 输出 gcc/clang 风格文本并支持 ANSI 颜色 | 可复用，需补充自动颜色策略 |
| `dsl_errors.py` | `ErrorCollector` 支持收集、上限、报告和清空 | 可复用，需明确恢复与计数语义 |
| `dsl_errors.py` | `_SUGGESTIONS` 和 `_COMMON_FIXES` 提供少量启发式建议 | 可复用，需避免误报 |
| `frontend/__init__.py` | 导出 `DSLSyntaxError`、`format_error` 和 `ErrorCollector` | 已形成部分公共接口 |
| `tests/test_dsl_errors.py` | 覆盖错误对象、颜色、格式化、收集器和工厂函数 | 单模块测试基础较好 |

### 3.2 主要差距

| 差距 | 当前表现 | 用户影响 |
|------|----------|----------|
| 解析器未集成 | `DSLParser` 和 `ExtendedDSLParser` 仍抛出 `DSLParseError` 或底层异常 | 美化器只可手工调用 |
| 原始位置丢失 | 基础解析器对每行执行 `strip()`，没有保存原始行号和缩进 | 无法可靠计算列号 |
| 扩展错误被吞掉 | `CompilerDriver._parse()` 捕获扩展解析器的所有异常后回退 | 真正错误原因可能被替换 |
| 参数错误泄漏 | 参数数量不足可能触发 `IndexError` | 用户看到 Python 内部异常 |
| 块结构不完整 | 缺少 `endif` 或 `endwhile` 时可能静默到文件结尾 | 错误程序可能产生不完整 IR |
| 多错误仅有容器 | `ErrorCollector` 能保存多个错误，但解析器没有同步和恢复机制 | 遇到首错仍无法继续 |
| 上下文不完整 | `context_lines` 没有完整源文件，只能输出空的上下文行号 | 无法显示真实前后文 |
| 颜色默认开启 | `use_color=True` 不检查 TTY 或 `NO_COLOR` | 重定向到文件时出现转义码 |
| 测试缺少端到端覆盖 | 没有“错误 DSL → 解析器 → CompileResult/CLI”测试 | 集成回归无法被发现 |

### 3.3 语义约束

基础解析器的 `_resolve()` 会把首次出现的名字创建为输入值，因此当前 DSL 没有严格的“变量未定义”错误语义。错误美化器不应在未改变语言规则的前提下报告“未定义变量”。如需该能力，应另立语言语义课题。

---

## 4. 设计目标与非目标

### 4.1 设计目标

1. 基础和扩展 DSL 的解析错误均携带稳定、准确的位置。
2. 默认库调用保持失败即停止，避免返回被错误污染的 IR。
3. 显式验证模式可以一次报告多个相互独立的错误。
4. 所有用户输入错误转换为结构化诊断，不暴露 `IndexError`、`KeyError` 等实现异常。
5. 纯文本输出稳定，适合单元测试、CI、日志和文件重定向。
6. 现有 `DSLParser().parse(text)` 的成功路径和 IR 结果保持兼容。
7. 错误码、位置规则和恢复规则可测试、可扩展。

### 4.2 非目标

本课题不包含：

- 重写完整词法器或引入第三方解析框架；
- 修改 DSL 的变量定义、类型检查或算子语义；
- 自动修改用户源码；
- LSP、编辑器插件或 IDE 实时诊断；
- JSON/SARIF 报告协议；
- 修改 ONNX 解析器的错误体系；
- 在语法错误存在时生成或执行部分 IR。

---

## 5. 设计原则

### 5.1 先验证，后生成 IR

当前解析器在读取源码的同时调用 `IRBuilder`。如果在错误后强行继续，builder 中可能残留半个循环或基本块，后续错误也容易成为连锁误报。

因此多错误模式采用“两阶段”策略：

```text
源文件 ──▶ 轻量语法验证 ──▶ 0 个错误？ ──是──▶ 现有 IR 生成流程
                  │
                  └────────否──▶ ErrorCollector ──▶ 格式化报告
```

验证阶段只检查可从源码直接确定的规则，不创建 IR。只有验证通过后才进入现有解析和 IR 构建流程。

### 5.2 精确错误优先于大量错误

多错误收集不是越多越好。一个缺失的右括号可能导致同一行出现多个派生错误。验证器应在确认一条语句的外层结构错误后停止分析该语句，只在下一条独立语句继续。

### 5.3 公共接口渐进兼容

现有 `dsl_errors.py` 的公开字段和常用函数继续保留。新增字段必须提供默认值；成功解析的调用方式不变；旧的 `DSLParseError` 导入路径继续有效。

---

## 6. 总体架构

```text
┌──────────────────────────────────────────────────────────────┐
│                     DSL source / filename                    │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ SourceBuffer                                                 │
│ 保留原始文本、物理行、文件名；负责 offset/line/column 映射     │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ DSLValidator                                                 │
│ 行级语法、算子签名、控制流块栈；不调用 IRBuilder               │
└───────────────┬───────────────────────────────┬──────────────┘
                │ 诊断                          │ 无诊断
                ▼                               ▼
┌────────────────────────────┐     ┌───────────────────────────┐
│ ErrorCollector             │     │ DSLParser / Extended...   │
│ 去重、排序、上限、抑制提示  │     │ 复用现有 IR 生成流程       │
└───────────────┬────────────┘     └─────────────┬─────────────┘
                ▼                                ▼
┌────────────────────────────┐     ┌───────────────────────────┐
│ DiagnosticRenderer         │     │ Program                   │
│ ANSI / plain text          │     │ 后续优化与代码生成         │
└────────────────────────────┘     └───────────────────────────┘
```

### 6.1 组件职责

| 组件 | 职责 | 不负责 |
|------|------|--------|
| `SourceBuffer` | 保存原始源码、获取行文本、映射位置 | 判断语法是否正确 |
| `DSLValidator` | 产生结构化错误并执行有限同步 | 创建 IR、修复源码 |
| `DSLSyntaxError` | 表示一个诊断 | 读取文件、打印到终端 |
| `ErrorCollector` | 收集、去重、排序和限制诊断 | 决定解析恢复点 |
| `format_error()` | 把单个诊断渲染为文本 | 推断语言语义 |
| `DSLParser` | 验证通过后生成 IR | 在错误状态下返回部分 Program |
| `CompilerDriver` | 选择解析器并把诊断交给 `CompileResult` | 吞掉异常后盲目回退 |

---

## 7. 数据模型设计

### 7.1 SourceBuffer（拟新增内部类型）

```python
@dataclass(frozen=True)
class SourceBuffer:
    text: str
    filename: str = "<dsl>"

    def line_text(self, line: int) -> str: ...
    def line_count(self) -> int: ...
```

约束：

- 行号和列号均从 1 开始。
- 列号以 Python 字符索引为基础，即按 Unicode code point 计数。
- 保留原始行，不在位置计算前执行 `strip()`。
- 渲染器把制表符按 4 列展开，但错误对象中的列号仍指向原始字符位置。caret 的显示列必须逐字符换算：普通字符增加 1 列，tab 增加到下一个 4 列制表位，不能直接把 raw column 当作显示空格数。
- Windows `\r\n` 与 Unix `\n` 均映射到相同的物理行。

### 7.2 DSLSyntaxError（兼容扩展）

建议保留现有必需字段，并增加可选的结束列：

```python
@dataclass
class DSLSyntaxError(DSLParseError):
    line: int
    col: int
    message: str
    source_line: str = ""
    filename: Optional[str] = None
    fix_hint: Optional[str] = None
    error_code: Optional[str] = None
    end_col: Optional[int] = None  # 1-based, exclusive
```

兼容策略：

- `DSLSyntaxError` 继承或等价兼容 `DSLParseError`，保留旧代码的捕获行为。
- 新字段放在已有字段之后并提供默认值。
- `end_col is None` 时继续使用当前 token 长度估算。
- `__str__()` 始终输出无颜色文本，避免异常字符串携带终端控制码。

### 7.3 验证结果

多错误收集使用显式验证入口，避免改变 `parse()` 的返回类型：

```python
def validate(
    self,
    text: str,
    *,
    filename: Optional[str] = None,
    max_errors: int = 20,
) -> ErrorCollector: ...
```

调用约定：

- `validate()` 不生成 IR。
- `collector.has_errors` 为真时，调用者不得继续编译。
- `parse()` 可先调用同一验证逻辑；发现错误时抛出第一个 `DSLSyntaxError`。
- CLI 或教学工具需要一次展示多个错误时，先调用 `validate()`，再调用 `report()`。

### 7.4 编译结果中的结构化诊断

CLI 需要根据实际 stderr 是否为 TTY 决定颜色，因此 `CompilerDriver` 不能只返回已经格式化的字符串。建议为 `CompileResult` 增加兼容字段：

```python
@dataclass
class CompileResult:
    # 已有字段保持不变
    errors: list[str]
    diagnostics: list[DSLSyntaxError] = field(default_factory=list)
    diagnostic_limit_reached: bool = False
    diagnostic_limit: int = 20
```

数据流约定：

- `diagnostics` 保存结构化错误，是 CLI 渲染的首选数据源。
- `diagnostic_limit_reached` 和 `diagnostic_limit` 把 collector 的抑制状态传到 CLI，保证 renderer 能输出 footer。
- `errors` 保留无 ANSI 文本，兼容现有库调用方和序列化逻辑。
- 两个字段同时存在时，CLI 只渲染 `diagnostics`，不得重复打印 `errors`。
- 只有无法表示为 DSL 诊断的预期业务错误才仅写入 `errors`。
- 错误上限属于报告元数据，不伪造为带 `line=0, col=0` 的 `DSLSyntaxError`。

---

## 8. 错误分类与错误码

错误码用于测试、文档检索和未来扩展。消息文字可以改进，错误码语义必须保持稳定。

| 错误码 | 分类 | 触发条件 | 建议高亮 |
|--------|------|----------|----------|
| `E100` | 未识别语句 | 整行不符合任何 DSL 语句 | 第一个非空 token |
| `E101` | 括号不配对 | 调用或条件缺少左右括号 | 缺失点或多余括号 |
| `E102` | 缺少分隔符 | 赋值号、逗号或控制流冒号缺失 | 邻近 token |
| `E103` | 非法标识符 | 目标名或循环变量不符合规则 | 完整标识符 |
| `E110` | 块结束符不匹配 | `endif`、`endwhile`、`endfor` 无匹配开始，或与当前栈顶块类型不一致 | 结束关键字 |
| `E111` | 块未闭合 | 到文件结尾仍存在打开的块 | 对应开始关键字 |
| `E112` | `else` 位置错误 | 没有对应 `if` 或同一块重复 `else` | `else` token |
| `E200` | 不支持的算子 | 算子名不在注册表 | 算子名 |
| `E201` | 参数数量错误 | 位置参数数量不符合算子签名 | 调用参数区 |
| `E202` | 关键字参数错误 | 未知、重复或缺失的 kwarg | 参数名 |
| `E203` | 参数值错误 | `rows`、`axis` 等需要数值但格式非法 | 参数值 |

不在本课题中使用的错误：

- “变量未定义”：现有语言把首次出现的名字视为输入值。
- 类型不匹配：当前 DSL 前端没有完整类型检查阶段。
- IR 验证错误：应由 `IRVerifier` 报告，而不是 DSL 语法层报告。

---

## 9. 验证与错误恢复

### 9.1 基础 DSL

基础 DSL 以物理行为自然同步边界。单行验证流程建议如下：

```text
跳过空行/注释
  ├─ for 语句      → 检查变量、范围和 block stack
  ├─ endfor        → 检查栈顶是否为 for
  ├─ return        → 检查关键字后是否具有非空操作数
  └─ assignment    → 检查 lhs、op、括号、参数与 kwargs
```

如果一行外层结构已经错误，验证器记录一条主要错误后直接进入下一物理行，不继续检查该行的算子参数。

### 9.2 扩展 DSL

扩展语法使用块栈：

```python
BlockFrame(kind="if", line=4, saw_else=False)
BlockFrame(kind="while", line=9)
BlockFrame(kind="for", line=12)
```

同步规则：

1. 普通语句失败后跳到下一物理行。
2. `else` 只与最近且尚未出现 `else` 的 `if` 匹配。
3. 结束关键字与栈顶匹配时正常弹栈；栈为空时报告 `E110` 后继续。
4. 栈非空但类型不匹配时，在当前结束符报告一个 `E110`，消息同时写明“发现什么”和“期望什么”。为避免 EOF 级联：若该结束符能匹配更外层块，则弹出到该外层块（含它）；若栈中没有可匹配块，则把它作为栈顶块的恢复性结束符并弹出栈顶。被恢复弹出的块不再报告 `E111`。
5. 到达文件尾时，只为仍留在栈中的每个未闭合块报告 `E111`。
6. 达到错误上限后停止验证，并把 `collector.limit_reached` 设为真；抑制提示作为报告 footer 输出，不加入 `errors` 列表。

恢复示例：

| 源码结构 | 诊断序列 | 恢复后栈 |
|----------|----------|----------|
| `while ... endif` | 当前 `endif` 报 1 个 `E110`：期望 `endwhile` | 弹出 `while`，EOF 不再报 `E111` |
| `if ... while ... endif` | 当前 `endif` 报 1 个 `E110`：应先出现 `endwhile` | 弹出 `while` 和匹配的 `if` |
| `if ... while ... EOF` | `while`、`if` 的开始位置各报 1 个 `E111` | 文件结束 |

### 9.3 去重和级联抑制

建议用以下键去重：

```text
(filename, line, col, error_code, message)
```

同一行最多报告一个结构错误和一个独立的算子签名错误。由结构错误直接导致的后续错误不再报告。

---

## 10. 修复建议策略

修复建议应当保守。错误建议错误时，比没有建议更影响用户判断。

优先级从高到低：

1. 解析器在明确上下文中提供的 `fix_hint`；
2. 精确拼写表，例如 `retrun → return`；
3. 基于错误码的固定建议，例如 `E101 → add the missing ')'`；
4. 没有足够信息时不输出建议。

建议规则：

- 拼写建议仅比较同类关键字或算子名。
- 不对任意源代码单词做全局替换建议。
- 不建议会改变程序语义的操作。
- 提示文字不参与控制流判断；逻辑只依赖错误码和结构化字段。

---

## 11. 渲染设计

### 11.1 纯文本格式

```text
{filename}:{line}:{col}: error[{code}]: {message}
 {line_width} | {source_line}
 {padding} | {caret_and_tildes}
note: {fix_hint}
```

约束：

- 文件名未知时显示 `<dsl>`，不输出空位置前缀。
- 行号宽度按本次报告的最大行号计算。
- `end_col` 存在时按跨度绘制 `^~~~`；不存在时才估算 token 长度。
- `source_line` 为空时不显示源码和 caret。
- 测试和 `str(error)` 强制无颜色。

### 11.2 ANSI 颜色

保留 `format_error(err, use_color: bool)` 作为兼容的纯格式化函数；另提供面向输出流的 renderer：

```python
def render_error(
    err: DSLSyntaxError,
    *,
    stream: TextIO,
    use_color: Optional[bool] = None,
) -> str: ...
```

`render_error()` 的颜色规则：

- `True`：强制开启；
- `False`：强制关闭；
- `None`：仅当传入的 `stream.isatty()` 为真且未设置 `NO_COLOR` 时开启。

颜色只改变显示，不得改变可见字符内容、错误码或换行数量。

### 11.3 多错误报告

`ErrorCollector.report()` 应区分真实错误和抑制消息：

```text
--- 3 error(s) found ---
...
note: error limit (3) reached; further errors suppressed
```

标题中的数量只统计真实源码错误。错误上限提示在验证阶段来自 `ErrorCollector.limit_reached`，进入编译驱动后复制到 `CompileResult.diagnostic_limit_reached`；它不占用错误位置，也不计入 `error_count`。

---

## 12. 解析器与编译器驱动集成

### 12.1 基础解析器

建议的成功路径保持不变：

```python
program = DSLParser().parse(source)
```

拟议扩展：

```python
program = DSLParser().parse(source, filename="model.dsl")
collector = DSLParser().validate(source, filename="model.dsl")
```

`_parse_line()` 应接收带原始位置的行上下文，而不是只接收 `strip()` 后的字符串。

### 12.2 扩展解析器

`ExtendedDSLParser` 与基础解析器共享 `SourceBuffer`、算子签名验证和错误工厂，只增加控制流块规则。不要复制一套错误消息和错误码。

Validator 与 Parser 还必须共享语法事实源，例如 `STATEMENT_PATTERNS`、`OP_SIGNATURES` 和关键字集合。Parser 的执行分派可以保留独立 handler，但其算子名集合必须由测试断言与 `OP_SIGNATURES` 完全一致，避免“验证通过、执行失败”或“Parser 接受、Validator 拒绝”。

### 12.3 CompilerDriver

当前“扩展解析失败后捕获所有异常并尝试基础解析器”的策略需要移除或收窄：

- 由于 `ExtendedDSLParser` 已继承基础语法，编译器驱动可统一使用扩展解析器；如果未来保留两种模式，则必须由显式配置选择，不能扫描关键字猜测，也不能通过异常回退选择；
- `DSLSyntaxError` 属于用户输入错误，必须原样保留，不得触发回退；
- 只有明确表示“该解析器不适用”的内部信号才允许回退；
- `CompileResult.diagnostics` 保存结构化诊断，`diagnostic_limit_reached`/`diagnostic_limit` 保存抑制状态，`errors` 保存兼容的无 ANSI 文本；
- CLI 把 `sys.stderr` 传给 renderer，由 renderer 根据该流的 TTY 状态决定颜色。

`CompilerDriver.compile()` 的异常边界也必须同步调整：它只捕获 `DSLSyntaxError` 等已知用户输入错误并生成失败的 `CompileResult`。`IndexError`、`AssertionError` 等意外异常不应被包装成普通 Parse error；库调用时让它们抛出以便测试发现，CLI 顶层再把它们转换为明确的 `internal compiler error` 和退出码 2。正常模式不打印 traceback，显式调试模式才打印。

这样可以避免扩展语法错误被基础解析器的次生错误覆盖。

---

## 13. 测试设计

### 13.1 单元测试

| 测试对象 | 重点 |
|----------|------|
| `SourceBuffer` | LF/CRLF、空文件、末尾换行、Unicode、制表符 |
| `DSLSyntaxError` | 兼容构造、继承关系、`end_col`、无颜色 `str()` |
| `format_error()` / renderer | 对齐、跨度、无文件名、无源码、指定 stream 的颜色自动策略 |
| `ErrorCollector` | 去重、排序、错误上限、真实错误计数、清空 |
| 建议规则 | 精确命中、大小写、无误报、显式提示优先 |

### 13.2 解析器集成测试

至少覆盖：

- 无法识别的语句；
- `retrun` 等拼写错误；
- 缺少左右括号；
- 不支持的算子；
- 一元、二元和带 kwarg 算子的参数数量错误；
- 多余 `endfor`、`endif`、`endwhile`；
- 缺失块结束符；
- `else` 无匹配或重复；
- 嵌套 `if/while/for` 的恢复；
- 同一文件多个独立错误；
- 正确 DSL 的 IR 与改动前一致。

### 13.3 端到端测试

```text
bad.dsl → CompilerDriver.compile() → CompileResult(success=False)
```

断言：

- 包含文件名、行、列、错误码和源码行；
- 不包含 Python traceback、`IndexError` 或 `KeyError`；
- 扩展语法错误没有被基础解析器错误覆盖；
- CLI 失败退出码为 1；
- 重定向输出时没有 ANSI 转义码。

### 13.4 快照测试原则

只对纯文本格式使用快照。每个快照应尽量只包含一个概念，避免所有错误共用一个巨大 golden 文件。错误码和位置做精确断言，建议文本可以单独断言。

---

## 14. 兼容性与迁移

| 风险 | 兼容措施 |
|------|----------|
| 调用方捕获 `DSLParseError` | 让 `DSLSyntaxError` 保持其子类或兼容别名关系 |
| 调用方只传 `text` | `filename` 使用 keyword-only 可选参数 |
| 测试依赖无颜色 `str(e)` | `__str__()` 固定调用 `use_color=False` |
| 正确程序 IR 发生变化 | 增加改动前后 IR 快照或结构对比测试 |
| 旧格式没有错误码 | 错误码作为新增内容，文档标明输出格式版本变化 |
| 外部调用直接使用 `format_error()` | 保留现有参数，新增行为通过可选参数提供 |

---

## 15. 性能要求

错误诊断不是编译热点，但验证阶段不能明显拖慢批量基准：

- 源码扫描时间复杂度为 `O(n)`，`n` 为源码字符数。
- 每行最多进行常数次正则匹配和算子表查询。
- 拼写建议优先使用字典精确匹配，不对所有词执行无界编辑距离搜索。
- 对 `benchmarks/cases/*.dsl` 执行 5 次预热，再执行 10 组测量；每组依次解析全部用例 100 次，使用组耗时中位数。相同机器、Python 版本和进程配置下，“验证 + IR 生成”的中位数目标不超过原解析流程的 1.5 倍。
- 错误数量达到上限后立即停止进一步分析。

报告必须同时给出基线、改动后中位数、倍率和测试环境；不能仅凭主观判断声明达成。

---

## 16. 验收标准

满足以下条件时，课题可判定完成：

1. 基础和扩展解析器的用户输入错误均转换为 `DSLSyntaxError`。
2. 每个错误至少包含文件名、1-based 行列、稳定错误码和可读消息。
3. 至少覆盖第 8 节列出的 `E100`、`E101`、`E110`、`E111`、`E200`、`E201`。
4. 显式验证模式能从一个文件报告至少 3 个独立错误。
5. 有语法错误时不返回可执行的部分 IR。
6. `CompilerDriver` 不再用基础解析器错误覆盖扩展解析器错误。
7. 纯文本输出不含 ANSI，终端自动颜色遵守 TTY 和 `NO_COLOR`。
8. 现有正确 DSL 示例、解析器测试和 L2 验证全部通过。
9. 新增单元、集成和端到端测试均通过。
10. 文档说明如何增加新的错误码、验证规则和修复建议。

---

## 17. 风险与权衡

| 选择 | 收益 | 代价 |
|------|------|------|
| 验证与 IR 生成分离 | 多错误收集安全，不污染 builder | 部分语法规则会被检查两次 |
| 保留正则解析 | 改动小，适合教学课题 | 复杂语法扩展能力有限 |
| 稳定错误码 | 测试与文档可长期引用 | 新错误分类需要谨慎评审 |
| 保守修复建议 | 降低误导用户的概率 | 可提供的建议数量较少 |
| 不做部分 IR | 保证下游阶段输入可信 | 无法展示错误后的局部编译结果 |

如果未来 DSL 语法明显增长，应把本方案视为迁移到真正 tokenizer/parser 之前的过渡层，而不是无限扩展行级正则验证器。

---

## 18. 待后续课题讨论

以下方向保留为未来工作，不作为本课题验收项：

- 统一 ONNX、DSL、IR 验证器的 `Diagnostic` 协议；
- JSON、SARIF 和机器可读错误输出；
- LSP 诊断与编辑器下划线；
- 跨行 SourceSpan 和相关位置（related locations）；
- 基于算子注册表自动生成参数诊断；
- 国际化错误消息；
- 将 DSL 正式迁移到 token 流和语法树。
