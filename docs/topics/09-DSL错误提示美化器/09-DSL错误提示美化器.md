# 课题9：DSL错误提示美化器

> **难度**：中 | **类型**：项目实战 | **源文件**：`scratchv/frontend/dsl_errors.py` | **行数**：~400
> **状态**：✅ 已完成

---

## 概述

改进DSL前端出错时的错误报告，显示错误位置（行号、列号）、出错代码行、标记错误位置，并给出修复建议。

---

## 理解背景

### 是什么？

DSL 错误提示美化器（`dsl_errors.py`）让 ScratchV 的 DSL 解析器在遇到语法错误时，产生 **gcc/clang 风格** 的错误信息——准确指出哪个文件的第几行第几列出错，高亮问题 token，并给出修复建议。

效果对比：
```
# 普通报错（看不清）
SyntaxError: unexpected token

# 美化后的报错（gcc 风格，一目了然）
test.dsl:5:12: error: unexpected token 'retrun'
  5 | result = retrun(x)
    |          ^~~~~~~
note: did you mean 'return'?
```

### 为什么？

写代码时，编译器报错的质量直接影响开发效率。好的错误信息告诉你三件事：

1. **在哪出错**（文件名:行号:列号）
2. **出了什么错**（一句话说清楚）
3. **怎么修**（修复建议）

gcc 和 clang 的错误格式已成为业界标准。这个模块让 ScratchV 的 DSL 错误信息也达到这个水准。

### 核心概念

#### 1. DSLSyntaxError

一个增强的异常类，携带精确的位置信息：

```python
@dataclass
class DSLSyntaxError(Exception):
    line: int              # 行号（从 1 开始）
    col: int               # 列号（从 1 开始）
    message: str           # 错误描述
    source_line: str       # 出错的源代码行
    filename: Optional[str] # 文件名
    fix_hint: Optional[str] # 修复建议
    error_code: Optional[str] # 错误码
```

#### 2. ErrorCollector

收集多个错误后统一输出（不会遇到第一个错就停）：

```python
collector = ErrorCollector(filename="test.dsl")
try:
    parser.parse(source)
except DSLSyntaxError as e:
    collector.add(e)
# ... 继续解析找更多错 ...
collector.report()  # 一次性输出所有错误
```

默认最多收集 20 个错误，超过后会提示 "further errors suppressed"。

#### 3. 拼写建议数据库

内置了常见拼写错误的修复建议：

| 写错了 | 建议 |
|--------|------|
| `retrun` | `did you mean 'return'?` |
| `endiff` | `did you mean 'endif'?` |
| `sofmax` | `did you mean 'softmax'?` |
| `matmal` | `did you mean 'matmul'?` |

### 一步步使用

#### Step 1: 创建和格式化一个错误

```python
from scratchv.frontend.dsl_errors import make_error, format_error

err = make_error(
    line=5,
    col=12,
    message="unexpected token 'retrun'",
    source_line="result = retrun(x)",
    filename="test.dsl",
    fix_hint="did you mean 'return'?",
)

print(format_error(err))
```

输出：
```
test.dsl:5:12: error: unexpected token 'retrun'
  5 | result = retrun(x)
    |          ^~~~~~~
note: did you mean 'return'?
```

#### Step 2: 使用 ErrorCollector 收集多个错误

```python
from scratchv.frontend.dsl_errors import ErrorCollector

collector = ErrorCollector(filename="my_program.dsl", max_errors=10)

# 发现错误时
collector.add_error(
    line=3, col=8,
    message="undefined variable 'x'",
    source_line="y = add(x, 1)",
    fix_hint="variable 'x' used before assignment",
)

# 最后统一输出
print(collector.report())
```

#### Step 3: 在解析器中使用

```python
collector = ErrorCollector(filename=filename)
try:
    ast = parser.parse(source_code)
except DSLSyntaxError as e:
    collector.add(e)

if collector.has_errors:
    print(collector.report(), file=sys.stderr)
    sys.exit(1)
```

---

## 详细任务

1. 修改词法/语法分析器，在解析过程中记录当前行号和列号（基于字符索引）。
2. 自定义异常类`DSLSyntaxError`，包含行号、列号、错误信息、源码行内容。
3. 在解析函数中捕获异常，抛出`DSLSyntaxError`。
4. 编写错误格式化函数：输出`文件名:行:列: error: 消息`，然后打印源码行，下一行用`^`标记错误位置。
5. 根据常见错误类型提供修复建议（如"缺少右括号"、"未定义的变量"）。
6. 支持多错误收集（不提前退出），输出所有错误。
7. 使用ANSI颜色高亮错误位置和文件名。

---

## 交付产物

- 修改后的`dsl_parser.py`和错误处理模块
- 测试用例（包含各种语法错误的DSL文件）及对应的预期错误输出
- 文档：如何扩展错误类型

---

## 代码走读

### format_error 函数

```python
def format_error(err, use_color=True):
    parts = []
    # 1. 位置信息: filename:line:col: error: message
    location = f"{err.filename}:{err.line}:{err.col}: "
    parts.append(f"{location}error: {err.message}")

    # 2. 源代码行显示
    parts.append(f"  {err.line} | {err.source_line}")

    # 3. 列标记 (^~~~)
    marker = " " * (err.col + 3) + "^"
    token_len = _estimate_token_length(err.source_line, err.col - 1)
    marker += "~" * max(token_len - 1, 1)
    parts.append(marker)

    # 4. 修复建议
    hint = err.fix_hint or _compute_suggestion(err.message, err.source_line)
    if hint:
        parts.append(f"note: {hint}")

    return "\n".join(parts)
```

**关键细节**：`_estimate_token_length` 从出错列位置向后扫描连续字母/数字，确定 `~` 波浪线的长度。

---

## 动手练习

### 练习 1: 故意写错 DSL 代码

创建 `bad.dsl`：
```
a = const(3)
b = const(5)
c = add(a, b)
d = retrun(c)     # 故意把 return 写成 retrun
```

用 DSL 解析器解析它，观察错误提示。

### 练习 2: 添加新的拼写建议

在 `_SUGGESTIONS` 字典中添加你的常见拼写错误和对应的修复建议。

### 练习 3: 实现多错误收集

写一个 DSL 文件包含 3 个语法错误，用 ErrorCollector 一次性收集并输出。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **列号从 1 开始** | `col` 是 1-based（人类习惯），不是 0-based（程序习惯），和 `line` 一致 |
| **错误上限** | ErrorCollector 默认最多 20 个错误，超过后静默丢弃（避免刷屏） |
| **拼写建议的匹配** | `_compute_suggestion` 会把源代码中的每个词和 `_SUGGESTIONS` 逐词比对，不是模糊匹配 |
| **ANSI 颜色** | 彩色输出在管道到文件时会变成乱码，对有颜色的场景用 `use_color=False` |

---

## 进阶阅读

- GCC 错误格式规范：[GCC Diagnostic Message Formatting](https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Message-Formatting-Options.html)
- Rust 编译器的错误信息设计：[Rust Compiler Error Index](https://doc.rust-lang.org/error-index.html)（行业最佳实践）
- 相关课题: [课题1 — DSL 前端增强器](../01-DSL前端增强器/01-DSL前端增强器.md) | [课题7 — 编译器日志增强器](../07-编译器日志增强器/07-编译器日志增强器.md)

---

## 12周每周目标

- **W1**：研究现有解析器出错时是否能得到行列号。手动构造错误DSL，观察输出。
- **W2**：修改解析器，在每次读取一行时记录行号，每匹配一个token记录列号。扩展AST节点携带位置信息。
- **W3**：自定义异常类`DSLSyntaxError`，包含行号、列号、消息、源码行。
- **W4**：在解析函数的关键位置（如期望特定token但未匹配）抛出`DSLSyntaxError`。测试捕获位置正确性。
- **W5**：编写错误格式化函数，输出`文件名:行:列: error: 消息`，并打印源码行和`^`标记。
- **W6**：为常见错误添加修复建议词典（如"缺少括号" → "你可能忘了加右括号"）。
- **W7**：集成到编译器主流程：捕获解析异常并调用格式化函数，优雅退出。
- **W8**：增加多行错误上下文（显示错误行前后各一行），使用ANSI颜色高亮。
- **W9**：处理词法错误（如非法字符）同样输出行列号。
- **W10**：实现错误收集：当有多个错误时，收集所有再一并输出（不提前退出）。
- **W11**：测试20个以上的错误用例，确保提示清晰且位置准确。
- **W12**：撰写文档：如何为新的语法规则添加位置跟踪、如何扩展错误建议。
