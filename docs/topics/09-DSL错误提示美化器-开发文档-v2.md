# ScratchV 课题 09「DSL 错误提示美化器」开发文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/frontend/dsl_errors.py`、`scratchv/frontend/dsl_parser.py`、`scratchv/frontend/dsl_extended.py`、`scratchv/frontend/__init__.py`、`scratchv/compiler.py`、`tests/`  
> 功能范围：接通 `DSLSyntaxError` → `format_error` → `ErrorCollector` → 两个解析器 → 编译驱动的完整错误报告链路  
> 取代关系：本文档（v2）取代同目录 `09-DSL错误提示美化器-开发文档.md`（v1）；接口契约、收集策略与实现结果一律以本文档为准，v1 与本文档冲突处以本文档为准。  
> 行号锚点基于 2026-09-14 仓库快照；实施时以"行号 ± 函数/代码内容"双锚点定位，若行号漂移以内容为准。

---

## 0. 接口契约

> 本节为唯一权威契约。任何实现与本节不一致视为缺陷。

### 0.1 模块依赖方向（消除循环导入的关键）

```
dsl_errors.py          ← 不 import 任何 frontend 模块（仅标准库）
    ↑
dsl_parser.py          ← from scratchv.frontend.dsl_errors import (...)
    ↑
dsl_extended.py        ← from scratchv.frontend.dsl_parser import (...)
    ↑
compiler.py
```

- `DSLParseError` 的**定义位置从 `dsl_parser.py` 迁移到 `dsl_errors.py`**；
- `dsl_parser.py` 必须 `from scratchv.frontend.dsl_errors import DSLParseError, DSLSyntaxError, ErrorCollector, ErrorCode` 并**重导出**，保证 `from scratchv.frontend.dsl_parser import DSLParseError` 继续可用（现存 `tests/test_dsl_extended.py:4` 依赖此路径）。

### 0.2 异常类契约

```python
class DSLParseError(Exception):
    """DSL 解析错误基类（向后兼容，可由调用方通用捕获）。"""
    pass


@dataclass(init=False)
class DSLSyntaxError(DSLParseError):
    line: int
    col: int
    message: str
    source_line: str = ""
    filename: Optional[str] = None
    fix_hint: Optional[str] = None
    error_code: Optional[str] = None

    def __init__(
        self,
        line: int,
        col: int,
        message: str,
        source_line: str = "",
        filename: Optional[str] = None,
        fix_hint: Optional[str] = None,
        error_code: Optional[str] = None,
        *,
        suggestion: Optional[str] = None,
    ) -> None: ...

    @property
    def suggestion(self) -> Optional[str]: ...          # fix_hint 读写别名
    @suggestion.setter
    def suggestion(self, value: Optional[str]) -> None: ...

    def __str__(self) -> str: ...   # return format_error(self, use_color=False)
```

| 字段 | 类型 | 契约 |
|------|------|------|
| `line` | `int` | 1-based 物理行号；`0` 仅用于文件级/聚合提示 |
| `col` | `int` | 1-based 字符列（code point 计数，tab 记 1 列） |
| `message` | `str` | 已格式化消息，**不含** `error:` 前缀、无尾随句点 |
| `source_line` | `str` | 出错行原文（不 strip、无换行符） |
| `filename` | `Optional[str]` | `None` 时渲染为 `<dsl>` |
| `fix_hint` | `Optional[str]` | 建议文本（渲染为 `note:`），可为 `None` |
| `suggestion` | 属性 | `fix_hint` 的读写别名；构造时 `suggestion=` 等价 `fix_hint=`（两者同时给出时以 `fix_hint` 为准） |
| `error_code` | `Optional[str]` | `E1xx`/`E2xx`/`E3xx`，渲染为 `error[E301]:` 前缀 |

兼容性硬约束：

1. 类继承自 `DSLParseError`（`except DSLParseError` 可捕获）；
2. 位置参数顺序 = 字段顺序，`DSLSyntaxError(1, 1, "msg")` 必须可用；
3. 现有关键字调用 `DSLSyntaxError(line=..., col=..., message=..., source_line=..., filename=..., fix_hint=..., error_code=...)` 全部保持；
4. `__init__` 内部调用 `Exception.__init__(self, message)` 使 `args == (message,)`。

### 0.3 错误码契约

```python
class ErrorCode:
    # 词法（lexical）
    LEX_ILLEGAL_CHAR       = "E101"
    # 语法（syntax）
    SYN_INVALID_STATEMENT  = "E201"
    SYN_INVALID_CONDITION  = "E202"
    SYN_MISSING_TERMINATOR = "E203"
    SYN_STRAY_TERMINATOR   = "E204"
    SYN_NESTED_CALL        = "E205"
    # 语义（semantic）
    SEM_UNKNOWN_OP         = "E301"
    SEM_ARITY              = "E302"
    SEM_UNKNOWN_KWARG      = "E304"
```

- 保留但不定义常量：`E206`（缺冒号）、`E303`（未定义变量）；
- 任何新码只能追加，不得改义。

### 0.4 公共函数契约

```python
def format_error(
    err: DSLSyntaxError,
    use_color: bool = True,
    context_lines: int = 0,
    show_column_marker: bool = True,
    source: Optional[str] = None,          # 新增：完整源码文本，用于真实上下文行
) -> str: ...


def make_error(
    line: int,
    col: int,
    message: str,
    source_line: str = "",
    filename: Optional[str] = None,
    fix_hint: Optional[str] = None,
    error_code: Optional[str] = None,
) -> DSLSyntaxError: ...


def suggest_spelling(
    source_line: str,
    col: int = 0,                          # 0 表示未指定，退化为全行扫描
) -> Optional[str]: ...                    # 返回如 "did you mean 'return'?"


def suggest_op(
    op: str,
    candidates: Iterable[str],
    cutoff: float = 0.6,
) -> Optional[str]: ...                    # 返回如 "did you mean 'mul'?"
```

内部函数（允许测试直接调用，但不保证跨版本稳定）：

```python
def _compute_suggestion(
    message: str,
    source_line: str,
    error_code: Optional[str] = None,      # 新增参数，默认 None
) -> Optional[str]: ...

def _identifier_at(source_line: str, col: int) -> Optional[str]: ...
def _estimate_token_length(source_line: str, col_start: int) -> int: ...  # 内部语义修正
```

### 0.5 `ErrorCollector` 契约

```python
class ErrorCollector:
    def __init__(
        self,
        filename: Optional[str] = None,
        use_color: bool = True,
        max_errors: int = 20,          # 必须 >= 1，否则 ValueError
        source: Optional[str] = None,      # 新增
        context_lines: int = 0,            # 新增
    ) -> None: ...

    @property
    def errors(self) -> list[DSLSyntaxError]: ...      # 副本
    @property
    def has_errors(self) -> bool: ...
    @property
    def error_count(self) -> int: ...                  # 仅真实错误，不含 overflow 提示
    @property
    def suppressed_count(self) -> int: ...             # 新增：被上限抑制的错误数

    def add(self, err: DSLSyntaxError) -> None: ...
    def add_error(self, line: int, col: int, message: str,
                  source_line: str = "", fix_hint: Optional[str] = None,
                  error_code: Optional[str] = None) -> None: ...
    def report(self) -> str: ...
    def report_and_exit(self, exit_code: int = 1) -> None: ...
    def clear(self) -> None: ...
```

- `report()` 无错误时返回 `""`；有错误时首行 `--- N error(s) found ---`；
- `errors` 按 `(line, col, error_code)` 排序返回；`add` 按 `(filename, line, col, error_code, message)` 去重，被抑制的错误同样参与去重；
- 发生抑制时，报告末尾追加一行：`note: error limit ({max_errors}) reached; {suppressed_count} further errors suppressed`；
- `max_errors < 1` 抛 `ValueError`（0 会导致"全部抑制但 `has_errors` 为假"的静默误判）；
- 不再向 `_errors` 注入 line=0 的"伪错误"哨兵（旧实现会污染 `error_count`）。

### 0.6 解析器契约

```python
# dsl_parser.py
class DSLParser:
    def parse(
        self,
        text: str,
        filename: Optional[str] = None,
        collector: Optional[ErrorCollector] = None,
    ) -> Program: ...

    def _parse_line(self, line: str, line_no: int = 0) -> None: ...

    # 新增内部辅助（供两个解析器共用）
    def _report_error(
        self, line_no: int, col: int, message: str,
        error_code: str, fix_hint: Optional[str] = None,
    ) -> None: ...

    def _col_of(self, line_no: int, needle: str, fallback: int = 1) -> int: ...

    def _report_unclosed_for(self) -> None: ...


# dsl_extended.py
class ExtendedDSLParser(DSLParser):
    def parse(
        self, text: str, filename: Optional[str] = None,
        collector: Optional[ErrorCollector] = None,
    ) -> Program: ...

    def _parse_if_block(self, lines: list[str], start_idx: int) -> int: ...
    def _parse_while_block(self, lines: list[str], start_idx: int) -> int: ...

    # 新增
    def _parse_block(
        self,
        lines: list[str],
        start_idx: int,
        terminators: tuple[str, ...],
        opener_kind: str,          # "if" | "while"
        opener_line: int,          # 1-based
        opener_col: int,           # 1-based
    ) -> tuple[int, Optional[str]]: ...        # (下一索引, 命中的终结符或 None)

    def _recover_after_bad_header(
        self, lines: list[str], start_idx: int, opener_kind: str,
    ) -> int: ...
```

**模式语义（两个 `parse` 一致）**：

| `collector` | 行为 |
|-------------|------|
| `None`（默认，严格模式） | 遇到第一个错误立即 `raise DSLSyntaxError`；合法输入行为与改动前逐字节一致 |
| 非 `None`（收集模式） | 错误写入 collector，跳过坏行/坏块继续解析；返回**可能不完整**的 `Program`，调用方必须先检查 `collector.has_errors` 再决定是否使用 |

### 0.7 内部状态约定

两个解析器共用以下实例状态（`__init__` / `parse` 中初始化）：

```python
self._raw_lines: list[str] = []              # text 归一化（\r\n/\r → \n）后 split("\n")，行号-1 即索引
self._filename: Optional[str] = None         # 透传给 DSLSyntaxError.filename
self._collector: Optional[ErrorCollector] = None
self._for_positions: list[tuple[int, int]] = []   # 新增：(line, col) 栈，与 _loop_stack 同步 push/pop
self._line_no: int = 0                       # 当前行（严格模式抛错时兜底）
```

- `_report_error` 的严格/收集分支：
  ```python
  def _report_error(self, line_no, col, message, error_code, fix_hint=None):
      raw = self._raw_lines[line_no - 1] if 0 < line_no <= len(self._raw_lines) else ""
      err = DSLSyntaxError(
          line=line_no, col=max(col, 1), message=message, source_line=raw,
          filename=self._filename, fix_hint=fix_hint, error_code=error_code,
      )
      if self._collector is not None:
          self._collector.add(err)
          return
      raise err
  ```
- 收集模式下调用方在 `_report_error` 之后**必须显式 `return`/跳过当前行**，避免带着半成品状态继续本行。

---

## 1. 逐文件改动清单（精确锚点）

### 1.1 `scratchv/frontend/dsl_errors.py`（444 行）

| # | 锚点（快照行号，内容锚点） | 现状 | 改动 |
|---|---------------------------|------|------|
| 1 | L17-22 imports | `enum, sys, dataclass, Optional` | 增加 `difflib`, `re`, `Iterable`（仅标准库） |
| 2 | L27-45 `Color` / `_color` | 无需变 | 不变 |
| 3 | L52-71 `_SUGGESTIONS` | 含 L63-70 `"add("`, `"mul("`, `"sub("`, `"div("`, `"matmul("` 键；`clean` 去掉括号后**永不可达** | 删除 L63-70 五个键；保留拼写键；新增 `_ARITY_HINTS: dict[str, str]`（键为裸算子名，值为 `add() requires exactly 2 arguments` 等） |
| 4 | L73-86 `_COMMON_FIXES` | 可用 | 保留；`old keys` 可继续作为兜底 |
| 5 | L90-116 `DSLSyntaxError` | 定义于 `dsl_errors.py`，继承 `Exception`，dataclass 自动 `__init__` | 上方新增 `class DSLParseError(Exception)`；改为 `@dataclass(init=False)` + 自定义 `__init__`（见 0.2）；新增 `suggestion` 属性；`Exception.__init__(self, message)` |
| 6 | L123-155 `_compute_suggestion` | `source_line.split()` 分词；`clean.strip("(){},:=* ")` 导致括号键不可达；仅按消息关键词 | 重写：`re.findall(r"[A-Za-z_]\w*", source_line)` 分词；优先 `_identifier_at(source_line, col)`；拼写库大小写不敏感精确匹配；函数签名加 `error_code=None` |
| 7 | L158-246 `format_error` | 插入符 `" " * (err.col + 3)`（未计行号位数）；`context_lines` 输出空行；filename 缺省时 `":5:12:"`；无 `source` 参数 | 按 2.1 公式重写 gutter 对齐；`col` clamp；新增 `source` 参数；filename 兜底 `<dsl>`；旧行为测试兼容（`^`/`note:` 仍存在） |
| 8 | L249-264 `_estimate_token_length` | 只认 `isalnum()`，忽略 `_` | 用 `[A-Za-z0-9_]` 扫描；起始越界返回 1；clamp 到行尾 |
| 9 | L271-406 `ErrorCollector` | 溢出时注入 line=0 伪错误；无去重；无 `suppressed_count`；无 `source` | 改为 `_suppressed: int` 计数 + 报告尾部 note；按 `(line, col, error_code, message)` 去重；新增 `source` 参数并传给 `format_error`；`error_count` 只计真实错误 |
| 10 | L413-444 `make_error` | 可用 | 签名不变；docstring 标注与 `DSLSyntaxError` 对齐 |
| 11 | 文件末尾新增 | — | `class ErrorCode`；`suggest_op`；`suggest_spelling`；`_identifier_at`；`_ARITY_HINTS` |

### 1.2 `scratchv/frontend/dsl_parser.py`（168 行）

| # | 锚点 | 现状 | 改动 |
|---|------|------|------|
| 1 | L22-24 imports | `re`、`IRBuilder`、`Value/Program` | 增加 dsl_errors 导入（见 0.1） |
| 2 | L27-28 `class DSLParseError(Exception): pass` | 本地定义 | **删除**，改为从 dsl_errors 导入 + 重导出 |
| 3 | L34-37 `__init__` | 3 个字段 | 增加 0.7 的状态字段 |
| 4 | L39-58 `parse` | `text.strip().split("\n")`（丢物理行号）；`for line in lines` 无行号；EOF `_loop_stack` 非空静默；无 filename/collector | 新签名（0.6）；`raw_lines = text.split("\n")`；索引循环 `for i, raw in enumerate(raw_lines)` 传 `i+1`；EOF 调用 `_report_unclosed_for()`；构建器重置 `_raw_lines/_filename/_collector/_for_positions` |
| 5 | L60-94 `_parse_line` | 精确串匹配 `line == "endfor"`；`DSLParseError(f"Cannot parse line: {line}")`；正则 `(\w+)\s*=\s*(\w+)\((.+)\)` 不锚定、可静默吞尾、嵌套调用被错误拆参 | 增加 `line_no` 参数；先剥内联注释（`" #"` 与扩展解析器一致）；语句分支见第 2 节 |
| 6 | L62-67 `for` 分支 | 只 push `_loop_stack` | 同步 push `_for_positions.append((line_no, col))` |
| 7 | L69-74 `endfor` 分支 | 无配对 → `raise DSLParseError(...)` | 无配对 → E204；正常 → 同步 pop `_for_positions` |
| 8 | L96-108 `_resolve` | 未定义变量自动建值 | **保持不变**（E303 保留）；在 docstring 注明 |
| 9 | L110-131 `_parse_kwargs` / `_parse_value` | 可用 | 修复轮：`_parse_kwargs(args, op, line_no, col)` 按 `OP_SIGNATURES` 校验未知 kwarg 与数值 kwarg（E304），校验失败返回 `None` |
| 10 | L133-168 `_dispatch_op` | 未知算子 `raise DSLParseError("Unsupported op: ...")`；参数不足泄漏裸 `IndexError`；参数过多静默忽略 | 改为：`op` 不在 `handlers` → E301（**先于实参解析**，避免副作用建值）；按 `_ARITY` 表校验普通实参个数 → E302；多余参数 → E302；返回 `Optional[Value]`（失败返回 `None`） |

### 1.3 `scratchv/frontend/dsl_extended.py`（379 行）

| # | 锚点 | 现状 | 改动 |
|---|------|------|------|
| 1 | L27-30 imports | `DSLParser, DSLParseError, IRBuilder, OpCode, Program, Value` | 增加 `DSLSyntaxError, ErrorCollector, ErrorCode`（从 dsl_errors 或经 dsl_parser 重导出） |
| 2 | L80-85 `__init__` | 3 个字段 | 增加 0.7 状态字段（`_for_positions` 由基类 `__init__` 提供） |
| 3 | L100-164 `parse` | 固定签名；注释剥除保留行位（可用）；顶层 `line.startswith("if ")` / `"while "`（`if(a>b)` 识别不到）；EOF 仅跳过 auto-ret，不报错 | 新签名；设置/重置状态；分发改 `re.match(r"^if\b", line)` / `r"^while\b"`；EOF 依次 `_report_unclosed_for()`、`self._report_unclosed_while()`（若栈非空） |
| 4 | L170-247 `_parse_if_block` | 内联三段重复扫描；L176-178 无定位抛错；L195-211 EOF 无 `endif` 静默收尾；L216-236 `else` 后二遇 `else` 静默；L243-244 `endif` 缺失静默 | 用 `_parse_block` 重构（见 2 节流程）；E202/E203/E204 规则见设计文档 2.5 |
| 5 | L253-311 `_parse_while_block` | 同样问题；L259-261 无定位；L288-301 EOF 无 `endwhile` 静默；L306-307 静默 | 同上 |
| 6 | L317-330 `_parse_condition` | 模式 `^(?:if\|while)\s*\(\s*(.+?)\s*(==\|...)\s*(.+?)\s*\)\s*:?\s*$` | **不变**（冒号继续可选；课题 15 才收紧） |
| 7 | L336-354 `_emit_cmp` | 未被调用（死代码） | 不删不改为本课题可选项（避免无关 diff） |
| 8 | L360-367 `_parse_line` | `endif/endwhile/else/else:` 与 `if /while ` 前缀直接 `return`（静默吞） | 签名改 `(self, line, line_no=0)`；`endif`/`endwhile` → E204 `'{tok}' without matching '{opener}'`；`else`/`else:` → E204 `'else' without matching 'if'`（块解析器不会把合法终结符路由到这里）；`if `/`while ` 前缀仍 return（由块分发处理） |
| 9 | 新增方法 | — | `_parse_block`、`_recover_after_bad_header`、`_report_unclosed_while` |

### 1.4 `scratchv/frontend/__init__.py`（13 行）

```python
from .dsl_errors import (
    DSLParseError, DSLSyntaxError, ErrorCode,
    format_error, make_error, ErrorCollector,
)
```

`__all__` 增加 `"DSLParseError"`, `"ErrorCode"`, `"make_error"`（保留现有项）。

### 1.5 `scratchv/compiler.py`

| # | 锚点 | 现状 | 改动 |
|---|------|------|------|
| 1 | L243-249 `compile` 解析段 | `except Exception as e: errors=[f"Parse error: {e}"]` | 前插 `except DSLSyntaxError as e: return CompileResult(success=False, errors=[str(e)])`；`str(e)` 已是多行 gcc 风格 |
| 2 | L325-346 `_parse` | `try: ExtendedDSLParser().parse(source) except Exception: DSLParser().parse(source)` —— **定位错误被 fallback 吞掉**，用户最终看到的是无行号的基础解析器报错 | 改为：`parse(source, filename=...)`；`except DSLSyntaxError: raise`；`except DSLParseError: 回退 DSLParser().parse(source, filename=...)`；仅对"扩展解析器不适用"的旧场景保留 fallback |

### 1.6 测试文件

| 文件 | 动作 |
|------|------|
| `tests/test_dsl_errors.py` | 追加单测（不改已有断言语义） |
| `tests/test_dsl_errors_integration.py` | **新增**（解析器错误分支 + 恢复 + golden 回归） |
| `tests/test_dsl_extended.py` | 不改；其 L228-241 `test_invalid_if_missing_parens` 依赖 `DSLParseError` 捕获，由继承关系保证通过 |
| `scratchv/ci/test_page.py` | 不改（`test_dsl_errors.py` 已有映射；新增文件可选登记） |

---

## 2. Parser 各错误分支改造清单

### 2.1 基础解析器 `dsl_parser.py`

| # | 位置（快照） | 现状 | 新行为 | 错误码 |
|---|--------------|------|--------|--------|
| B1 | L44-48 `parse` 主循环 | `for line in lines: self._parse_line(line)` | `for i, raw in enumerate(raw_lines): self._parse_line(raw.strip(), i+1)` | — |
| B2 | L50-57 EOF 检查 | `_loop_stack` 非空则跳过 auto-ret（静默） | 先 `_report_unclosed_for()`（LIFO，每条定位其 `for` 行），再走原 auto-ret 逻辑 | E203 |
| B3 | L69-74 `endfor` | 无配对抛裸 `DSLParseError` | `_report_error(line_no, col_of("endfor"), "'endfor' without matching 'for'", E204, hint="remove this line or add a matching 'for'")`；若有配对则 pop 两个栈 | E204 |
| B4 | **L83-91 赋值语句（含嵌套调用）** | `m = re.match(r"(\w+)\s*=\s*(\w+)\((.+)\)", line)`；不匹配即抛无定位异常；不锚定可吞尾；`c = add(mul(a,b), d)` 被拆成 `mul(a`、`b)`、`d` 三个"变量"**静默生成错误 IR** | 先剥内联注释；用锚定正则 `r"^(\w+)\s*=\s*(\w+)\s*\((.*)\)\s*$"`；实参文本 `args`：含 `(` → E205（col=内层 `(`，hint=`assign the inner call to a temporary variable first`）；含 `)` 而括号不平衡 → E201（hint=`missing opening '('`）；括号不平衡（`(` 多于 `)`）→ E201（hint=`missing closing ')'`）；无匹配 → E101/E201（B7） | E205/E201 |
| B5 | L84-91 实参解析 | `args.split(",")` 直接喂 `_dispatch_op` | 保持不变；空实参留给 B8 的 E302（`add()` → got 0） | — |
| B6 | L165-167 未知算子 | `raise DSLParseError(f"Unsupported op: {op}")` 且已解析实参（副作用建值） | 在 `_parse_kwargs/_resolve` **之前**判 `op not in handlers` → `_report_error(line_no, col_base + m.start(2), f"unknown operation '{op}'", E301, hint=suggest_spelling(...) or suggest_op(op, handlers))`，返回 `None` | E301 |
| B7 | L86 无匹配兜底 | `Cannot parse line` 无定位 | 扫描非法字符 `re.search(r"[^A-Za-z0-9_(),:=.+\-*/#%\s]", line)`：命中 → E101（col=字符位置，message=`unexpected character '{ch}'`，hint=`remove or replace '{ch}'`）；否则 → E201（col=语句首字符，hint 由括号平衡/`_compute_suggestion` 链给出） | E101/E201 |
| B8 | L133-164 `_dispatch_op` 参数校验 | `resolved[0], resolved[1]` 越界 → 裸 `IndexError`；多余实参静默忽略 | 新增模块级 `_ARITY: dict[str, int]`（add/sub/mul/div=2；neg/exp/relu/gelu=1；dot/matmul=2；softmax/maxpool=1）；`len(resolved) != _ARITY[op]` → `_report_error(line_no, col_of_op, f"{op}() expects {n} argument(s), got {m}", E302, hint=_ARITY_HINTS.get(op))`，返回 `None` | E302 |
| B9 | L93-94 结果登记 | `self._vars[dest_name] = result` | `result is None`（错误已报）时 `return`，不登记 dest，避免污染后续解析 | — |

**严格模式下的抛出点**：B3/B4/B6/B7/B8 经 `_report_error` 在 `collector is None` 时抛 `DSLSyntaxError`，异常即首错。

### 2.2 扩展解析器 `dsl_extended.py`

| # | 位置（快照） | 现状 | 新行为 | 错误码 |
|---|--------------|------|--------|--------|
| E1 | L137-150 顶层分发 | `startswith("if ")` / `"while "` | `re.match(r"^if\b", line)` / `r"^while\b"`，使 `if(a>b)` 走条件解析并报 E202 而非 E201 | — |
| E2 | L175-178 `if` 头非法 | `raise DSLParseError(f"Invalid if condition: {line}")` | `_report_error(opener_line, opener_col, f"invalid condition in 'if'; expected 'if (<expr>) <op> (<expr>):'", E202, hint=...)`；`_recover_after_bad_header` 返回下一行索引，**不产生派生错误**（括号不平衡时 hint 改为 `missing closing ')'`） | E202 |
| E3 | L195-211 then 分支扫描 | 三段重复循环；EOF 无 `endif` 静默补块；`endwhile` 落入 `_parse_line` 被吞 | 调 `_parse_block(lines, start+1, terminators=("else", "else:", "endif"), ...)`；返回 `None` → E203 定位 `if` 行；返回 `endwhile`（外来终结符）→ E203 且**不消费** | E203 |
| E4 | L216-236 else 分支 | 遇第二个 `else` 静默；EOF 无 `endif` 静默 | `_parse_block(terminators=("endif",))`；`else` 非法位置 → E204 跳过；EOF → E203 | E204/E203 |
| E5 | L238-241 无 else 补空块 | 正常 | 保留（`test_if_without_else_block` 的 IR 块数断言依赖） | — |
| E6 | L243-244 `endif` 消费 | `if lines[idx] == "endif": idx += 1` 否则静默 | 由 `_parse_block` 返回值驱动：`term == "endif"` 才 `return idx + 1`；`term is None` 或外来终结符时 `return idx`（不消费） | — |
| E7 | L259-261 `while` 头非法 | 同 E2 | E202 定位 `while`；`_recover_after_bad_header(kind="while")` | E202 |
| E8 | L287-301 while 体扫描 | EOF 无 `endwhile` 静默 | `_parse_block(terminators=("endwhile",))`；`None` → E203；外来 `endif` → E203 且不消费 | E203 |
| E9 | L303-311 收尾 | `while_stack.pop()` 仅在正常路径 | 错误路径也必须 pop（`try/finally` 或两处显式），保证 EOF 检查不重复报 | — |
| E10 | **L363-367 `_parse_line` 覆写** | 终结符与 `else` 直接 `return`（静默吞游离终结符） | 签名补 `line_no=0`；`endif`/`endwhile` → E204（`'{tok}' without matching '{opener}'`）；`else`/`else:` → E204（`'else' without matching 'if'`） | E204 |
| E11 | L152-162 EOF auto-ret | `_loop_stack`/`_while_stack` 非空则静默跳过 | 非空时先报 E203（`for` 由 `_report_unclosed_for`，`while` 由 `_report_unclosed_while`），再走原条件 | E203 |

### 2.3 `_parse_block` 恢复流程（伪代码，两个块函数共用）

```python
def _parse_block(self, lines, start_idx, terminators, opener_kind, opener_line, opener_col):
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        if not line:
            idx += 1
            continue
        if line in ("endif", "endwhile"):
            return idx, line                      # 不管是否本块终结符，均不消费
        if line in ("else", "else:"):
            if line in terminators:
                return idx, line
            self._report_error(idx + 1, self._col_of(idx + 1, "else"), 
                               "'else' without matching 'if'", ErrorCode.SYN_STRAY_TERMINATOR,
                               hint="remove this line or add a matching 'if'")
            idx += 1
            continue
        if line == "endfor":
            self._parse_line(line, idx + 1)       # 交基础解析器：配对/游离判定
            idx += 1
            continue
        if re.match(r"^if\b", line):
            idx = self._parse_if_block(lines, idx)
        elif re.match(r"^while\b", line):
            idx = self._parse_while_block(lines, idx)
        else:
            self._parse_line(line, idx + 1)
            idx += 1
    return idx, None
```

调用侧处理：

```python
idx, term = self._parse_block(lines, start_idx + 1, ("else", "else:", "endif"), "if", line_no, col)
if term is None:
    self._report_error(line_no, col, "missing 'endif' for 'if' opened here",
                       ErrorCode.SYN_MISSING_TERMINATOR, hint="add 'endif' to close this block")
    idx = len(lines)
elif term == "endwhile":
    self._report_error(line_no, col, "missing 'endif' for 'if' opened here",
                       ErrorCode.SYN_MISSING_TERMINATOR, hint="add 'endif' to close this block")
    # 不消费 endwhile，交外层；外层若已闭合则顶层兜底报 E204
```

`_recover_after_bad_header(lines, start_idx, kind)`：

```python
# 从 start_idx+1 起扫描；depth 统计 if/while 开块；
# depth==0 且遇到与本块匹配的终结符 → 消费并返回 idx+1；
# depth==0 且遇到其他终结符 → 返回 idx（不消费）；
# EOF → 返回 len(lines)。本函数不报错（E202 已报），避免级联。
```

---

## 3. ErrorCollector 多错误收集策略

### 3.1 收集边界

| 错误类型 | 恢复粒度 | 后续行为 |
|----------|----------|----------|
| E101/E201/E205/E301/E302 | 行级 | 跳过当前行，下一行继续 |
| E202 | 块级 | `_recover_after_bad_header` 跳到块终结符后继续 |
| E203 | 块级/EOF | 补齐标签、结束该块，继续外层 |
| E204 | 行级 | 跳过当前行 |

### 3.2 规则

1. **同一行最多一条错误**：错误分支命中即跳过该行，杜绝连锁。
2. **去重**：`(filename, line, col, error_code, message)` 五元组相同不重复加入（被抑制的错误同样参与去重）。
3. **上限**：默认 `max_errors=20`，必须 ≥1（否则 `ValueError`）；达到上限后新的（未去重的）错误只递增 `suppressed_count`，不再存储；报告末尾输出 `note: error limit ({max_errors}) reached; {N} further errors suppressed`。validator 达上限后不再提前 `break`，因此 `suppressed_count` 统计全部未报告错误。
4. **顺序**：`errors` 属性按 `(line, col, error_code)` 排序返回，不保持插入顺序。
5. **部分 Program 契约**：`collector.has_errors` 为真时，`parse()` 返回的 `Program` 仅用于诊断/继续收集，**禁止**用于代码生成；调用方（`compiler.py`）不进入收集模式。
6. **状态一致**：`for`/`while` 栈在错误路径也须 pop；`_vars[dest]` 在失败时不得登记。

### 3.3 报告格式

```
--- {error_count} error(s) found ---
{format_error(err, use_color=..., source=self.source) for err in errors}
[note: error limit ({max_errors}) reached; {suppressed_count} further errors suppressed]
```

---

## 4. `format_error` 修复清单

| # | Bug | 旧实现 | 修复 |
|---|-----|--------|------|
| F1 | 插入符偏移（未计行号位数） | `marker = " " * (err.col + 3) + "^"` | `gutter_src = f"  {err.line} \| "`；`gutter_mark = " " * (3 + len(str(err.line))) + "\| "`；`marker = gutter_mark + " " * (col-1) + "^" + "~" * (token_len-1)`；要求 `len(gutter_src) == len(gutter_mark)` |
| F2 | token 长度不含 `_`、未 clamp | `while ... .isalnum()` | 改 `[A-Za-z0-9_]`；`col_start >= len(source_line)` 返回 1；波浪线长度 `max(token_len-1, 1)` |
| F3 | `context_lines` 输出空行 | 只打行号无内容 | 新增 `source` 参数；有源码时按行切分渲染 `line-context .. line-1`；无源码则忽略 `context_lines` |
| F4 | 无 filename 时前缀以冒号开头 | `location = ":5:12: "` | `f"{err.filename or '<dsl>'}:{err.line}:{err.col}: "` |
| F5 | `col` 越界导致插入符超出行宽 | 未 clamp | `col_eff = min(max(err.col, 1), len(err.source_line) + 1)` 参与 marker 计算（header 仍显示原 col） |
| F6 | 彩色 gutter 与 marker 宽度不一致风险 | marker 独立硬编码 | marker 由**未着色**的 `gutter_mark` 构造，仅对 `^` 着色 |
| F7 | 不可达建议键 | `_SUGGESTIONS` 含 `add(` 等 | 删除；E302 用 `_ARITY_HINTS`；`_compute_suggestion` 改用标识符正则分词 |
| F8 | 空 `source_line` 时仍尝试建议 | `_compute_suggestion` 返回 None 无碍 | 保持；但 `format_error` 对空 `source_line` 不输出 marker（现状已如此） |

---

## 5. 测试文件与用例

### 5.1 `tests/test_dsl_errors.py`（追加）

| 用例 | 断言要点 |
|------|----------|
| `test_syntax_error_is_parse_error` | `issubclass(DSLSyntaxError, DSLParseError)`；`except DSLParseError` 可捕获 |
| `test_suggestion_alias` | `err.suggestion` 读写与 `fix_hint` 同步；`DSLSyntaxError(..., suggestion="x")` 生效 |
| `test_marker_alignment_single_digit` | 输入 line=2, col=5, source=`b = retrun(a, 1)` → 输出含精确行 `"    |     ^~~~~~"` |
| `test_marker_alignment_double_digit` | line=10 同 col → 精确行 `"     |     ^~~~~~"`（验证行号位数修复） |
| `test_token_length_includes_underscore` | `_estimate_token_length("foo_bar(x)", 0) == 7` |
| `test_no_filename_uses_placeholder` | 无 filename → 输出以 `<dsl>:1:1: ` 开头 |
| `test_suggestion_arity_hint` | `_compute_suggestion("add() expects 2 arguments, got 1", "a = add(1)", "E302")` → 含 `requires exactly 2 arguments` |
| `test_collector_dedup` | 两次加入相同 `(line,col,code,message)` → `error_count == 1` |
| `test_collector_suppressed_count` | `max_errors=3`，加 10 条互异 → `error_count == 3`，`suppressed_count == 7`，report 含 `7 further errors suppressed` |
| `test_collector_source_context` | 传 `source` 且 `context_lines=1` → 输出含上一行原文且不含空上下文行 |
| `test_format_context_without_source_ignored` | 不传 `source`、`context_lines=2` → 不出现空上下文行 |

### 5.2 `tests/test_dsl_errors_integration.py`（新增）

| 用例 | 输入/操作 | 关键断言 |
|------|-----------|----------|
| `test_unknown_op_location_and_suggestion` | 设计文档测试 1 | `line/col/code/hint`；marker 精确行；`str(e)` 逐字节含 header |
| `test_missing_endif_reported_at_opener` | 设计文档测试 2 | E203、定位 `if` 行；collector 模式 `error_count == 1` |
| `test_missing_endwhile` / `test_missing_endfor` | 同上（while/for） | E203 定位开块行；`endfor` 用例验证 `_loop_stack` 清空后不再 auto-ret |
| `test_stray_terminators` | 顶层裸 `endif`/`endwhile`/`endfor`/`else:` | 各 E204，col=1 |
| `test_invalid_condition_is_e202_and_parse_error` | `if a > b:` | `isinstance(e, DSLParseError)`；code E202 |
| `test_arity_and_nested_call` | `a = add(1)`；`c = add(mul(a,b), d)` | E302 got 1；E205 col=内层 `(` |
| `test_multi_error_collection_and_recovery` | 设计文档测试 3 | `error_count == 4`；顺序；`--- 4 error(s) found ---`；严格模式首错 E302 |
| `test_legal_dsl_zero_diagnostics` | `examples/**/*.dsl` + `benchmarks/cases/*.dsl` | 收集模式 0 错误；`Program.dump()` 与 `git show HEAD` 基线一致（可用改动前输出作 golden） |
| `test_parse_default_signature_compat` | `DSLParser().parse(text)` / `ExtendedDSLParser().parse(text)` | 旧调用方式可用；合法输入无异常 |

### 5.3 回归

- `tests/test_dsl_extended.py` 全部通过（重点 L228-241 `test_invalid_if_missing_parens`）；
- `tests/test_parser.py`、`tests/test_cfg_builder.py`、`tests/test_ir_verifier.py`、`tests/test_llvm_codegen.py` 中所有 `DSLParser/ExtendedDSLParser` 调用不受影响；
- `make test` 全量通过。

---

## 6. 实施顺序

1. **`dsl_errors.py` 独立改造**（异常迁移/ErrorCode/format 修复/collector），跑 `pytest tests/test_dsl_errors.py`；
2. **`dsl_parser.py`** 接入行号与语句级错误分支，跑 `pytest tests/test_parser.py tests/test_dsl_errors_integration.py -k "op or arity or nested"`；
3. **`dsl_extended.py`** 块重构（`_parse_block` 抽取 → E202 → E203 → E204），跑扩展测试与集成测试；
4. **`compiler.py` + `__init__.py`** 集成，CLI 冒烟；
5. **补全测试**（单测 + golden 回归），全量 `make test`；
6. **L2 验证**：`python .claude/harness/verify/run.py --level L2`；
7. **文档回填**：设计文档 2.2 表状态列、错误码扩展指南。

依赖关系：步骤 1 必须先于 2/3（异常基类迁移）；2 与 3 可并行（同文件不同区）；4 依赖 2/3。

---

## 7. 验收标准

- [ ] `DSLSyntaxError` 是 `DSLParseError` 子类；`from scratchv.frontend.dsl_parser import DSLParseError` 仍可用；`DSLSyntaxError(1, 1, "msg")` 与全部旧关键字调用可用。
- [ ] 两个解析器 `parse(text)` 默认严格模式：首错抛 `DSLSyntaxError`，携带 `line/col/message/source_line/filename/fix_hint/error_code`。
- [ ] `parse(text, collector=...)` 收集模式可报告 ≥4 条错误且能恢复（设计文档测试 3 逐字节通过）。
- [ ] 插入符与源码列严格对齐：单位/双位数行号用例均通过；`_SUGGESTIONS` 中 `add(` 类键已移除且 E302 建议可达。
- [ ] 缺 `endif`/`endwhile`/`endfor` 均报 E203 且定位开块行；游离终结符报 E204。
- [ ] 合法 DSL 全量 golden：`examples/**/*.dsl`、`benchmarks/cases/*.dsl` 零诊断，IR 输出不变。
- [ ] `python3 -m pytest tests/ -q` 全绿；`python .claude/harness/verify/run.py --level L2` 通过。
- [ ] CLI 冒烟：含错 DSL 输出 gcc 风格多行错误、退出码 1；含 `--dsl` 的内联源码正常路径不受影响。
- [ ] 零新第三方依赖（仅 `re`/`difflib`）。

验收命令：

```bash
python3 -m pytest tests/test_dsl_errors.py tests/test_dsl_errors_integration.py -v
python3 -m pytest tests/test_dsl_extended.py tests/test_parser.py -q
make test
python .claude/harness/verify/run.py --level L2
printf 'a = add(1)\nb = retrun(a, 2)\n' > /tmp/bad.dsl
python -m scratchv --dsl /tmp/bad.dsl -o /tmp/out.s; echo "exit=$?"
```

---

## 8. 风险与回退

| # | 风险 | 等级 | 缓解 |
|---|------|------|------|
| R1 | 原先被静默容忍的畸形输入（缺终结符、游离终结符、嵌套调用）改为报错，属行为变更 | 中 | 仅影响非法输入；合法 DSL golden 全量回归；`compiler.py` 仅对 DSL 路径生效，ONNX 路径不受影响 |
| R2 | 循环导入（dsl_errors ↔ dsl_parser） | 高 | 严格单向依赖：`DSLParseError` 迁至 `dsl_errors.py`，`dsl_parser` 仅导入不反向；`__init__.py` 保持导入顺序 |
| R3 | 基础解析器行号计算改变（`strip()` → 原文切分）导致错误行内容与之前不同 | 低 | 只影响错误输出；语义解析用 `raw.strip()`，IR 不变；golden 比对保证 |
| R4 | 锚定赋值正则改变合法输入接受范围 | 低 | 合法语句必然匹配锚定式；行内注释在匹配前剥离（与扩展解析器一致）；golden 覆盖 |
| R5 | `_parse_block` 重构引入 IR 结构差异（块数量/跳转） | 中 | 保留"无 else 也建空 else 块"的既有形状；`test_dsl_extended.py` IR 块数断言 + 全量 goldens |
| R6 | dataclass 异常自定义 `__init__` 与既有测试冲突 | 低 | 保持 7 个位置参数顺序；`suggestion` 仅关键字；追加单测锁定 |
| R7 | 收集模式下部分 IR 被误用 | 中 | 在 docstring/文档明确契约；`compiler.py` 不使用收集模式 |

**回退方案**：

1. 全部改动限定在单个 commit（建议 message：`feat(dsl): wire gcc-style DSL diagnostics into parsers (#9)`），失败时 `git revert <sha>` 即可；
2. 异常继承链保证回退前后 `except DSLParseError` 调用方语义连续，不需要同步回退调用方；
3. 不引入环境变量/开关；若需灰度，可先在 `compiler.py` 侧不接 E203 报错（临时 `except DSLSyntaxError` 后回落旧文案），但**不推荐**，会破坏验收 5。

---

## 实现结果（2026-09-14 集成）

> **分支集成 commit**：`795067f`（`feat(topic09): integrate DSL syntax diagnostics with gcc-style error reporting`），文档提交 `6248f15`，基于 main `73c3926`
> **分支全量**：`PYTHONPATH=. python3.11 -m pytest tests/ -q` → **716 passed / 0 failed**（评审基线）

### 实现文件与要点

| 文件 | 要点 |
|------|------|
| `scratchv/frontend/dsl_errors.py` | 异常契约：`DSLSyntaxError(DSLParseError)`、`ErrorCode`、`format_error`、`ErrorCollector` |
| `scratchv/frontend/dsl_parser.py` | 行号/列号定位 + 语句级错误分支（严格模式首错抛出） |
| `scratchv/frontend/dsl_extended.py` | `_parse_block` 重构 + E202/E203/E204（缺 `endif`/`endwhile`/`endfor`、游离终结符） |
| `scratchv/compiler.py` | 错误传播：gcc 风格多行诊断、CLI 退出码 1，DSL 路径生效、ONNX 路径不受影响 |
| `scratchv/frontend/__init__.py` | 导出顺序调整（保持单向依赖） |
| `tests/test_dsl_errors.py`（追加）、`tests/test_dsl_errors_integration.py`（新增）、`tests/data/dsl_golden_ir.json` | 单测 + 集成 + 合法 DSL golden 回归 |

### 修复轮（2026-09-14，评审后）

评审 `topic09-review.md` 的 P1/P2 修复与本轮代码同步：

- **F1**：`for`/`return` 语句改 `fullmatch` 锚定；`_parse_kwargs` 按 `OP_SIGNATURES` 校验未知 kwarg 与数值 kwarg（E304）。collector 模式不再静默放行 `return x junk`、`for ... junk`、未知/非数值 kwargs。
- **F2**：`DSLValidator.validate` 达到 `max_errors` 后不再提前 `break`，`suppressed_count` 统计全部未报告错误（50 错 / max=3 → 47）。
- **F5/F6**：`ErrorCollector(max_errors<1)` 抛 `ValueError`；抑制分支同样写入去重键，重复的被抑制错误不重复计数。
- **F7**：两个解析器 `_raw_lines` 统一 `\r\n`/`\r` → `\n` 归一化，CRLF 源 strict/collector 的 `source_line` 一致。
- **F3/F4**：设计文档 §2.2 补双错误码空间映射表；v2 文档按实现校正格式前缀、排序、note 文案、`context_lines`、E304 状态等。

### 测试数字（修复轮后）

| 口径 | 结果 |
|------|------|
| 定向（`test_dsl_errors*.py` + `test_dsl_validator.py` + `test_dsl_extended.py` + `test_parser.py` 等） | 231 passed |
| 分支全量 | **736 passed / 0 failed**（修复前 716，新增 20 条回归用例） |

### 与本文档的偏差 / 未完成项

- `examples/cnn_model.dsl` 基线本来就不可解析：实现保持报错行为并将其纳入 golden 锁定（**不是**本课题引入的回归）。
- `$` 错误列号以实现为 15 列，设计文档示例已同步为 15。

### 已知限制

- 原先被静默容忍的畸形输入（缺终结符、游离终结符、嵌套调用、畸形 `for`/`return`、非法 kwarg）改为报错，属行为变更；仅影响非法 DSL，合法 DSL golden 全量回归。
- `collector=` 收集模式仅用于测试/诊断，`compiler.py` 不使用收集模式。
- strict 与 collector 的错误码空间不同（见设计文档 §2.2 映射表）；strict 下 validator 漏检的形态仍可能抛富码（如 E205）。
