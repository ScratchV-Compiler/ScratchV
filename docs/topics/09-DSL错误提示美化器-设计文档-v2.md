# ScratchV DSL 错误提示美化器技术设计文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/frontend/dsl_errors.py`（错误对象 / 格式化 / 收集器）、`scratchv/frontend/dsl_parser.py`（基础语句解析器）、`scratchv/frontend/dsl_extended.py`（if/while 块解析器）、`scratchv/compiler.py`（编译驱动集成）  
> 功能范围：DSL 词法/语法/语义错误的精确定位（`file:line:col`）、gcc 风格渲染、修复建议、多错误收集、块结构缺失（缺 `endif`/`endwhile`/`endfor`）诊断。**不含**表达式解析器重写（课题 15）与 ONNX 侧诊断；**保证**合法 DSL 的解析行为与生成 IR 逐字节不变。  
> 取代关系：本文档（v2）取代同目录 `09-DSL错误提示美化器-设计文档.md`（v1）；错误码、消息格式与收集策略一律以本文档为准，v1 与本文档冲突处以本文档为准。

---

## 一、功能介绍

### 1.1 功能概述

- **现状**：`dsl_errors.py` 已实现 `DSLSyntaxError`、`ErrorCollector`、`format_error`、`_SUGGESTIONS`，但两个解析器从不抛出 `DSLSyntaxError`（只抛无行号的 `DSLParseError`），`ErrorCollector` 零集成；`format_error` 的插入符列号未计入行号位数，拼写库中 `add(`/`mul(` 等键不可达（被 `strip("(){},:=* ")` 去掉括号后永远匹配不到）。课题 9 名义完成、实际未落地。
- **本次目标**：接通"错误产出（parser）→ 错误承载（exception）→ 错误渲染（format）→ 错误聚合（collector）→ 驱动集成（compiler）"整条链路。改动只做**错误报告链路**，不重写表达式解析。

效果对比：

```
# 现状（无行号、无源码行、无建议）
DSLParseError: Cannot parse line: d = retrun(c)

# 本次交付（gcc 风格）
bad.dsl:4:5: error[E301]: unknown operation 'retrun'
  4 | d = retrun(c)
    |     ^~~~~~
note: did you mean 'return'?
```

### 1.2 设计目标

1. **定位精确**：1-based 行号/列号（列按源码字符计），展示错误行原文，插入符与源码列严格对齐。
2. **分类稳定**：错误码 `E1xx`（词法）/`E2xx`（语法）/`E3xx`（语义），消息模板固定，便于测试断言与文档化。
3. **建议有效**：显式 hint 优先，其次拼写库，再其次相似算子（difflib），最后通用修复；一条错误至多一行 `note:`。
4. **可恢复收集**：collector 模式下跳过坏行/坏块继续解析，默认最多 20 条，超出显示 suppressed 计数；严格模式（默认）保持首错即抛。
5. **向后兼容**：`DSLSyntaxError` 是 `DSLParseError` 的子类，既有 `except DSLParseError` 代码不受影响；`parse(text)` 默认行为与合法输入完全不变。
6. **零新依赖**：只用标准库 `re`、`difflib`。

---

## 二、设计规范

### 2.1 错误消息格式

gcc/clang 风格，逐字节格式定义（BNF）：

```
diagnostic     ::= header NL source_display [NL marker] [NL note]

header         ::= location error_label ": " message
error_label    ::= "error" [ "[" error_code "]" ]
location       ::= (filename | "<dsl>") ":" line ":" col ": "
source_display ::= gutter_src source_line
marker         ::= gutter_mark (col-1)*SP "^" ("~")*
note           ::= "note: " suggestion

gutter_src     ::= 2*SP line_str " | "
gutter_mark    ::= (3 + len(line_str))*SP "| "
```

| 元素 | 规则 |
|------|------|
| `line` / `col` | 十进制无前导零；`line` 1-based，`col` 1-based |
| `filename` | 为 `None` 时渲染 `<dsl>`，冒号 prefix 永不为空 |
| `error_code` | 可选，大写，渲染为 `error[E201]:` 前缀（无码时仅 `error:`） |
| `message` | 全小写、无句点、token 用单引号包裹（如 `unknown operation 'retrun'`） |
| `source_line` | 原文行（不 strip、不含换行符） |
| 列对齐 | `len(gutter_src) == len(gutter_mark)`；插入符起始显示列 = `len(gutter_src) + (col-1)` |
| 波浪线 | `token_len` = 从 `col` 起向后扫描 `[A-Za-z0-9_]` 的长度；长度 `max(token_len-1, 1)`；`col` 超出行尾时 clamp 到行尾+1 且 `token_len=1` |
| `note` | 仅在存在建议时输出；同一错误最多一行 |
| `context_lines` | `>0` 时必须同时提供 `source`，否则忽略（禁止输出无内容的上下文行） |
| 颜色 | `use_color=True` 时：location=BOLD、`error`=RED、gutter=GRAY、`^`=GREEN、`note`=CYAN；颜色码不参与列宽计算 |
| 换行 | `format_error` 返回的字符串不含尾随换行 |

**逐字节样例**（`use_color=False`）：

```
unknown_op.dsl:2:5: error[E301]: unknown operation 'retrun'
  2 | b = retrun(a, 1)
    |     ^~~~~~
note: did you mean 'return'?
```

对齐推导：`gutter_src = "  2 | "`（6 字符），`gutter_mark = "    | "`（6 字符），`col=5` 故插入符在第 `6+(5-1)=10` 个字符位，正对 `retrun` 首字母 `r`。

**双位数行号样例**（L=10，验证行号位数修复）：

```
test.dsl:10:5: error[E301]: unknown operation 'retrun'
  10 | d = retrun(c)
     |     ^~~~~~
```

`gutter_src = "  10 | "`（7 字符），`gutter_mark = "     | "`（7 字符），竖线与插入符仍严格对齐（这是现状 bug：旧实现 `" " * (col + 3)` 在双位数行号下会左偏 1 列）。

### 2.2 错误分类与错误码表

| 错误码 | 类别 | 常量名 | 触发条件 | 消息模板 | 定位（line:col） | 本轮 |
|--------|------|--------|----------|----------|------------------|------|
| E101 | 词法 | `LEX_ILLEGAL_CHAR` | 语句无法解析且含非法字符 | `unexpected character '{ch}'` | 非法字符处 | ✅ |
| E201 | 语法 | `SYN_INVALID_STATEMENT` | 行不匹配任何语句形式 | `cannot parse statement; expected 'name = op(args)'` | 语句首字符 | ✅ |
| E202 | 语法 | `SYN_INVALID_CONDITION` | `if`/`while` 头不匹配条件模式 | `invalid condition in '{kw}'; expected '{kw} (<expr>) <op> (<expr>):'` | `if`/`while` 关键字 | ✅ |
| E203 | 语法 | `SYN_MISSING_TERMINATOR` | 缺少 `endif`/`endwhile`/`endfor` | `missing '{end}' for '{kw}' opened here` | 开块关键字 | ✅ |
| E204 | 语法 | `SYN_STRAY_TERMINATOR` | 游离终结符 / 无配对开块 | `'{tok}' without matching '{opener}'` | 终结符 | ✅ |
| E205 | 语法 | `SYN_NESTED_CALL` | 实参文本含内层 `(`/`)` | `nested function call is not supported` | 内层 `(` | ✅ |
| E206 | 语法 | `SYN_MISSING_COLON` | `if`/`while` 缺 `:` | 预留（当前语法容忍可选冒号，不报错） | — | ◻ |
| E301 | 语义 | `SEM_UNKNOWN_OP` | 未知算子名 | `unknown operation '{op}'` | 算子名 | ✅ |
| E302 | 语义 | `SEM_ARITY` | 实参个数不符 | `{op}() expects {n} argument(s), got {m}` | 算子名 | ✅ |
| E303 | 语义 | `SEM_UNDEFINED_VAR` | 使用未赋值变量 | 预留（`_resolve` 首次出现即建值，保持现状） | — | ◻ |
| E304 | 语义 | `SEM_UNKNOWN_KWARG` | 未知 `key:value` 实参，或数值 kwarg 值非数字 | `invalid keyword argument '{k}'` / `'{k}' requires a numeric value` | kwarg 名首字符 | ✅ |

**双错误码空间映射（strict validator ↔ collector 富解析器）**：同一输入在 strict 模式下先由 `DSLValidator` 预校验（`E1xx`/`E2xx`），collector 模式跳过预校验、由富解析器报右侧码。**同一个数字码在两个空间含义不同**，跨模式比较错误码没有意义，下表为显式映射与歧义标注：

| strict（validator） | collector（富解析器） | 说明 / 歧义 |
|---------------------|----------------------|-------------|
| `E100` cannot parse statement / for statement | `E201` invalid statement | 含 `return x junk`、`for i = 0, 4 junk` 等畸形语句 |
| `E100` return requires a value | `E201`（兜底分支） | 富解析器不单列"return 缺值"码 |
| `E101`（括号不平衡） | `E201`（带括号修复 hint） | **歧义**：`E101` 在 strict 为括号错误，在富空间为非法字符 |
| `E101`（`if`/`while` 缺括号条件） | `E202` invalid condition | — |
| `E103` invalid identifier / loop variable | `E201` | 富解析器不区分标识符合法性 |
| `E110` 游离/错配终结符 | `E204` stray terminator | — |
| `E111` unterminated block | `E203` missing terminator | — |
| `E112` else 问题 | `E204` | — |
| `E200` unsupported operation | `E301` unknown operation | **歧义**：`E200` 仅存在于 strict 空间 |
| `E201` arity（positional 个数） | `E302` arity | **歧义**：`E201` 在 strict 为 arity，在富空间为 invalid statement |
| `E202` 未知/缺少 keyword | `E304` invalid keyword argument | — |
| `E203` 值缺失/非数字 | `E304` requires a numeric value | **歧义**：`E203` 在 strict 为参数值错误，在富空间为 missing terminator |
| （strict 归入 `E100`） | `E101` illegal character | — |
| `E201`（实参被逗号拆分后计数不符）或 `E205`（validator 漏检走富路径） | `E205` nested call | 嵌套调用 strict 下码不稳定（如 `c = add(mul(a, b), d)` → `E201`，`a = add(mul(b, c))` → `E205`） |

> 另注：strict 模式在 validator 未命中时会继续走富解析器并抛出富码（上表 E205 一行的后一种情况），因此 strict 观测到的码不全是 `E1xx`/`E2xx`。若后续要求"错误码跨模式稳定"，需将 validator 码重编号或映射进统一命名空间（本课题不做）。

**算子实参个数基准**（E302 判定，仅统计普通实参，`k:v` kwargs 不计）：

| 算子 | 普通实参期望个数 | 备注 |
|------|------------------|------|
| `add` `sub` `mul` `div` | 2 | `add() requires exactly 2 arguments` |
| `neg` `exp` `relu` `gelu` | 1 | |
| `dot` | 2 | 另接受 `len:`/`length:` |
| `matmul` | 2 | 另接受 `rows:/cols:/inner:` 或 `m:/n:/k:` |
| `softmax` | 1 | 另接受 `axis:` |
| `maxpool` | 1 | 另接受 `kernel:/stride:` |

### 2.3 位置信息契约

- `line`：物理行号，1-based，等于 `text.split("\n")` 下标 + 1；`line=0` 仅保留给文件级错误（如收集器溢出提示）。
- `col`：字符列，1-based，按 Unicode code point 计数；**tab 记 1 列**，不做 tab 展开。
- `source_line`：该行原始文本（不 strip、不含 `\n`）；由解析器在构造错误时从 `self._raw_lines[line-1]` 取得。
- `filename`：`parse(filename=...)` 透传；CLI 传输入路径；内联字符串为 `None`，渲染为 `<dsl>`。
- 所有错误必须经统一入口 `_report_error(line_no, col, message, error_code, fix_hint)` 构造，禁止在解析器内手工拼装消息或多行字符串。

**各类错误定位基准表**：

| 错误码 | col 定位 |
|--------|----------|
| E101 | 首个不在 `[A-Za-z0-9_(),:=.+\-*/#%\s]` 内的字符 |
| E201 | 语句缩进后首字符（通常为 1） |
| E202 | `if`/`while` 关键字首字符 |
| E203 | 对应 `if`/`while`/`for` 关键字首字符 |
| E204 | 终结符/`else` 首字符 |
| E205 | 实参文本中内层 `(` 的位置 |
| E301 / E302 | 算子名首字符（由正则 `m.start(2)` 精确定位） |

### 2.4 修复建议生成规则

`note:` 文本按以下优先级链产生，命中即停止：

1. **解析器显式 `fix_hint`**：E202/E203/E204/E205/E302 由解析器直接给出确定建议。
2. **拼写库 `_SUGGESTIONS`**：对 `source_line` 用 `[A-Za-z_]\w*` 正则分词（不再用 `str.split()`），优先匹配 `col` 所在 token，其次全行扫描；大小写不敏感精确匹配。**本次修复**：删除现有不可达的 `add(`/`mul(`/`sub(`/`div(`/`matmul(` 键，改由 `_ARITY_HINTS`（键为裸算子名）服务 E302。
3. **相似算子 `suggest_op(op, candidates, cutoff=0.6)`**：`difflib.get_close_matches` 在支持算子表内找最近者，输出 `did you mean '{cand}'?`。
4. **通用修复 `_COMMON_FIXES`**：按消息关键词（unterminated / undefined / paren / operator）兜底。
5. 全不命中 → 不输出 `note:` 行。

注：E301 的建议顺序为"拼写库优先、difflib 次之"，保证 `retrun` 稳定提示 `return`，不会被 difflib 误配为 `relu`。

### 2.5 DSL 块结构缺失报错规则

块级结构由 `ExtendedDSLParser` 的 `_parse_if_block`/`_parse_while_block` + 新公共例程 `_parse_block` 管理；`for` 由基础解析器 `_loop_stack` 管理。

| 场景 | 触发点 | 错误码 | 定位 | 恢复动作 |
|------|--------|--------|------|----------|
| 缺 `endif` 直到 EOF | `_parse_block` 返回 `(len, None)` | E203 | `if` 关键字 | 收尾 emitted `endif` 标签，结束该块 |
| 缺 `endif` 却遇 `endwhile` | 内层块扫描遇非本块终结符 | E203 | `if` 关键字 | **不消费**该终结符，返回给外层 `while` 处理 |
| 缺 `endwhile` 直到 EOF / 遇 `endif` | 同上 | E203 | `while` 关键字 | 不消费外来终结符，交给外层 |
| 缺 `endfor` 直到 EOF | 基础解析器 `_loop_stack` 非空 | E203 | `for` 行 | 逐条上报后清空循环栈；不再追加自动 `ret` |
| 游离 `endif`/`endwhile` | 顶层 `_parse_line` | E204 | 终结符 | 跳过该行 |
| 游离 `else` / `else:` | 非 `if` 上下文（`else` 分支内再遇 `else`） | E204 | 关键字 | 跳过该行 |
| 游离 `endfor` | `_loop_stack` 为空 | E204 | 关键字 | 跳过该行 |
| `if`/`while` 头非法 | `_parse_condition` 返回 `None` | E202 | 关键字 | `_recover_after_bad_header`：扫描至本块匹配终结符或 EOF；期间不再报派生错误 |

约束规则：

- 同一未闭合块**只报一条** E203，避免级联噪音。
- 嵌套多个未闭合块时按"发现顺序"上报（递归自内向外，后进先出）；收集器按 `(line, col, error_code)` 排序后输出，不保持插入顺序。
- 缺终结符时，块内已成功的合法语句保留其 IR；块结束时补齐跳转标签（与正常解析的块结构一致），避免 IR verifier 报未终结块。
- E203 的插入符覆盖关键字本身（`if` → `^~`，`while` → `^~~~~`，`for` → `^~~`）。

### 2.6 合法/非法示例

**合法示例（必须 0 诊断、IR 不变）**：

```
# 1. 无 else 的 if
if (a > b):
  c = add(a, b)
endif
return c

# 2. else / else: 两种写法 + 嵌套 if/while
if (a > 0):
  while (a < 10):
    a = add(a, 1)
  endwhile
else:
  a = sub(a, 1)
endif
return a

# 3. 冒号可省（E206 预留，不报错）
if (a > b)
  c = mul(a, b)
endif
return c

# 4. for 循环（基础解析器）
for i = 0, 4
  acc = add(acc, i)
endfor
return acc

# 5. 行内注释与空行
# 注释行
x = add(a, b)   # 行内注释
return x
```

**非法示例与预期诊断**：

| 输入（关键行） | 预期 header |
|----------------|-------------|
| `d = retrun(c)` | `f.dsl:1:5: error[E301]: unknown operation 'retrun'` |
| `a = add(1)` | `f.dsl:1:5: error[E302]: add() expects 2 arguments, got 1` |
| `a = add()` | `f.dsl:1:5: error[E302]: add() expects 2 arguments, got 0` |
| `if a > b:` | `f.dsl:1:1: error[E202]: invalid condition in 'if'; expected 'if (<expr>) <op> (<expr>):'` |
| `if (a > b)` 无 endif 到 EOF | `f.dsl:<if行>:1: error[E203]: missing 'endif' for 'if' opened here` |
| `while (i < 9)` 无 endwhile 到 EOF | `f.dsl:<while行>:1: error[E203]: missing 'endwhile' for 'while' opened here` |
| `for i = 0, 4` 无 endfor 到 EOF | `f.dsl:<for行>:1: error[E203]: missing 'endfor' for 'for' opened here` |
| 顶层裸 `endwhile` | `f.dsl:1:1: error[E204]: 'endwhile' without matching 'while'` |
| 顶层裸 `endfor` | `f.dsl:1:1: error[E204]: 'endfor' without matching 'for'` |
| `if` 块外裸 `else:` | `f.dsl:1:1: error[E204]: 'else' without matching 'if'` |
| `c = add(mul(a, b), d)` | `f.dsl:1:12: error[E205]: nested function call is not supported` |
| `a = add(b, c) $` | `f.dsl:1:15: error[E101]: unexpected character '$'` |
| `a = add(x, y, foo:1)` | `f.dsl:1:15: error[E304]: invalid keyword argument 'foo'` |
| `m = matmul(a, b, rows:abc)` | `f.dsl:1:18: error[E304]: 'rows' requires a numeric value` |

（注：`c = add(mul(a, b), d)` 中第 12 列是内层 `(`：`c = add(` 占 8 列，`mul` 占 9–11 列。）

---

## 三、测试设计

测试分两层：`tests/test_dsl_errors.py`（单元：异常/格式化/收集器）与新增 `tests/test_dsl_errors_integration.py`（集成：解析器错误分支 + 多错误恢复 + 合法 DSL 回归）。以下至少 3 个核心用例。

### 测试用例 1：未知算子（E301，列号 + 插入符 + 拼写建议）

**文件**：`tests/test_dsl_errors_integration.py::test_unknown_op_location_and_suggestion`

**输入 DSL**（`unknown_op.dsl`）：

```
1: a = add(x, y)
2: b = retrun(a, 1)
```

**执行**：`ExtendedDSLParser().parse(source, filename="unknown_op.dsl")`，严格模式捕获异常。

**预期错误消息**（`format_error(e, use_color=False)` 逐字节）：

```
unknown_op.dsl:2:5: error[E301]: unknown operation 'retrun'
  2 | b = retrun(a, 1)
    |     ^~~~~~
note: did you mean 'return'?
```

**验证点**：

- `e.line == 2`、`e.col == 5`、`e.error_code == "E301"`、`e.filename == "unknown_op.dsl"`；
- `e.source_line == "b = retrun(a, 1)"`；
- `e.fix_hint == "did you mean 'return'?"`（拼写库优先于 difflib）；
- 输出含精确 marker 行 `"    |     ^~~~~~"`（4 空格 + `| ` + 4 空格 + `^` + 5 个 `~`），列号与 `retrun` 首字符对齐；
- `isinstance(e, DSLParseError) is True`（向后兼容）。

### 测试用例 2：缺 `endif`（E203，块结构缺失）

**文件**：`tests/test_dsl_errors_integration.py::test_missing_endif_reported_at_opener`

**输入 DSL**（`missing_endif.dsl`）：

```
1: i = add(x, 1)
2: if (i > 0):
3:   y = mul(i, 2)
4: return y
```

**执行**：严格模式 `ExtendedDSLParser().parse(source, filename="missing_endif.dsl")`。

**预期错误消息**：

```
missing_endif.dsl:2:1: error[E203]: missing 'endif' for 'if' opened here
  2 | if (i > 0):
    | ^~
note: add 'endif' to close this block
```

**验证点**：

- `e.line == 2`、`e.col == 1`、`e.error_code == "E203"`（定位在开块关键字而非 EOF）；
- 消息不含大写开头、无句点；插入符 `^~` 覆盖 `if` 两个字符；
- 收集模式下 `collector.error_count == 1`（块内合法语句 `y = mul(...)` 不产生派生错误）；
- 对比用例：`while (i < 9):` 无 `endwhile` → `missing 'endwhile' for 'while' opened here`；顶层 `for` 无 `endfor` → `missing 'endfor' for 'for' opened here`。

### 测试用例 3：多错误收集（E302 + E301 + E202 + E204，恢复能力）

**文件**：`tests/test_dsl_errors_integration.py::test_multi_error_collection_and_recovery`

**输入 DSL**（`multi_error.dsl`）：

```
1: a = add(1)
2: b = retrun(a, 2)
3: if a > 0:
4:   c = mul(a, 2)
5: endwhile
```

**执行**：`collector = ErrorCollector(filename="multi_error.dsl", use_color=False)`，`ExtendedDSLParser().parse(source, filename="multi_error.dsl", collector=collector)`；解析返回部分 Program（不抛异常）。

**预期输出**（`collector.report()`）：

```
--- 4 error(s) found ---
multi_error.dsl:1:5: error[E302]: add() expects 2 arguments, got 1
  1 | a = add(1)
    |     ^~~
note: add() requires exactly 2 arguments
multi_error.dsl:2:5: error[E301]: unknown operation 'retrun'
  2 | b = retrun(a, 2)
    |     ^~~~~~
note: did you mean 'return'?
multi_error.dsl:3:1: error[E202]: invalid condition in 'if'; expected 'if (<expr>) <op> (<expr>):'
  3 | if a > 0:
    | ^~
note: expected one of ==, !=, <, >, <=, >= and parentheses around each operand
multi_error.dsl:5:1: error[E204]: 'endwhile' without matching 'while'
  5 | endwhile
    | ^~~~~~~~
note: remove this line or add a matching 'while'
```

**验证点**：

- `collector.error_count == 4`，顺序为发现顺序（行 1 → 2 → 3 → 5）；
- 第 3 行的坏块恢复：扫描至 `endwhile` 时**不消费**（因它不属于该 `if`），第 5 行独立报 E204；
- 每行最多一条错误（同行不叠加）；
- `collector.has_errors is True`，`report()` 含 `--- 4 error(s) found ---` 头；
- 部分 Program 不可信但可构造（`program.functions` 存在）；
- 严格模式对照：同一输入首错抛 `DSLSyntaxError(line=1, col=5, error_code="E302")`。

### 测试用例 4（回归）：合法 DSL 零诊断 golden

**文件**：`tests/test_dsl_errors_integration.py::test_legal_dsl_zero_diagnostics`

**输入**：`examples/**/*.dsl` 与 `benchmarks/cases/*.dsl` 全量文件，按是否含扩展关键字选择 `ExtendedDSLParser` 或 `DSLParser`。

**预期输出**：collector 模式下全部 `has_errors == False`，且与改动前解析得到的指令序列一致（对每个文件比较 `Program.dump()` 的规范化文本）。

**验证点**：

- 合法 DSL 的 IR 输出逐字节不变；
- 现有 `tests/test_dsl_extended.py::test_invalid_if_missing_parens` 仍以 `except DSLParseError` 捕获成功（验证异常继承）；
- `make test` 全量通过。

---

## 四、修改模块与实现步骤

### 4.1 涉及文件

| 文件 | 角色 | 改动量级 |
|------|------|----------|
| `scratchv/frontend/dsl_errors.py` | 异常基类、错误码、格式化、建议、收集器 | 中 |
| `scratchv/frontend/dsl_parser.py` | 语句级错误分支、行号/列号、for 栈位置 | 中 |
| `scratchv/frontend/dsl_extended.py` | 块解析重构、E202/E203/E204、恢复 | 大 |
| `scratchv/compiler.py` | 不再用 fallback 吞掉定位错误；解析错误带格式输出 | 小 |
| `scratchv/frontend/__init__.py` | 导出 `DSLParseError`/`ErrorCode`/`make_error` | 极小 |
| `tests/test_dsl_errors.py` | 补格式化/收集器新行为单测 | 小 |
| `tests/test_dsl_errors_integration.py` | 新增集成测试（本课题核心验收） | 新文件 |

（注：实际文件路径可能不同；以仓库当前快照为准，锚点见开发文档。）

### 4.2 `dsl_errors.py` 改造

1. **迁移异常基类**：在 `dsl_errors.py` 定义 `class DSLParseError(Exception)`；`dsl_parser.py` 删除本地定义并改为导入 + 重导出（消除循环依赖：`dsl_errors` 不依赖任何 frontend 模块）。
2. **错误码常量**：新增 `class ErrorCode`，定义 `E101/E201/E202/E203/E204/E205/E301/E302`。
3. **`DSLSyntaxError`**：继承 `DSLParseError`；增加 `suggestion` 读写别名属性；构造器支持 `suggestion=` 关键字（等价 `fix_hint=`）。
4. **建议引擎**：重写 `_compute_suggestion` 为"指定列 token 优先 + 全行扫描"；新增 `suggest_spelling`、`suggest_op`；`_SUGGESTIONS` 删除不可达的带括号键，新增 `_ARITY_HINTS`。
5. **`format_error` 修复**：gutter 对齐公式（2.1）、插入符 clamp、`source` 参数支持真实上下文、`<dsl>` 兜底、token 长度含 `_`。
6. **`ErrorCollector`**：`source` 参数、去重、`suppressed_count`、溢出 note 替换旧"伪错误"哨兵。

### 4.3 `dsl_parser.py` 改造

1. `__init__` 增加 `self._raw_lines`、`self._filename`、`self._collector`、`self._for_positions`。
2. `parse(text, filename=None, collector=None)`：改用 `raw_lines = text.split("\n")` 保留物理行号；逐行带行号调用 `_parse_line`；EOF 时对 `_loop_stack` 逐条报 E203。
3. `_parse_line(line, line_no=0)` 各错误分支改造（详见开发文档"错误分支清单"）：内联注释剥离 → 锚定赋值正则 → 空实参/嵌套调用/括号不平衡 → 未知算子 → 参数个数 → 兜底 E101/E201。
4. 新增 `_report_error` / `_col_of` 公共辅助（放基类供扩展解析器复用）。

### 4.4 `dsl_extended.py` 改造

1. `parse(text, filename=None, collector=None)`：设置行列上下文；顶层分发用 `^if\b` / `^while\b`；EOF 检查 `_loop_stack`/`_while_stack`。
2. 新增 `_parse_block(lines, start_idx, terminators, opener_kind, opener_line, opener_col)`：统一 then/else/while-body 三处重复扫描逻辑；只返回遇到的终结符（**不消费**）；`else` 在非法位置报 E204。
3. `_parse_if_block` / `_parse_while_block` 按 2.5 规则处理 E202/E203/E204，并在错误路径下仍补齐块标签。
4. `_parse_line` 覆写：`endif`/`endwhile`/`else`/`else:` 在块解析器未消费而落到此处时 → E204（不再静默 return）。
5. 新增 `_recover_after_bad_header`：E202 后跳过整个块到匹配终结符或 EOF。

### 4.5 `compiler.py` 集成

1. `_parse`：`except DSLSyntaxError: raise`（**不得**再被 `DSLParser` fallback 吞掉）；`except DSLParseError:` 保留旧 fallback；向 parser 透传 `filename`。
2. `compile`：`except DSLSyntaxError as e: return CompileResult(success=False, errors=[str(e)])`，使 CLI 输出的错误即为 gcc 风格多行文本（`str(DSLSyntaxError)` 内部走 `format_error(use_color=False)`）。
3. 不改 CLI 参数与退出码语义（失败仍返回 1）。

### 4.6 集成与回归测试

- 新增集成测试（第三部分用例 1–4）；
- `python3 -m pytest tests/test_dsl_errors.py tests/test_dsl_errors_integration.py -q`；
- `make test`（全量 348+ 用例）；
- `python .claude/harness/verify/run.py --level L2`；
- 手工 CLI 冒烟：对含 2 个错误的 DSL 验证退出码 1 与 stderr 输出格式。

---

## 五、附录

### 5.1 完整错误输出示例

**示例 A：多个错误的完整报告（对应测试用例 3）**

```
--- 4 error(s) found ---
multi_error.dsl:1:5: error[E302]: add() expects 2 arguments, got 1
  1 | a = add(1)
    |     ^~~
note: add() requires exactly 2 arguments
multi_error.dsl:2:5: error[E301]: unknown operation 'retrun'
  2 | b = retrun(a, 2)
    |     ^~~~~~
note: did you mean 'return'?
multi_error.dsl:3:1: error[E202]: invalid condition in 'if'; expected 'if (<expr>) <op> (<expr>):'
  3 | if a > 0:
    | ^~
note: expected one of ==, !=, <, >, <=, >= and parentheses around each operand
multi_error.dsl:5:1: error[E204]: 'endwhile' without matching 'while'
  5 | endwhile
    | ^~~~~~~~
note: remove this line or add a matching 'while'
```

**示例 B：异常对象的 `str()`**

```python
from scratchv.frontend.dsl_extended import ExtendedDSLParser

try:
    ExtendedDSLParser().parse(source, filename="bad.dsl")
except Exception as e:
    print(e)          # 等价 format_error(e, use_color=False)，多行 gcc 风格
```

**示例 C：收集模式调用模板**

```python
from scratchv.frontend.dsl_errors import ErrorCollector
from scratchv.frontend.dsl_extended import ExtendedDSLParser

collector = ErrorCollector(filename=path, use_color=False)
program = ExtendedDSLParser().parse(source, filename=path, collector=collector)
if collector.has_errors:
    print(collector.report())          # 全部错误一次性输出
    # 注意：此时 program 为部分结果，不可用于代码生成
```

### 5.2 错误消息模板速查

| 码 | 消息模板 | 显式 hint 模板 |
|----|----------|----------------|
| E101 | `unexpected character '{ch}'` | `remove or replace '{ch}'` |
| E201 | `cannot parse statement; expected 'name = op(args)'` | 括号不平衡时 `missing closing ')'` / `missing opening '('` |
| E202 | `invalid condition in '{kw}'; expected '{kw} (<expr>) <op> (<expr>):'` | `expected one of ==, !=, <, >, <=, >= and parentheses around each operand` |
| E203 | `missing '{end}' for '{kw}' opened here` | `add '{end}' to close this block` |
| E204 | `'{tok}' without matching '{opener}'` | `remove this line or add a matching '{opener}'` |
| E205 | `nested function call is not supported` | `assign the inner call to a temporary variable first` |
| E301 | `unknown operation '{op}'` | 拼写库 / `did you mean '{cand}'?` |
| E302 | `{op}() expects {n} argument(s), got {m}` | `{op}() requires exactly {n} arguments` |
| E304 | `invalid keyword argument '{k}'` / `'{k}' requires a numeric value` | `'{k}' is not accepted by {op}()` / `pass a number for '{k}', got '{v}'` |

### 5.3 如何扩展新错误类型

1. 在 `ErrorCode` 增加常量（编码规则：词法 `E1xx`、语法 `E2xx`、语义 `E3xx`，不复用旧值）。
2. 在解析器对应分支调用 `_report_error(line_no, col, message, ErrorCode.XXX, fix_hint=...)`，禁止手工构造异常。
3. 若需要自动建议：静态映射放 `_COMMON_FIXES`（按错误码优先于关键词），算子相关放 `_ARITY_HINTS`，拼写相关放 `_SUGGESTIONS`。
4. 在 `tests/test_dsl_errors_integration.py` 增加"输入 / 精确 header / 列号 / 验证点"三件套用例。
5. 更新本文档 2.2 错误码表状态列。

### 5.4 参考资料

- 课题文档：`docs/topics/09-DSL错误提示美化器.md`
- GCC 诊断格式：<https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Message-Formatting-Options.html>
- Rust Compiler Error Index：<https://doc.rust-lang.org/error-index.html>
- 相邻课题：课题 15（表达式/语法增强，本课题明确不涉及）；课题 07（编译器日志增强器）
