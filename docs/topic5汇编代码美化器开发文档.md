# ScratchV RISC-V 汇编代码美化器开发文档

> **文档版本**：v1.2
> **创建日期**：2026-07-13
> **更新日期**：2026-08-02
> **涉及模块**：`scratchv/backend/asm_beautifier.py`、`scratchv/backend/asm_parser_for_beautifier.py`

---

## 1. 功能概述与目标

### 1.1 背景与动机
- **现状问题**：RISC-V 汇编代码通常难以直观表达程序行为，机器生成代码还可能存在字段位置不统一、标签与指令混排等问题，阅读门槛较高。
- **应用场景**：编译器开发、调试和教学过程中需要阅读、理解汇编代码；使用美化器可提高阅读效率，降低理解门槛。

### 1.2 功能描述
- **一句话定义**：美化器在保持汇编语义不变的前提下，对字段进行列对齐、为已支持指令生成可读的语义注释，并插入段与函数分隔标题。
- **核心价值**：提升汇编代码的可读性、调试效率和教学展示效果。

### 1.3 目标与非目标
| 类型 | 内容 |
|------|------|
| ✅ 包含范围 | 对 RISC-V 汇编进行安全解析、列对齐、语义注释和段/函数标记，并提供 Python API 与命令行工具。 |
| ❌ 不包含范围 | 不改变汇编语义，也不承担汇编、链接、非法输入修复或未知指令语义推断。 |

---

## 2. 设计与规格说明

### 2.1 用户视角（外部接口）

- **输入语法基准**：

BNF 表示：
```
asm_line          ::= [label ":"] [opcode [operands]] [comment]
label             ::= standard_label | metadata_label
standard_label    ::= identifier
metadata_label    ::= "_op_/" metadata_path
                    | "_op_PPQ_Operation_" number "_" number
metadata_path     ::= metadata_char { metadata_char }
metadata_char     ::= any non-whitespace character except ":"
opcode            ::= instruction | pseudo_instruction | directive
pseudo_instruction ::= "li" | "mv" | "call" | "ret" | ...
operands          ::= operand { top_level_comma operand }
top_level_comma   ::= ","  (* only when parenthesis depth is zero *)
comment           ::= "#" text
```

实现时，解析器应提取标签、操作码、原始操作数字符串、操作数列表和注释；操作数按顶层逗号拆分，括号内的逗号不得作为操作数分隔符。`metadata_path` 可包含 `/`，但不得包含空白或 `:`。`_op_/...:` 和 `_op_PPQ_Operation_<number>_<number>:` 作为算子元数据标签完整保留，不得识别为函数入口。标准指令、汇编器伪指令和汇编器指示必须分类处理。


- **新增/修改的 API**：

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

ParseStatus = Literal[
    "valid",
    "metadata_label",
    "incomplete_operands",
    "unknown_opcode",
    "malformed",
]

class FieldLengths(TypedDict):
    label: int
    opcode: int
    operands_str: int
    comment: int

@dataclass
class ParsedAsmLine:
    raw: str
    label: str | None
    opcode: str | None
    operands_str: str
    operands: list[str]
    comment: str | None
    lineno: int
    parse_status: ParseStatus
    field_lengths: FieldLengths

def beautify_asm(
    asm_text: str,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
) -> str:
    """返回美化后的 RISC-V 汇编文本。"""

def parse_asm_line(
    line: str,
    lineno: int = 0,
) -> ParsedAsmLine:
    """解析单行汇编并返回结构化结果。"""

def parse_asm(
    asm_text: str,
) -> list[ParsedAsmLine]:
    """按原始顺序解析完整汇编文本，保留空行和行号。"""

def beautify_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
    encoding: str = "utf-8",
) -> str:
    """读取并美化汇编；可选地原子写入目标文件，并返回结果。"""
```

| 参数 | 类型 | 默认值 | 行为 |
|------|------|--------|------|
| `asm_text` | `str` | 无 | 待处理的完整汇编文本。 |
| `align` | `bool` | `True` | 是否对可安全格式化的行进行列对齐。 |
| `add_comments` | `bool` | `True` | 是否为已登记指令添加自动语义注释。 |
| `abi_register_names` | `bool` | `False` | 是否仅在自动注释中将 `x0` 至 `x31` 转为 ABI 别名。 |

`beautify_asm()` 返回值始终为字符串，不得修改输入对象或执行文件写入。空输入和仅含空白的输入必须稳定返回字符串且不抛异常，具体换行形式由兼容性测试固定。

`ParsedAsmLine`、`parse_asm_line()` 与 `parse_asm()` 由美化器专用模块 `scratchv/backend/asm_parser_for_beautifier.py` 提供。该模块需要保留原始操作数字符串、字符串字面量中的 `#` 与逗号、元数据标签、五种 `parse_status` 和字段长度；这些能力超出现有共享 `_asm_parser.py` 的契约，因此不得通过修改 `_asm_parser.py` 强行满足美化器需求，以免影响 peephole、const-merge、inst-counter、inst-scheduler 等既有调用方。`beautify_file()` 同步暴露 `align`、`add_comments`、`abi_register_names`，默认使用 UTF-8。指定 `output_path` 时先写入同目录临时文件，成功后原子替换目标；未指定时只返回字符串。`abi_register_names` 默认值为 `False`，且仅影响自动生成的语义注释。

#### 独立命令行接口

```bash
python3 -m scratchv.backend.asm_beautifier input.s
python3 -m scratchv.backend.asm_beautifier input.s -o output.s
python3 -m scratchv.backend.asm_beautifier input.s --no-align
python3 -m scratchv.backend.asm_beautifier input.s --no-comments
python3 -m scratchv.backend.asm_beautifier input.s --abi-register-names
```

| 参数 | 是否必需 | 说明 |
|------|----------|------|
| `input` | 是 | 输入 `.s` 文件路径。 |
| `-o` | 否 | 输出路径；未指定时写入标准输出。 |
| `--no-align` | 否 | 关闭列宽扫描和列对齐。 |
| `--no-comments` | 否 | 关闭自动语义注释，但保留用户原始注释。 |
| `--abi-register-names` | 否 | 仅在自动注释中启用 ABI 寄存器别名；默认关闭。 |

### 2.2 内部设计（核心逻辑）

#### 数据结构变更

不新增 AST 节点或 IR 指令，仅新增逐行解析结果，保存 `raw`、`label`、`opcode`、`operands_str`、`operands`、`comment`、`lineno`、`parse_status` 和 `field_lengths`。指令规格、语义注释模板及 ABI 寄存器映射使用模块级只读常量。

#### 关键算法/流程

```text
输入文本
  → 切分行尾注释并保留原始行
  → 识别完整标签并分类普通标签/算子元数据标签
  → 解析操作码及原始操作数字段
  → 按顶层逗号拆分操作数（识别括号深度）
  → 分类 parse_status
  → 统计合法行各字段长度
  → 第一遍收集段状态、函数符号和安全列宽
  → 第二遍插入段/函数标题并格式化有效行
  → 合并原始注释与自动语义注释
  → 输出文本
```

格式化器仅把括号深度为零的操作数分隔符规范化为 `, `（无前置空白、后置一个空格），括号内部格式和操作数词法内容保持不变。三列填充宽度按 `min(实际最大长度, 30/12/40)` 计算；超长字段不截断、不填充，其后字段允许自然右移。

#### 状态管理

不新增全局可变状态。单次调用仅维护解析结果、段与函数信息、列宽和输出缓冲区，多次调用之间互不影响。

### 2.3 接口定义（模块间交互）

- **上游依赖**：接收 ScratchV 汇编发射器或用户文件产生的 RISC-V 文本。美化器依赖最终文本、显式选项，以及专用模块 `scratchv/backend/asm_parser_for_beautifier.py` 提供的 `ParsedAsmLine`、`parse_asm_line()` 与 `parse_asm()`。
- **解析器并存边界**：`scratchv/backend/_asm_parser.py` 保持现有接口和行为，继续供 peephole、const-merge、inst-counter、inst-scheduler 等汇编级 pass 使用；`asm_parser_for_beautifier.py` 只供 `asm_beautifier.py` 使用。两个模块不得互相导入，也不得让其他 pass 在本课题中迁移到专用解析器。
- **重复逻辑控制**：`asm_beautifier.py` 本身不得再实现逐行解析正则或操作数拆分逻辑；所有美化器专用解析行为集中在 `asm_parser_for_beautifier.py`。两套解析器允许因契约不同而并存，但公共合法汇编子集应通过兼容性测试保持字段含义一致。
- **下游影响**：输出仍是可供 RISC-V 汇编器读取的 `.s` 文本，也可交给指令统计、终端输出和人工审阅。新增标题必须全部使用 `#` 注释，不能形成伪指令。

---

## 3. 模块修改与实现步骤

### 3.1 涉及的文件清单

| 文件路径 | 修改类型 | 修改内容概述 |
|----------|----------|--------------|
| `scratchv/backend/asm_beautifier.py` | 重构 | 完善指令规格表、列对齐、注释生成、段/函数识别、文件 API 和 CLI。 |
| `scratchv/backend/asm_parser_for_beautifier.py` | 新建 | 实现美化器专用安全解析器、原文字段保留、字符串感知、元数据标签识别和解析状态；不修改 `_asm_parser.py`。 |
| `tests/test_asm_beautifier.py` | 重构 | 按新规格重写单元测试，覆盖正常、边界、异常及 CLI 行为。 |
| `tests/test_asm_line_parser.py` | 新增 | 测试 `parse_asm_line()` 的字段提取、字段长度、操作数拆分、字符串边界、元数据标签和状态分类。 |
| `benchmarks/bench_asm_beautifier.py` | 修改 | 使用固定样例和合成大输入统计耗时、波动、输出大小及膨胀比例。 |

### 3.2 分步实现计划

每个模块均遵循“先编写至少一个对应测试并确认失败，再编写实现使其通过”的顺序，测试不单独作为实施步骤。

| 步骤 | 任务描述 | 预期产出 | 验证方式 |
|------|----------|----------|----------|
| 1 | 先编写解析测试，再实现逐行解析、操作数拆分及 `parse_status` 分类。 | 能正确提取汇编字段，并安全处理未知、缺失和异常输入。 | 解析与状态分类测试通过。 |
| 2 | 先编写格式化测试，再实现两遍列宽扫描、字段对齐和保守输出。 | 普通标签独占一行；状态正常的运行时指令与伪指令对齐，顶层操作数分隔符统一为 `, `；汇编器指示保持原样，三种异常行的注释前代码部分逐字保留，仅追加状态警告注释。 | 输入输出对比及格式化测试通过。 |
| 3 | 先编写语义注释测试，再实现指令模板、伪指令、注释合并和 ABI 别名转换。 | 已支持指令生成正确注释；三种异常状态输出明确警告，且不生成或拼接指令模板结果。 | 指令语义、伪指令、异常警告及 ABI 测试通过。 |
| 4 | 先编写结构识别测试，再实现段标题、函数入口、元数据标签和数据定义处理。 | 汇编段与函数结构清晰，算子元数据和数据内容保持安全。 | 段、函数、元数据及数据测试通过。 |
| 5 | 先编写文件与 CLI 测试，再实现 `beautify_file()`、原子写入、命令行参数和错误处理。 | Python API 与命令行工具可稳定处理正常及异常文件。 | 文件、CLI 子进程及错误路径测试通过。 |
| 6 | 先补充端到端验收用例，再进行模块联调、重构和性能检查。 | 完整实现、测试结果、基准数据及回归结论。 | 目标测试、全量测试和 benchmark 通过。 |

### 3.3 异常处理与边界条件

- [ ] 空输入和仅含空白的输入稳定返回字符串，不抛异常；具体换行形式由兼容性测试固定。
- [ ] `28(sp)`、嵌套括号和负偏移不会被错误拆分。
- [ ] `add a0`、`lw t0`、`beq a0,a1` 标记为 `incomplete_operands`，不崩溃、不补写原始操作数；输出注释为 `[warning: operand missing]`，不进入指令模板匹配。
- [ ] `li,,a0,,3`、`sw ra,,28(sp)`、`main::`、未闭合字符串/括号和 `this is not asm` 等非汇编文本标记为 `malformed`；注释前的代码部分逐字保持不变，输出注释为 `[warning: malformed instruction]`。
- [ ] 未知操作码标记为 `unknown_opcode`；注释前的代码部分逐字保持不变，输出注释为 `[warning: unknown opcode]`，不得使用具体指令模板或 fallback 生成语义说明。
- [ ] `_op_/layer1.0/Conv_5:` 和 `_op_PPQ_Operation_6_29:` 标记为 `metadata_label`，完整保留且不生成函数标题；`_input_copy_done:` 仍是普通标签。
- [ ] 已有算子说明注释的 `nop` 只保留原注释，不追加 `no operation`；无原注释的普通 `nop` 仍可生成该自动注释。
- [ ] 数据定义行不生成运行时指令注释，超长字段不被截断。
- [ ] 输入输出路径不存在、权限不足或编码错误时，CLI 向标准错误输出路径和原因，并以非零状态码退出。
- [ ] 输出写入失败时不覆盖原文件，不残留被误认为成功结果的部分文件。

三种异常状态采用统一的“保留指令字段、替换自动注释”策略：

| `parse_status` | 输出注释 | 模板处理 |
|----------------|----------|----------|
| `incomplete_operands` | `[warning: operand missing]` | 跳过模板匹配，不用空字符串填充占位符 |
| `unknown_opcode` | `[warning: unknown opcode]` | 跳过模板匹配和 fallback 推测 |
| `malformed` | `[warning: malformed instruction]` | 跳过模板匹配，不解释局部片段 |

若输入已有用户注释，必须先保留原注释，再用 ` | ` 追加上述警告；不得追加任何指令模板匹配结果。标签、操作码、已有操作数及其空白组成的代码部分不得因生成警告而改变。例如：

```asm
L1: add a0                    # [warning: operand missing]
L2: custom_op x1,x2,x3       # [warning: unknown opcode]
L3: li,,a0,,3                # [warning: malformed instruction]
```

---

## 4. 测试与验证方案

### 4.1 测试概述

测试资产不拆分为独立汇编样例文件，统一放在两个单元测试文件和一个性能基准文件中：

| 测试 | 描述 | 预期 |
|------|------|------|
| 正常输入解析 | 在 `tests/test_asm_line_parser.py` 中测试普通指令、标签、指示、注释、空行及操作数拆分。 | 各字段提取正确，`parse_status` 为 `valid`。 |
| 元数据标签解析 | 在 `tests/test_asm_line_parser.py` 中测试算子元数据标签与普通内部标签的区分。 | 元数据标签完整保留并标记为 `metadata_label`，且不被识别为函数入口。 |
| 操作数缺失 | 在两个单元测试文件中检查已知操作码缺少操作数时的解析和输出。 | 标记为 `incomplete_operands`，注释前代码部分逐字保留，注释只输出 `[warning: operand missing]`。 |
| 异常输入 | 在两个单元测试文件中检查异常标签、重复分隔符、未知操作码和非汇编文本。 | 正确区分 `unknown_opcode` 与 `malformed`，注释前代码部分逐字保留，分别输出 `[warning: unknown opcode]` 与 `[warning: malformed instruction]`。 |
| 字段长度与格式化 | 测试字段长度统计、列对齐、段与函数结构标记。 | 长度统计和输出位置正确，美化前后语义保持不变。 |
| 语义注释 | 在 `tests/test_asm_beautifier.py` 中测试指令注释、异常警告、原始注释保留及 `nop` 注释处理。 | 正常指令注释符合语义；三种异常状态只输出对应警告，不含模板结果；原始注释不丢失，已有说明的 `nop` 不追加 `no operation`。 |
| 接口与配置 | 测试 Python API、文件处理、命令行接口及配置开关。 | 各接口和选项行为一致，错误路径能够稳定报告。 |
| 解析器隔离与兼容 | 检查 beautifier 只导入 `asm_parser_for_beautifier.py`，既有 pass 继续导入 `_asm_parser.py`；对双方都支持的合法汇编子集比较标签、opcode、操作数与注释字段。 | 不修改既有 pass 的解析行为；公共字段含义一致，美化器特有状态和原文字段仅由专用解析器提供。 |
| 性能基准 | 在 `benchmarks/bench_asm_beautifier.py` 中生成不同规模的汇编文本并记录指标。 | 性能表现稳定，不出现无法解释的明显退化。 |

### 4.2 验收标准（Definition of Done）

- [ ] 新实现不再依赖单个宽松正则表达式完成整行解析。
- [ ] `asm_beautifier.py` 只从 `asm_parser_for_beautifier.py` 导入解析能力，自身不保留第二份逐行解析或操作数拆分实现。
- [ ] `_asm_parser.py` 及其 peephole、const-merge、inst-counter、inst-scheduler 等既有调用方的接口和行为不变，相关回归测试通过。
- [ ] 五种 `parse_status` 均有直接单元测试和明确输出策略。
- [ ] 测试概述表中的各类测试均达到预期。
- [ ] `tests/test_asm_line_parser.py` 和 `tests/test_asm_beautifier.py` 测试全部通过。
- [ ] `python3 -m pytest tests/ -v` 全部通过，不引入项目回归。
- [ ] 元数据标签保持原样；三种异常行的注释前代码部分逐字保持不变，输出对应警告注释，不生成函数标题或指令模板结果。
- [ ] 已有算子说明的 `nop` 不追加 `no operation`，无注释的普通 `nop` 行为保持不变。
- [ ] 数据字符串、标签、操作数词法内容、括号内部格式和用户注释不被破坏；仅将顶层操作数分隔符统一为 `, `。
- [ ] 开启 ABI 别名仅影响自动注释，默认输出保持向后兼容。
- [ ] `beautify_file()` 同步暴露 `align`、`add_comments`、`abi_register_names`，CLI 使用设计规定的 `--abi-register-names` 正向开关。
- [ ] 四类段标题映射、标题插入顺序和 60 个 `=` 分隔线与设计文档第 2.2 节一致；函数入口识别与标题插入位置与设计文档第 2.4 节一致。
- [ ] 标签、操作码和操作数的填充宽度分别为 `min(实际最大长度, 30/12/40)`；超长字段保持原样且允许后续字段右移，不与常规字段对齐。
- [ ] CLI 文件错误返回非零状态码，失败写入不会留下不完整目标文件。
- [ ] 基准结果已记录且没有无法解释的明显时间或输出体积退化。
- [ ] 公共 API、CLI 和相关用户文档已更新，代码包含必要类型标注和文档字符串。

---

## 5. 风险评估与依赖

| 风险项 | 影响程度 | 缓解措施 |
|--------|----------|----------|
| 字符串、注释和操作数边界处理错误，导致合法数据被改写 | 高 | 使用状态扫描器并补充边界与异常输入测试。 |
| 未知扩展指令被误分类并生成错误语义 | 高 | 使用显式指令规格表；未登记操作码保留标签、操作码和操作数内容，只输出 `[warning: unknown opcode]`，不提供 fallback 或模板注释。 |
| 专用解析器与 `_asm_parser.py` 随时间产生无意漂移 | 中 | 明确调用边界；公共合法汇编子集增加字段兼容性测试；美化器特有差异在设计文档和测试名称中显式记录。 |
| 普通跳转标签被误判为函数 | 中 | 结合段状态、符号指示和调用目标分析函数，不确定时不标记。 |
| 算子元数据标签被拆分或误判为函数 | 高 | 在普通标签和操作码解析前识别元数据标签，并覆盖正确识别与误判场景。 |
| 数据定义被当作运行时指令处理 | 高 | 独立维护汇编器指示和数据定义集合，并结合当前段状态禁用语义注释。 |
| 不同指令的操作数角色映射错误 | 高 | 由 `InstructionSpec.operand_roles` 显式驱动，分别测试算术、访存、分支、跳转和伪指令。 |
| 新 ABI 参数破坏已有位置参数调用或默认输出 | 中 | 新参数追加到签名末尾且默认 `False`，增加兼容性测试。 |
| 全文件两遍扫描增加大输入内存和耗时 | 中 | 复用解析结果，复杂度保持 O(n)；用 1000/5000 条输入监测耗时和输出膨胀。 |
| 文件输出失败留下部分文件 | 中 | 同目录临时文件写入成功后使用原子替换，并在失败路径清理临时文件。 |

- **外部依赖**：实现仅使用 Python 3.12+ 标准库，不新增第三方运行时依赖。
- **对现有功能的兼容性**：保留现有函数名、默认参数和 CLI 开关；新 ABI 选项默认关闭。

---

## 6. 开发进度跟踪

| 阶段 | 计划完成日期 | 状态（待开始/进行中/已完成） |
|------|--------------|------------------------------|
| 技术设计编写 | 2026-07-10 | ✅ 已完成 |
| 测试用例编写 | 2026-07-24 | ✅ 已完成 |
| 编码实现 | 2026-08-23 | ⬜ 进行中 |
| 自测与调试 | 2026-08-31 | ⬜ 进行中 |
| 代码审查（PR） | 待排期 | ⬜ 待开始 |
| 合并主分支 | 待排期 | ⬜ 待开始 |

---

## 7. 附录

### 7.1 参考资料

- 《ScratchV RISC-V 汇编代码美化器技术设计文档》v1.2
- `docs/topics/05-汇编代码美化器.md`
- RISC-V Assembly Programmer's Manual
- RISC-V ELF psABI Specification
- RISC-V Instruction Set Manual
