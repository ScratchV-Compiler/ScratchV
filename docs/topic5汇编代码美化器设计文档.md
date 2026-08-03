# ScratchV RISC-V 汇编代码美化器技术设计文档

> 文档版本：v1.2
> **创建日期**：2026-07-13
> 更新日期：2026-08-02
> 涉及模块：`scratchv/backend/asm_beautifier.py`（汇编代码美化器）、`scratchv/backend/asm_parser_for_beautifier.py`（美化器专用汇编解析器）
> 功能范围：RISC-V 汇编列对齐、语义注释、分段标记、命令行美化工具

---

## 一、功能介绍

### 1.1 功能概述

#### RISC-V 汇编列对齐

- RISC-V 汇编列对齐用于将编译器生成的原始 `.s` 汇编文件整理为**字段清晰、缩进统一、列宽稳定**的文本格式。其核心功能是：识别汇编行中的标签、操作码、操作数和注释，先扫描全部汇编行得到标签列、操作码列和操作数列的最大宽度，再按固定宽度列重新排版，使汇编代码更适合人工阅读、调试和教学展示。
- 列对齐解决了机器生成汇编中常见的字段位置混乱问题，例如指令缩进不一致、标签与指令混排、不同长度操作码导致操作数列不齐等。美化器只对标签、操作码和操作数字段做列级对齐，不重写操作数内部的逗号和空格内容；通过固定宽度排版，开发者可以更快速地观察指令序列、寄存器使用和内存访问模式。

#### RISC-V 指令语义注释

- 语义注释用于为常见 RISC-V 指令自动生成清晰、可读的解释说明。其核心功能是：根据指令助记符和操作数角色，将 `add rd, rs1, rs2`、`lw rd, imm(rs1)`、`beq rs1, rs2, label` 等指令转换为接近高级语言表达式的注释。
- 语义注释降低了初学者阅读汇编代码的门槛，也便于编译器开发者检查后端代码生成是否符合预期。例如 `addi sp, sp, -32` 可注释为 `sp = sp + -32`，`sw ra, 28(sp)` 可注释为 `MEM[sp + 28] = ra`。

#### ABI 寄存器别名注释（可选）

- 美化器可选择在**自动生成的语义注释**中，将通用寄存器编号转换为 RISC-V ABI 别名，例如 `x1 -> ra`、`x2 -> sp`、`x5 -> t0`、`x10 -> a0`。该功能帮助读者快速理解返回地址、栈指针、临时寄存器和参数寄存器的角色。
- 该选项只影响自动注释中的寄存器显示方式，不得改写原始汇编的操作数字段、标签、用户注释或程序语义。默认关闭，以保持现有输出兼容；开启后，已使用 ABI 别名的操作数（如 `a0`、`sp`）保持不变。

#### 汇编分段标记

- 分段标记用于识别 `.text`、`.data`、`.bss`、`.rodata` 等汇编段，以及函数入口标签，并插入清晰的分隔标题。其核心功能是：将汇编文件划分为代码段、可写数据段、未初始化数据段、只读数据段和函数块，使长汇编文件结构更加清楚。
- 段标题必须与实现保持一致：`.text` 输出 `CODE SECTION`，`.data` 输出 `DATA SECTION`，`.bss` 输出 `BSS SECTION`，`.rodata` 输出 `READ-ONLY DATA SECTION`。
- 分段标记适用于大型模型或复杂 DSL 程序生成的汇编文件。对于包含大量函数、标签和数据声明的输出文件，分段标记可以帮助用户快速定位函数入口、代码段和数据段。

### 1.2 设计目标

- **语义保持**：美化器只改变汇编文本格式和注释内容，不改变指令顺序、标签名称、操作数含义和程序执行语义。
- **阅读友好**：输出结果应便于人工阅读，标签、操作码、操作数和注释列清晰分离。
- **寄存器可读性可选增强**：支持在自动语义注释中以 ABI 别名展示 `x0` 至 `x31`，同时保证原始汇编文本不被改写。
- **指令覆盖充分**：内置常见 RISC-V 整数指令、访存指令、分支跳转指令的注释模板；对 `li`、`mv`、`call`、`ret` 等汇编器伪指令单独提供模板，避免把语法糖误当作真实机器指令解释。
- **工具独立**：既支持 Python API 调用，也支持命令行方式处理 `.s` 文件。
- **性能可预测**：能够处理不同规模的汇编文件，对大规模输入保持稳定的处理耗时。
- **原始注释保留**：已有行尾注释必须保留；若同时生成自动注释，二者以 ` | ` 分隔。

---

## 二、语法规范

### 2.1 汇编行的语法定义

**BNF 表示**：
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

`metadata_path` 可包含 `/`（例如 `layer1.0/Conv_5`），但不得包含空白或 `:`。操作数只在括号深度为零时按逗号拆分；括号内的逗号属于当前操作数，不是分隔符。

| 元素 | 说明 |
|------|------|
| `label` | 汇编标签，用于表示函数入口或跳转目标，例如 `main:`、`.L1:` |
| `metadata_label` | ScratchV 生成的算子边界标签，例如 `_op_/layer1.0/Conv_5:`、`_op_PPQ_Operation_6_29:`；用于标记算子代码块，不是函数入口 |
| `opcode` | 操作码、汇编器伪指令或汇编器指示，例如 `add`、`addi`、`lw`、`sw`、`li`、`mv`、`call`、`.text` |
| `operands` | 操作数序列，可以是寄存器、立即数、标签或内存寻址表达式 |
| `comment` | 原始注释内容，以 `#` 开始 |
| `instruction` | RISC-V 真实指令，例如 `add`、`lw`、`addi` |
| `pseudo_instruction` | 汇编器伪指令，是对真实 RISC-V 指令序列的语法糖，例如 `li`、`mv`、`call`、`ret` |
| `directive` | 汇编器指示，不对应运行时指令，例如 `.text`、`.data`、`.bss`、`.rodata`、`.globl main` |

**合法示例**：
```
main:
addi sp,sp,-32
sw ra,28(sp)
li a5,3
add a5,a5,a4
```

### 2.2 美化输出的格式定义

**BNF 表示**：
```
pretty_line ::= [aligned_label] [aligned_opcode] [aligned_operands] [aligned_comment]
section_mark ::= section_bar newline section_title newline section_bar
section_bar ::= "# " "=" * 60
section_title ::= "#  " section_name " SECTION"
function_mark ::= "# --- Function: " function_name " ---"
function_block ::= function_mark newline function_label newline function_body
function_label ::= function_name ":"
```

| 元素 | 说明 |
|------|------|
| `aligned_label` | 标签列左对齐，标签后保留冒号；填充宽度最大为 30 字符，超长标签保持原样输出 |
| `aligned_opcode` | 操作码列左对齐；填充宽度最大为 12 字符，超长操作码保持原样输出 |
| `aligned_operands` | 操作数字符串作为整体左对齐；填充宽度最大为 40 字符，超长操作数字段保持原样输出，不改写操作数内部格式 |
| `aligned_comment` | 原始注释或自动生成的语义注释 |
| `section_bar` | 段标记用`=`隔开 |
| `section_title` | 段标记 |
| `function_mark` | 函数入口标记，用于突出 `main` 等函数标签 |
| `function_block` | 函数标记后，函数标签独占一行；函数体无标签指令从行首开始，不与函数标签同行，也不添加前导缩进 |

段标题与原始 directive 的输出顺序是强制契约：段标题紧邻插入在对应 directive 之前，原始 `.text`、`.data`、`.bss`、`.rodata` 行完整保留，directive 之后插入一个空行。四类映射如下：

| 原始 directive | `section_title` |
|----------------|-----------------|
| `.text` | `#  CODE SECTION` |
| `.data` | `#  DATA SECTION` |
| `.bss` | `#  BSS SECTION` |
| `.rodata` | `#  READ-ONLY DATA SECTION` |

对已经紧邻同类标题的 directive 不重复插入标题，保证重复运行美化器不会不断叠加段标题。

**合法示例**：
```
# --- Function: main ---
main:
addi  sp, sp, -32     # sp = sp + -32
sw    ra, 28(sp)      # MEM[sp + 28] = ra
li    a5, 3           # a5 = 3
add   a5, a5, a4      # a5 = a5 + a4
```

### 2.3 Python API 与数据结构

美化器所需的解析能力由专用模块 `scratchv/backend/asm_parser_for_beautifier.py` 提供。现有 `scratchv/backend/_asm_parser.py` 已被 peephole、const-merge、inst-counter、inst-scheduler 等汇编级 pass 使用，其数据结构和宽松解析行为属于既有兼容契约；美化器需要更强的原文保真、字符串感知、元数据标签和错误状态分类，因此不修改或替换 `_asm_parser.py`。专用解析器的目标接口如下：

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

def parse_asm_line(line: str, lineno: int = 0) -> ParsedAsmLine: ...
def parse_asm(asm_text: str) -> list[ParsedAsmLine]: ...

def beautify_asm(
    asm_text: str,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
) -> str: ...

def beautify_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
    encoding: str = "utf-8",
) -> str: ...
```

`raw` 始终保存不含行尾换行符的原始行；`label` 不含末尾冒号；`comment` 不含 `#` 和紧随其后的分隔空格。`field_lengths` 按上述规范化字段计算，缺失字段长度为 `0`。`beautify_file()` 始终返回美化结果；指定 `output_path` 时，先写入同目录临时文件，成功后原子替换目标文件，失败时不得留下部分输出。

两个解析器的职责边界如下：

| 模块 | 调用方 | 稳定契约 | 本课题处理方式 |
|------|--------|----------|----------------|
| `_asm_parser.py` | peephole、const-merge、inst-counter、inst-scheduler 等既有 pass | 通用标签、opcode、操作数和注释解析，以及现有 `ParsedAsmLine` 行为 | 不修改、不迁移调用方、不增加美化器专用字段 |
| `asm_parser_for_beautifier.py` | 仅 `asm_beautifier.py` | `raw`/`operands_str` 精确保留、保留 directive 前导 `.` 的 `opcode`、引号与括号状态扫描、元数据标签、五种 `parse_status`、`field_lengths` | 新建并独立测试 |

`asm_beautifier.py` 必须从专用模块导入 `ParsedAsmLine`、`parse_asm_line()` 和 `parse_asm()`，自身不得保留逐行解析正则、注释切分器或操作数拆分器。两个解析器不得互相导入。其他汇编级 pass 不得在本课题中改为依赖专用解析器，避免扩大变更范围。

对于双方都支持的普通合法汇编子集，标签、opcode、操作数和注释的字段含义应保持一致；美化器特有的 `operands_str`、`parse_status`、`field_lengths`、字符串字面量保护和元数据标签规则属于有意差异，不要求反向加入 `_asm_parser.py`。

### 2.4 段与函数入口识别

函数标题只能在当前段为 `.text` 且目标标签存在于本文件时生成。识别按以下优先级执行：

1. 匹配 `_op_/...` 或 `_op_PPQ_Operation_<number>_<number>` 的元数据标签永远不是函数，即使名称同时出现在其他指示或调用中。
2. `.type <symbol>, @function` 明确声明的符号是函数。
3. `.globl <symbol>` 或 `.global <symbol>` 声明且在 `.text` 段定义的符号是函数。
4. 直接 `call <symbol>` 的目标在本文件 `.text` 段存在同名定义时，识别为函数。
5. `.L1`、`L1`、`loop1`、`_L1` 等局部或控制流标签，以及仅作为分支目标出现的普通标签，不生成函数标题；其他无法确定的标签采用保守策略，也不生成标题。

显式声明优先于命名启发式；实现不得仅凭“标签不以点号开头”判定函数。函数标题插入在函数标签之前；`.globl`、`.type` 等原始指示保持原位置，不移动到函数块内部。

### 2.5 约束规则

- 美化器不得删除、重排或替换原始指令。
- 标签名称必须保持不变，跳转目标不得被改写。
- `.text`、`.data`、`.bss`、`.rodata`、`.globl` 等汇编器指示必须保留。
- **数据段处理**：美化器必须识别 `.data`、`.bss`、`.rodata` 段，并在段切换处插入对应标题；数据段内的 `.asciz`、`.string`、`.word`、`.half`、`.byte`、`.float`、`.double`、`.zero`、`.space`、`.align` 等数据定义行属于汇编器指示，不参与列宽扫描、字段对齐或操作数空格规范化，也不生成运行时语义注释，整行按 `raw` 原样输出。无法可靠分离字符串与注释的行标记为 `malformed`，同样逐字节保留 `raw`。
- 对 `unknown_opcode` 与 `malformed` 行，必须原样输出，不得导致程序崩溃。
- 原始注释必须保留；若同时开启自动注释，应避免覆盖用户已有注释。
- 内存操作数如 `28(sp)` 必须作为整体识别，不能被错误拆分。
- **ABI 寄存器别名开关**：默认使用输入中的寄存器写法；启用 ABI 别名后，仅自动语义注释中的完整通用寄存器 token（`x0` 至 `x31`）按 ABI 约定替换，如 `x1 -> ra`、`x2 -> sp`、`x10 -> a0`。内存操作数中的基址寄存器也应适用该规则，例如 `0(x2)` 的注释显示为 `MEM[sp + 0]`；立即数、标签、字符串和原始汇编操作数不得转换。
- **未知指令**：不在真实指令、汇编器伪指令或 ScratchV 已登记自定义伪指令模板中的操作码，必须将整行原样输出，不进行列对齐、不追加自动语义注释，也不得套用 `opcode; operands` 等 fallback 注释，避免误导或改变未识别扩展的文本格式。
- **操作数数量不足**：对于 `add a0`、`lw t0`、`beq a0,a1` 等操作码可识别但操作数不足的行，不做严格合法性校验，也不得导致程序崩溃。输出必须保留原始操作数字符串；注释生成阶段可在内部用空字符串补齐缺失位置，并生成包含空白位置的尽力说明。不得删除、猜测或补写缺失的原始操作数。

### 2.6 异常输入处理

美化器的异常输入策略是“保守保留、不中断后续行”。解析器应为每行记录 `parse_status`，至少区分 `valid`、`metadata_label`、`incomplete_operands`、`unknown_opcode` 和 `malformed`，并按以下规则处理：

- **算子元数据标签**：完整匹配 `_op_/...:` 或 `_op_PPQ_Operation_<number>_<number>:` 的标签标记为 `metadata_label`，整行原样保留且不得被识别为函数入口。`_input_copy_done:`、`_mp_done:` 等阶段或控制流标签仍按普通标签处理。
- **缺少操作数**：操作码和已有操作数仍按原样输出；内部可用空字符串填充模板占位符，保证注释生成流程稳定，但不修改原始指令文本。
- **重复逗号、异常标签、非汇编文本或字段边界不明确**：例如 `li,,a0,,3`、`main::`、`this is not asm`。这类行标记为 `malformed`，必须原样输出，不生成自动语义注释，也不得把其中的一部分误识别为未知操作码。
- **未知但结构完整的操作码**：例如 `custom_op x1,x2,x3`。这类行标记为 `unknown_opcode`，不属于 `malformed`，但必须整行原样输出，不追加自动语义注释；已有用户注释保持不变。
- **文件级错误**：输入文件不存在、无法读取或输出文件无法写入时，命令行工具应向标准错误输出清晰的路径和原因，并以非零状态码退出；不得输出不完整文件。

### 2.7 命令行接口

```bash
python -m scratchv.backend.asm_beautifier input.s
python -m scratchv.backend.asm_beautifier input.s -o output.s
python -m scratchv.backend.asm_beautifier input.s --no-align
python -m scratchv.backend.asm_beautifier input.s --no-comments
python -m scratchv.backend.asm_beautifier input.s --abi-register-names
```

| 参数 | 默认值 | 行为 |
|------|--------|------|
| `input` | 无，必需 | UTF-8 汇编输入路径。 |
| `-o`, `--output` | 未指定 | 指定时原子写入该路径；未指定时只向标准输出写入结果。 |
| `--no-align` | 关闭 | 传入后设置 `align=False`。 |
| `--no-comments` | 关闭 | 传入后设置 `add_comments=False`，但保留用户原始注释。 |
| `--abi-register-names` | 关闭 | 传入后设置 `abi_register_names=True`，仅改变自动注释。 |

成功时退出码为 `0`。参数错误由 `argparse` 输出到标准错误并返回 `2`；文件读取、编码或写入失败输出包含目标路径与原因的单行错误并返回 `1`。失败时标准输出不得混入部分美化结果。

---

## 三、测试设计

测试资产不拆分为独立汇编样例文件，统一放在两个单元测试文件和一个性能基准文件中：

| 测试 | 描述 | 预期 |
|------|------|------|
| 正常输入解析 | 在 `tests/test_asm_line_parser.py` 中测试普通指令、标签、指示、注释、空行及操作数拆分。 | 各字段提取正确，`parse_status` 为 `valid`。 |
| 元数据标签解析 | 在 `tests/test_asm_line_parser.py` 中测试算子元数据标签与普通内部标签的区分。 | 元数据标签完整保留并标记为 `metadata_label`，且不被识别为函数入口。 |
| 操作数缺失 | 在两个单元测试文件中检查已知操作码缺少操作数时的解析和输出。 | 标记为 `incomplete_operands`，保留已有内容且不中断处理。 |
| 异常输入 | 在两个单元测试文件中检查异常标签、重复分隔符、未知操作码和非汇编文本。 | 正确区分 `unknown_opcode` 与 `malformed`，不得改写原始行。 |
| 字段长度与格式化 | 测试字段长度统计、列对齐、段与函数结构标记。 | 长度统计和输出位置正确，美化前后语义保持不变。 |
| 语义注释 | 在 `tests/test_asm_beautifier.py` 中测试指令注释、原始注释保留及 `nop` 注释处理。 | 注释符合指令语义，原始注释不丢失，已有说明的 `nop` 不追加 `no operation`。 |
| 接口与配置 | 测试 Python API、文件处理、命令行接口及配置开关。 | 各接口和选项行为一致，错误路径能够稳定报告。 |
| 解析器隔离与兼容 | 静态检查 beautifier 只导入专用解析器，既有 pass 继续导入 `_asm_parser.py`；对双方支持的普通合法汇编比较公共字段。 | 调用边界不串用；公共字段含义一致，美化器专有差异有独立测试。 |
| 性能基准 | 在 `benchmarks/bench_asm_beautifier.py` 中生成不同规模的汇编文本并记录指标。 | 性能表现稳定，不出现无法解释的明显退化。 |

## 四、修改模块与实现步骤

### 4.1 涉及文件

- [asm_beautifier.py](/ScratchV/scratchv/backend/asm_beautifier.py)：负责逐行解析、列宽扫描、格式化输出、语义注释、段标题及命令行入口。
- [asm_parser_for_beautifier.py](/ScratchV/scratchv/backend/asm_parser_for_beautifier.py)：美化器专用解析模块，负责原文字段保留、字符串与括号状态扫描、元数据标签和解析状态；不替换现有 `_asm_parser.py`。
- [test_asm_line_parser.py](/ScratchV/tests/test_asm_line_parser.py)：负责验证解析字段、字段长度、操作数拆分、元数据标签和异常状态分类。
- [test_asm_beautifier.py](/ScratchV/tests/test_asm_beautifier.py)：集中保存格式化、语义注释、结构标记、文件接口和 CLI 用例，汇编输入以内联字符串或参数化数据提供。
- [bench_asm_beautifier.py](/ScratchV/benchmarks/bench_asm_beautifier.py)：在内存中生成不同规模的汇编输入并记录性能指标，不依赖独立样例 `.s` 文件。

### 4.2 语法与处理约定

实现必须遵循第 2.1 至 2.7 节的输入语法、输出格式、API、结构识别、约束、异常策略和 CLI 契约；BNF 与行为规格只在第二章维护，避免设计与实现章节出现重复定义后发生漂移。

主流程依次处理空行、纯注释、普通标签、算子元数据标签、指令、汇编器伪指令、汇编器指示和带行尾注释的指令，并为每行设置 `parse_status`。

### 4.3 新增美化器专用解析器

专用模块 `asm_parser_for_beautifier.py` 的 `parse_asm_line()` 需要把每一行汇编拆分为结构化字段，字段契约以第 2.3 节为准，包括：

- `raw`：未经改写的原始输入行；
- `label`：可选标签字段，去掉末尾冒号后保存；
- `opcode`：真实指令、汇编器伪指令或汇编器指示字段；汇编器指示保留前导 `.`；
- `operands_str`：原始操作数字符串，用于后续列宽统计和输出；
- `operands`：按逗号拆分后的操作数列表，用于语义注释生成；
- `comment`：去掉 `#` 及分隔空格后的行尾注释内容；
- `lineno`: 解析的代码行数序号，便于输出定位错误；
- `parse_status`：解析状态，取值为 `valid`、`metadata_label`、`incomplete_operands`、`unknown_opcode` 或 `malformed`；
- `field_lengths`：`label`、`opcode`、`operands_str`、`comment` 四个字段的字符长度，用于后续列宽统计。

操作数拆分时需要识别括号深度，保证 `28(sp)`、`0(a0)` 这类内存寻址表达式作为一个整体处理，不被错误拆开。行尾注释分隔符 `#` 仅在字符串字面量和括号之外生效，确保 `.asciz "a#b"` 等数据定义保持完整。解析器只识别字段边界，不重写操作数内部的空格或逗号格式。

解析前先识别完整标签行，再分类标签类型。完整匹配 `^_op_/[^:\s]+$` 或 `^_op_PPQ_Operation_\d+_\d+$` 的标签标记为 `metadata_label`；标签内容必须完整保存，不能按 `/` 或 `.` 拆分。`_input_copy_done`、`_mp_done` 等不匹配上述格式的内部标签仍标记为 `valid`。出现重复逗号、重复标签冒号、无法确定字段边界等情况时，标记为 `malformed`，保留 `raw` 行并跳过列对齐和自动语义注释。`field_lengths` 按解析后的字段值计算：标签不含冒号，注释不含 `#` 和分隔空格，缺失字段记为 `0`。

本课题不得修改 `_asm_parser.py` 来承载上述字段和状态，也不得把 peephole、const-merge 等既有调用方迁移到专用解析器。为控制两套解析器的维护风险，测试应覆盖：导入边界、普通合法汇编的公共字段兼容性，以及字符串、元数据和异常输入等美化器有意扩展场景。

### 4.4 列宽扫描与对齐输出

列对齐采用两阶段处理：

1. **第一阶段：扫描列宽**。先解析全部汇编行，仅对解析状态为 `valid` 或 `incomplete_operands` 的运行时指令与伪指令统计 `label`、`opcode` 和规范化后操作数字符串的最大长度，得到 `max_label`、`max_opcode` 和 `max_operands`；用于填充的列宽上限分别为 30、12 和 40。`DIRECTIVES` 集合中的汇编器指示不参与扫描。
2. **第二阶段：格式化输出**。对参与扫描的指令行，将操作码和操作数字段按对应列宽左对齐，顶层操作数逗号后统一保留一个空格。所有普通标签必须单独输出，不能与指令同行；无标签指令从行首开始，不添加前导缩进。已有行尾注释保留；自动注释存在时，以 ` | ` 分隔。纯注释补充前导空格，使 `#` 与指令注释列对齐。空行、汇编器指示、`metadata_label`、`unknown_opcode` 和 `malformed` 行不参与列对齐，按 `raw` 原样输出。元数据标签不得触发函数标题。

数据定义行和其他汇编器指示按第 2.5 节原样输出，不参与字段对齐。段标题必须在原始分段 directive 之前输出，directive 本身完整保留，随后输出一个空行。函数入口严格使用第 2.4 节的符号规则。

列宽上限只限制填充宽度，不截断标签、操作码或操作数字符串。超长字段保持原样输出，允许超过视觉列宽，以保证美化后的 `.s` 文件仍可被汇编并保持原有语义。开启 `--no-align` 时跳过列宽扫描与对齐，按原字段顺序输出。

### 4.5 语义注释模板与伪指令处理

语义注释模板（使用字典实现）应按指令类别分组维护：

- **真实 RISC-V 指令模板**：覆盖整数算术、位运算、访存、分支跳转、乘除法扩展和浮点访存/计算等指令，例如 `add`、`addi`、`lw`、`sw`、`beq`、`jal`、`mul`、`flw`。
- **汇编器伪指令模板**：覆盖 `li`、`mv`、`call`、`ret`、`nop`、`not`、`neg`、`seqz`、`snez`、`bnez`、`beqz` 等语法糖。它们不是真正的 RISC-V 机器指令，通常会被汇编器展开为一条或多条真实指令，因此需要单独解释其高层含义。
- **ScratchV 自定义伪指令模板**：覆盖项目内部扩展或教学用途的伪指令，例如 `max`，模板需明确标注其自定义性质。
- **汇编器指示与未知指令处理**：`.text`、`.data`、`.bss`、`.rodata`、`.globl`、`.loc` 等 `DIRECTIVES` 集合中的指示不参与列宽扫描、字段对齐或操作数空格规范化，不按运行时指令解释，也不生成自动语义注释，始终按 `raw` 原样输出；其中分段指示须在原指示之前分别插入 `CODE`、`DATA`、`BSS`、`READ-ONLY DATA` 标题。未知 opcode 标记为 `unknown_opcode` 后整行原样输出，不套用任何具体语义模板或 fallback 注释。

模板匹配时先去掉 opcode 前导的 `.` 用于查表，但必须区分汇编器指示和真实指令/伪指令的语义类别。对 `li`、`mv`、`call` 等伪指令，不应按普通三操作数 RISC-V 指令规则解释，而应使用专门的操作数映射规则，例如 `li rd, imm`、`mv rd, rs`、`call symbol`。

注释生成前必须检查 `parse_status`：仅 `valid` 和 `incomplete_operands` 行允许进入指令模板匹配；`metadata_label`、`unknown_opcode` 和 `malformed` 行不生成自动语义注释。汇编器指示也必须跳过模板匹配。

若 `nop` 已经带有原始注释，尤其是紧跟 `metadata_label` 的 `# --- Conv: ...`、`# --- Relu: ...` 等算子说明，则该注释已经表达了该行用途，美化器只保留原始注释，不得再追加 `no operation`。无原始注释的普通 `nop` 仍可使用 `no operation` 自动注释。

对于操作数数量不足的已知指令，模板生成不执行严格参数个数校验。实现上先将操作数列表补齐为空字符串，再统一填充 `{rd}`、`{rs1}`、`{rs2}`、`{imm}` 等占位符；这样即使输入为 `add a0`、`lw t0`、`beq a0,a1` 这类不完整汇编行，也能保持输出流程稳定，并暴露出缺失操作数造成的空白语义位置。

ABI 寄存器别名应在模板占位符填充阶段处理。实现需维护 `x0` 至 `x31` 到 ABI 名称的映射，并仅对 `{rd}`、`{rs1}`、`{rs2}` 和内存寻址表达式中提取出的基址寄存器执行转换；`{imm}`、分支目标和调用目标不得转换。Python API 增加 `abi_register_names: bool = False` 参数，并由 `beautify_asm()` 传递至注释生成逻辑；`beautify_file()` 同步暴露该参数。命令行新增 `--abi-register-names` 开关，启用后传入 `abi_register_names=True`。

### 4.6 集成与回归测试

集成与回归测试不再编写独立的详细用例清单，统一按照第三章表格规定的两个单元测试文件和一个性能基准文件组织和验证。

---
## 五、附录

### 5.1 示例美化输出

**输入汇编**（美化前）：
```
.text
main:
addi sp,sp,-32
sw ra,28(sp)
li a5,3
li a4,5
add a5,a5,a4
ret
```

**生成的美化汇编输出**：
```
# ============================================================
#  CODE SECTION
# ============================================================
.text

# --- Function: main ---
main:
addi  sp, sp, -32     # sp = sp + -32
sw    ra, 28(sp)      # MEM[sp + 28] = ra
li    a5, 3           # a5 = 3
li    a4, 5           # a4 = 5
add   a5, a5, a4      # a5 = a5 + a4
ret                   # return
```

### 5.2 参考资料

- ScratchV 项目文档：`docs/topics/05-汇编代码美化器.md`
- RISC-V Assembly Programmer's Manual
- RISC-V ELF psABI 文档
- RISC-V 指令集手册（分支与跳转指令）
