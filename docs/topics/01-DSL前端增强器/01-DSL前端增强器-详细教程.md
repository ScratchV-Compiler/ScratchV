# 课题1：DSL前端增强器（if/else + while）

> **难度**：中 | **类型**：项目实战 | **源文件**：`scratchv/frontend/dsl_extended.py`
> **状态**：✅ 已完成

> 上述状态指已有的初版解析器。后续控制流语义完善方案见
> [设计文档](01-DSL前端增强器-设计文档.md)与[开发文档](01-DSL前端增强器-开发文档.md)，尚未实施。

---

## 概述

为现有 DSL 增加条件判断（`if/else`）和循环（`while`）语法，扩展解析器并生成对应的三地址码（IR）。这是 ScratchV 从"表达式计算器"升级为"图灵完备编译器"的关键一步。

---

## 理解背景

### 是什么？

DSL 前端增强器（`dsl_extended.py`）在基础 DSL 解析器之上扩展了**控制流**能力——新增 `if/else` 条件分支和 `while` 循环。它用**递归下降**方式解析新语法，生成带有标签和条件跳转的 IR。

基础 DSL 只能写线性计算的"计算器"：
```
a = const(3)
b = const(5)
c = add(a, b)
return c
```

增强后可以写带逻辑的"程序"：
```
if (a > b):
    c = add(a, b)
else:
    c = mul(a, b)
endif
return c
```

### 为什么？

真实世界的程序都有分支和循环。没有 `if/else`，你只能写死计算路径；没有 `while`，你只能写固定次数的 `for`。

这个模块让 ScratchV 从"表达式计算器"升级为"图灵完备的编译器"——可以表达任意算法。

### 核心概念

#### 1. 语法设计

```
if_stmt    → 'if' '(' cond ')' ':' block ('else' ':' block)? 'endif'
while_stmt → 'while' '(' cond ')' ':' block 'endwhile'
cond       → operand ('==' | '!=' | '<' | '>' | '<=' | '>=') operand
block      → (statement)+
```

`CondExpr` 类封装条件表达式：
```python
class CondExpr:
    def __init__(self, lhs: str, op: str, rhs: str):
        self.lhs = lhs.strip()   # 左操作数（变量名或数字）
        self.op = op.strip()     # 比较运算符
        self.rhs = rhs.strip()   # 右操作数
```

#### 2. IR 生成模式

**if/else**:
```
entry:
    cmp = cmp(a, b)           # 比较
    br_if cmp -> if_then, if_else
.if_then:
    c = add a b
    br -> if_end
.if_else:
    c = mul a b
    br -> if_end
.if_end:
    ret c
```

**while**:
```
entry:
    br -> while_hdr
.while_hdr:
    cmp = cmp(i, 10)          # 先判断条件
    br_if cmp -> while_body, while_exit
.while_body:
    acc = add acc x           # 循环体
    br -> while_hdr            # 跳回判断
.while_exit:
    ret acc
```

#### 3. 嵌套处理

`if` 和 `while` 可以任意嵌套。用全局计数器确保标签唯一：
```python
self._label_counter = 0

def _new_label(self, prefix):
    self._label_counter += 1
    return f".{prefix}_{self._label_counter}"
```

---

## 详细任务

1. 理解现有 `dsl_parser.py`（递归下降解析）和 `ir_builder.py`。
2. 设计新语法的 BNF 规则：`if_stmt -> 'if' '(' cond ')' block ('else' block)?`，`while_stmt -> 'while' '(' cond ')' block`。
3. 修改解析器，增加 `parse_if()` 和 `parse_while()` 方法，构建 AST 节点（`IfNode`, `WhileNode`）。
4. 扩展 IR 生成：为 `IfNode` 生成条件跳转（`BR cond, label_then, label_else`）和标签；为 `WhileNode` 生成循环结构。
5. 处理嵌套语句，确保标签编号唯一。
6. 编写至少 3 个完整的 DSL 程序（含分支和循环），使用后端生成汇编并用模拟器验证。

---

## 交付产物

- 增强后的 `dsl_parser.py` 和 `ir_builder.py`
- 示例 DSL 程序（`if_else.dsl`, `while_sum.dsl`, `nested_loop.dsl`）
- 文档：新增语法说明、使用示例

---

## 代码走读

### 核心：`parse_if()` 方法

```python
def parse_if(self):
    self._consume("if")
    self._consume("(")
    lhs = self._parse_primary()     # 左操作数
    op = self._current_token()      # ==, !=, <, >, <=, >=
    self._advance()
    rhs = self._parse_primary()     # 右操作数
    self._consume(")")
    self._consume(":")

    cond = CondExpr(lhs, op, rhs)

    # 生成 IR: 比较指令 + 条件跳转
    cmp_val, cmp_instr = self._build_cmp(cond)
    label_then = self._new_label("if_then")
    label_else = self._new_label("if_else")
    label_end = self._new_label("if_end")

    self._emit_br_if(cmp_val, label_then, label_else)

    # then 分支
    self._emit_label(label_then)
    self.parse_block()  # 递归解析 then 块
    self._emit_jump(label_end)

    # else 分支（可选）
    self._emit_label(label_else)
    if self._peek() == "else":
        self._consume("else")
        self._consume(":")
        self.parse_block()  # 递归解析 else 块
    self._emit_jump(label_end)

    self._emit_label(label_end)
    self._consume("endif")
```

### while 循环的 IR 生成

关键区别：while 的条件在循环体**之前**检查，循环体末尾跳回条件检查（不是跳回循环体开头）。

```python
def parse_while(self):
    label_hdr = self._new_label("while_hdr")
    label_body = self._new_label("while_body")
    label_exit = self._new_label("while_exit")

    self._emit_jump(label_hdr)     # 跳到条件检查

    self._emit_label(label_hdr)    # 条件检查
    cmp_val = self._build_cmp(cond)
    self._emit_br_if(cmp_val, label_body, label_exit)

    self._emit_label(label_body)   # 循环体
    self.parse_block()
    self._emit_jump(label_hdr)     # 跳回条件检查

    self._emit_label(label_exit)
```

---

## 动手练习

### 练习 1: 写一个嵌套 if/while 程序

写一个 DSL 程序：如果 `a > 0`，用 while 计算 `a` 的累加；否则直接返回 0。

### 练习 2: 用 TinyFive 验证

把你写的 DSL 程序编译成 RISC-V 汇编，用 TinyFive 仿真执行，验证结果是否正确。

### 练习 3: 添加 `for` 循环语法糖

在 `ExtendedDSLParser` 中添加 `for i = start to end:` 语法，自动展开为 while 循环的 IR。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **标签冲突** | 嵌套结构中的标签必须全局唯一，用计数器而不是固定标签名 |
| **条件中的嵌套表达式** | 当前条件只支持 `变量 op 变量/常量`，不支持 `(a+b) > (c+d)` |
| **else 分支可选** | `if` 可以没有 `else`，此时 else 分支直接跳到 `if_end` |
| **递归解析** | `parse_block()` 自身会调用 `parse_if()` 和 `parse_while()`，形成递归下降——确保不会无限递归 |

---

## 进阶阅读

- 龙书第 4 章：Syntax Analysis（递归下降解析原理）
- DSL 语法参考：[ScratchV DSL 文档](../../reference/developer_guide.md)
- 相关 topic: [课题9 — DSL 错误提示美化器](../09-DSL错误提示美化器/09-DSL错误提示美化器.md) | [课题3 — IR 系统](../../archive/后续课题/03-IR系统/03-IR系统.md) | [课题11 — 控制流图生成器](../11-控制流图生成器/11-控制流图生成器.md)

---

## 12周每周目标

- **W1**：搭建环境，运行现有 DSL 示例。阅读 `dsl_parser.py` 和 `ir_builder.py`，画出现有流程思维导图。
- **W2**：学习递归下降解析原理，为 `if` 语法设计 BNF 规则，编写伪代码。
- **W3**：添加 `parse_if()` 方法，识别 `if` 关键字和括号，构建 `IfNode`（简单存储条件、then 块、else 块）。
- **W4**：实现条件表达式的解析（支持 `==, <, >` 等），输出 AST 结构。
- **W5**：学习项目 IR 表示（三地址码，含 `BR`, `LABEL`）。为 `IfNode` 编写 IR 生成函数。
- **W6**：实现 `if-else` 完整 IR 生成（两个分支，汇合标签）。测试简单 `if` 程序。
- **W7**：添加 `while` 语法，解析为 `WhileNode`。设计 IR 模式：条件判断→循环体→跳回。
- **W8**：实现 `while` 的 IR 生成，确保退出条件正确。测试 `while` 求和程序。
- **W9**：处理嵌套 `if` 和 `while`，确保标签编号不冲突（使用计数器）。测试嵌套例子。
- **W10**：增加错误恢复（可结合课题 9 的成果），完善注释。
- **W11**：编写 3 个完整 DSL 程序，使用后端生成汇编，用 `tinyfive.py` 验证结果。
- **W12**：撰写文档（使用说明、新增语法示例、内部设计图），准备演示。
