## 课题1：DSL前端增强器

**难度**：中

**概述**：为现有DSL增加条件判断（`if/else`）和循环（`while`）语法，扩展解析器并生成对应的三地址码（IR）。

**详细任务**：
1. 理解现有`dsl_parser.py`（递归下降解析）和`ir_builder.py`。
2. 设计新语法的BNF规则：`if_stmt -> 'if' '(' cond ')' block ('else' block)?`，`while_stmt -> 'while' '(' cond ')' block`。
3. 修改解析器，增加`parse_if()`和`parse_while()`方法，构建AST节点（`IfNode`, `WhileNode`）。
4. 扩展IR生成：为`IfNode`生成条件跳转（`BR cond, label_then, label_else`）和标签；为`WhileNode`生成循环结构。
5. 处理嵌套语句，确保标签编号唯一。
6. 编写至少3个完整的DSL程序（含分支和循环），使用后端生成汇编并用模拟器验证。

**交付产物**：
- 增强后的`dsl_parser.py`和`ir_builder.py`
- 示例DSL程序（`if_else.dsl`, `while_sum.dsl`, `nested_loop.dsl`）
- 文档：新增语法说明、使用示例

**12周每周目标**：
- **W1**：搭建环境，运行现有DSL示例。阅读`dsl_parser.py`和`ir_builder.py`，画出现有流程思维导图。
- **W2**：学习递归下降解析原理，为`if`语法设计BNF规则，编写伪代码。
- **W3**：添加`parse_if()`方法，识别`if`关键字和括号，构建`IfNode`（简单存储条件、then块、else块）。
- **W4**：实现条件表达式的解析（支持`==, <, >`等），输出AST结构。
- **W5**：学习项目IR表示（三地址码，含`BR`, `LABEL`）。为`IfNode`编写IR生成函数。
- **W6**：实现`if-else`完整IR生成（两个分支，汇合标签）。测试简单`if`程序。
- **W7**：添加`while`语法，解析为`WhileNode`。设计IR模式：条件判断->循环体->跳回。
- **W8**：实现`while`的IR生成，确保退出条件正确。测试`while`求和程序。
- **W9**：处理嵌套`if`和`while`，确保标签编号不冲突（使用计数器）。测试嵌套例子。
- **W10**：增加错误恢复（可结合课题9的成果），完善注释。
- **W11**：编写3个完整DSL程序，使用后端生成汇编，用`tinyfive.py`验证结果。
- **W12**：撰写文档（使用说明、新增语法示例、内部设计图），准备演示。