# 课题 01：DSL 前端增强器设计文档

> 状态：设计提案；本次 PR 仅提交文档，不表示下述增强已经实现。
> 调研基线：`99538fe5059599b337d4ab3da81be5406c29d8c4`（2026-09-18 的 upstream/main）。
> 配套文档：[开发文档](01-DSL前端增强器-开发文档.md) · [课题说明](01-DSL前端增强器.md)

## 1. 目标与边界

课题 01 的目标是让 DSL 的 `if/else`、`while` 及嵌套控制流具有正确、可验证的执行语义。主分支已经有解析器和结构测试；本方案从这些实现继续完善，不能把“成功生成 IR”当成“程序执行正确”。

本阶段覆盖六种标量比较、可选 else、循环零次/一次/多次执行、分支赋值合流、循环变量更新、块内 return、嵌套块及与现有 for 的兼容。保留 `ExtendedDSLParser.parse(text, *, filename=None) -> Program` 和 `validate(...) -> ErrorCollector` 的公开入口，继续使用课题 09 的诊断对象、源码定位和 CLI 输出。

不引入新语言、通用表达式解析、`&&/||`、函数定义、break/continue、数组条件或新的 `for i = start to end` 语法。矩阵和神经网络算子的原有能力不由本课题重新定义。首次交付不以吞掉错误、关闭测试或增加静默回退来扩大支持范围。

## 2. 已有实现与需要补齐的部分

以下结论针对上述基线，后续开发前须重新核对主分支。

| 位置 | 已有能力 | 当前缺口及验证方式 |
| --- | --- | --- |
| `scratchv/frontend/dsl_parser.py` | 逐行识别算子、return、for，使用 `_vars` 映射变量 | 不是带 token 流的完整语法树解析器；赋值会替换映射，不能直接表达跨块的可变变量 |
| `scratchv/frontend/dsl_extended.py` | `CondExpr`、递归解析 if/while、唯一标签 | 尚无 `IfNode`/`WhileNode`；then/else 共用变量映射；while 头引用解析循环体之前的值 |
| `scratchv/frontend/dsl_validator.py` | 无 IR 的语法校验、块栈、错误数量上限 | 条件操作数约束需与解析器统一，避免把表达式字符串当成变量名 |
| `scratchv/ir/builder.py` | BR、BR_IF、LOAD/STORE/ALLOCA 等构建方法 | `br_if` 接收一个条件，扩展解析器却生成双操作数和 `cmp_op`，需要明确契约 |
| `scratchv/backend/instruction_select.py` | 基本分支及访存指令选择 | `_select_br_if` 仅读第一个操作数；块定义加 `.`、分支目标未在此处统一；局部槽位分配也需要独立验证 |
| `scratchv/backend/llvm_codegen.py` | 文本 LLVM IR 输出 | `_emit_br_if` 按单个 `i1` 输出，不消费双操作数比较；不能以文本生成成功替代 LLVM 验证 |
| `scratchv/simulator/rv32_emulator.py` | DSLInterpreter 的线性运算 | `_op_br`、`_op_br_if`、for/endfor 为空操作，不能作为控制流正确性的执行依据 |
| `tests/test_dsl_extended.py` | 24 个条件解析及结构测试 | 主要检查块数、标签、解析成功，尚不足以证明六种比较和循环更新正确 |

例如，以下源码在当前基线生成的循环头仍比较初始 `i` 对应的值；循环体的新值没有传回循环头，零次执行路径上的 return 又引用了循环体产生的值。

```text
i = add(0, 0)
while (i < 3):
    i = add(i, 1)
endwhile
return i
```

这是本地打印 IR 可重现的问题，不是本次文档 PR 已修复的功能。旧课题说明中的 `const(...)` 也不在当前算子集合内；新示例使用已有的 `add(0, 0)` 等语法初始化。

## 3. 方案选择

| 方案 | 优点 | 代价 | 结论 |
| --- | --- | --- | --- |
| 继续直接逐行生成 IR | 改动少，容易保持接口 | 校验、块解析和变量状态耦合，错误恢复及合流处理难维护 | 可用于补回归测试，不作为长期结构 |
| 轻量语句 AST + 局部变量槽位 | 先完成结构校验，再独立生成 CFG；槽位能表达循环更新 | 需要验证 LOAD/STORE/ALLOCA 在后端的真实语义 | **推荐**，按测试门槛逐步落地 |
| 完整 SSA + phi + 通用表达式前端 | 便于后续数据流优化 | 当前 OpCode 没有 PHI，需同时补后端、优化器和 SSA 消解 | 超出首次交付范围 |

推荐方案不会假设现有访存实现已经足够。槽位、类型、比较和标签的后端契约测试是执行验收的前置依赖；未通过时不能宣布课题完成。先保证正确性，再考虑 mem2reg 或 SSA，首次交付不承诺超过 LLVM 的性能。

## 4. 语法契约

下面是拟固定的规范语法；现有省略冒号的 if/while 和 `else` 兼容形式继续接受并测试。缩进用于可读性，块边界由结束关键字决定。关键字后至少一个空白，暂不增加 `if(a < b)` 形式。

```ebnf
program    = { statement } ;
statement  = assignment | return_stmt | if_stmt | while_stmt | for_stmt ;
if_stmt    = "if", condition, [":"], newline, block,
             ["else", [":"], newline, block], "endif" ;
while_stmt = "while", condition, [":"], newline, block, "endwhile" ;
condition  = "(", operand, comparator, operand, ")" ;
comparator = "==" | "!=" | "<" | ">" | "<=" | ">=" ;
operand    = identifier | number ;
block      = { statement } ;
for_stmt   = "for", identifier, "=", unsigned_integer, ",",
             unsigned_integer, newline, block, "endfor" ;
```

assignment、return、identifier 和 number 沿用基础 DSL 的约束；条件数值首先采用现有十进制数字规则，不额外承诺科学计数法。条件必须完整匹配；`if (a + b > c)`、缺操作数和比较链必须得到诊断。保留空块、空文件、空行、注释、CRLF、Unicode 标识符及 tab 的行为。显式 return 仍要求一个值，不能误把语法扩展为无参 return。

统一处理原始源码和注释，保留物理行列，AST 节点携带起止位置。校验器和 AST 解析器共享条件及块识别逻辑；不能一处放行另一处报错。错误输入不进入 IR 构建；多个错误按现有 ErrorCollector 的排序和上限输出。

## 5. 模块划分与数据流

```text
SourceBuffer（原始源码、物理行列）
    -> 共享语法识别 / 结构校验
    -> 轻量语句 AST（携带位置）
    -> 变量读写及路径分析
    -> IR lowering（基本块、槽位、分支）
    -> IR 验证 -> 后端生成 -> 真实执行核对
```

计划新增内部模块 `dsl_ast.py` 和 `dsl_lowering.py`。前者定义 Block、AssignNode、ReturnNode、IfNode、WhileNode、ForNode 及条件节点；后者只接收已校验 AST，维护当前块、终结状态、变量绑定及函数内标签计数。两者不是新的用户 API。

`dsl_extended.py` 保留入口并负责协调，`dsl_validator.py` 与其共享结构解析结果或识别函数，`dsl_errors.py` 继续负责诊断表示。基础算子仍复用现有签名表和 builder，不复制另一份算子注册表。解析器重复使用以及先失败后成功的解析都必须重置所有状态。

### 5.1 变量与路径

直线表达式继续生成原有值。对跨分支/循环写入的标量建立函数入口槽位：初始化写入槽位，读取生成 LOAD，赋值生成 STORE；while 头每次执行都重新 LOAD。不同变量必须使用不同槽位，循环中不得反复分配槽位。

保留基础 DSL 的隐式输入习惯，例如 `y = add(x, 1)` 中 x 是输入。为避免改变现有 `while (i < 10)` 和 `acc = add(acc, x)` 的含义，块内首次读取且尚无绑定的输入须在结构分析时登记为入口输入，先初始化槽位，再进入控制流。

分支分析必须分别从同一入口环境开始，else 不能继承 then 新建的绑定。两条可达分支都赋值时，汇合后读取同一槽位；只有一条赋值且入口已有值时，另一条保留入口值。若变量只在一条分支中首次定义，之后使用时不能拿该分支的值冒充全路径定义，也不能临时补成隐式输入：报告未初始化诊断。while 体内首次定义、循环后读取且可能零次执行的变量同样处理。两条分支均 return 时不存在可达汇合路径。

这会拒绝此前“能生成 IR 但可能读取未定义值”的程序，属于明确的语义修正。实施时先列出受影响的旧测试和用例，补初始化或断言诊断；不得删除它们来获得绿灯。

### 5.2 IR 与后端契约

保留现有单条件 `BR_IF operands=[cond]`，增加公开 builder 方法（拟名 `br_compare`）生成已存在的双操作数形式：`operands=[lhs, rhs]`、`attrs['cmp_op']`、`target='true,false'`。两种形式按操作数数量及属性显式区分；不支持的组合必须失败。

LLVM 后端对比较形式生成有类型的比较指令，再以结果 `i1` 分支。INT32 采用有符号比较；FLOAT32 使用相应浮点比较，`!=` 在 NaN 时为真，其余有序关系在 NaN 时为假；不将浮点位模式作整数比较。旧单条件形式保留其语义和回归测试。

RISC-V 后端必须消费左右操作数及比较符，并统一标签定义和跳转引用。基线选择器把数值常量转成整数，所以不能据此承诺完整 FLOAT32 支持。第一组真实执行验收采用整数值标量、add/sub/mul、无溢出且可精确表示的输入；非整数条件另外进入后端能力测试。未实现的类型组合要显式报告不支持，不能截断数值后声称通过。完整 FLOAT32 执行属于后续后端协作门槛，不混同于语法支持。

槽位接口须明确大小单位、元素类型、对齐、地址和生命周期。目前 LLVM ALLOCA 将 size 用作元素个数，RISC-V 用作字节数；RISC-V 分配也未形成完整独立栈槽布局。因此，先用两个以上槽位的读写测试固定契约，再接入 lowering。建议新增明确的标量槽位封装，由后端换算大小，保持原 ALLOCA 调用兼容；禁止仅调用旧 `alloca(4)` 并假定跨后端等价。

### 5.3 块、终结与嵌套

if 的 then/else 从相同入口环境生成，只给未终结的可达出口追加 BR。while 由 preheader、header、body、exit 组成，回边必须回 header。遇到 RETURN 后不再往同一块追加 BR 或普通指令；之后的源码仍做语法校验，但不会成为可达指令。

标签在一次函数生成期间唯一，多次 parse 结果确定。现有 for 的 `start <= i < end`、步长 1 规则保持；混合嵌套使用 AST 的块层次匹配，禁止一类结束关键字关闭另一类块。既有 FOR/ENDFOR 指令跨基本块的后端行为必须先验证，不能仅凭父类支持 for 就宣称混合嵌套执行正确。

## 6. 正确性与性能验收

| 层次 | 必须验证的内容 | 通过标准 |
| --- | --- | --- |
| 语法/诊断 | 六种比较、可选 else、空块、嵌套、非法操作数、错配结束符 | 正确 AST 或准确诊断，错误不产生 IR |
| IR/CFG | 目标存在、唯一标签、单个块末尾终结指令、变量初始化和回边 | IR verifier 加定向断言，无未定义值路径 |
| LLVM | 分支类型及所有生成块 | llvmlite 解析并 verify；标量实例执行结果正确 |
| RISC-V | 比较、标签、槽位、输出、循环终止 | 编码后真实 TinyFive 执行，禁止 Stub 作为正确性证据 |
| 编译集成 | 优化关闭/默认配置、现有 regalloc 路径、CLI | 支持组合结果相同；未支持组合显式失败 |
| 回归 | 基础 DSL、课题 09、ONNX、CFG、后端 | 现有 CI 测试通过，无新增隐式跳过 |

至少交付三个自包含示例：if_else 返回 7；while_sum 对 1..5 累加返回 15；nested_loop 执行 3×2 次累加返回 6。完整源码和反例矩阵见开发文档。所有执行测试有步数及进程超时，超时为失败，不以“运行了一部分”计成功。

benchmark 衡量解析/IR 构建主机耗时、块数、指令数和端到端正确性，记录输入哈希、提交、Python/平台、重复次数及原始采样。旧正确直线用例做同输入前后对比；控制流修正用例比较预期输出，不要求与旧错误 IR 哈希一致。首期只报告性能变化，正确性不通过则报告失败；性能阈值待稳定基线后单独评审，不借用课题 09 的 1.5 倍阈值。

## 7. CI 与交付边界

实现阶段沿用 `.github/workflows/ci.yml` 的 test 和 benchmark jobs，不新增独立 workflow。新测试放在 tests 下自动发现；控制流真实模拟测试单独命名，避免被现有 `--ignore=tests/test_simulator.py` 排除。必要依赖缺失在 CI 中应失败。

报告复用 benchmark-reports 与 Actions Summary，名称使用 topic01/frontend，汇总和失败原因常显，详细源码/IR/日志默认折叠。保留现有课题 09、汇编美化器及其他 benchmark。deploy-pages 仍遵循主分支部署条件。

本次文档 PR 不更改解析器、测试或 CI。实现完成的定义是上述支持范围内的语义、后端及回归验收全部通过；文档合并和现有结构测试绿灯均不等价于增强器开发完成。
