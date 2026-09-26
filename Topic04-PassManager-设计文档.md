# Topic 04：PassManager 注册、管线与开关

## 需求与代码基线

本实现承接 [PR #48](https://github.com/ScratchV-Compiler/ScratchV/pull/48)，
并合入上游主分支 `11a2c3c`（2026-09-26），新 PR 以主分支为合并目标。
保留 #48 的统一优化接口、逐 Pass 计数、耗时和失败即停语义。
本次设计和历史课题材料按导师要求归档在 ScratchV 仓库根目录。

检查范围覆盖仓库目录、#48 之后的提交历史，以及前端、IR/CFG、优化器、
编译驱动、后端、CLI、示例、benchmark、测试和 CI 的调用关系。
上游已合并的 DSL 诊断与控制流、统一 CFG、寄存器分配、汇编窥孔、
常量加载合并及 benchmark 功能均以当前主分支为准。

| 导师建议 | 本次实现 |
| --- | --- |
| 设计文档归档到根目录 | 本文、两份 #48 历史文档及 README/文档索引入口 |
| 把已有常量折叠纳入 PM | 复用 `ConstantFolder`；保留 basic/all 默认调度，增加独立选择与禁用 |
| 简单 if 控制 Pass | 调度前用 `if`，或 `register(pass_, enabled=flag)` |
| 通过注册管理 Pass | `PassRegistry` 保存名称与工厂，`build` 按显式顺序构建管线 |
| 管理优化与功能性 Pass | 同一执行器支持 IR 优化、数据转换、只读统计；汇编阶段实际接入 |

## 参考 LLVM 的范围

[LLVM New Pass Manager](https://llvm.org/docs/NewPassManager.html) 将默认管线构建、
Pass 添加及执行区分开来，并要求 Pass 与所处理的 IR 层级匹配。
本项目据此采用注册与调度分离、显式顺序和分阶段管线。
这是适合当前 Python 编译器规模的设计取舍，并非对 LLVM 接口的逐项复制。

当前不引入 LLVM 的多层 AnalysisManager、分析缓存失效协议、动态插件装载，
也不隐式重复运行到不动点。自定义分析 Pass 每次重新运行；使用者负责安排依赖顺序。

## 模块与接口

- `scratchv/pass_interface.py`：保留 #48 的 `OptimizationPass.optimize(Program) -> int`；
  功能性 Pass 使用现有 `CompilerPass.run(data) -> PassResult`。
- `scratchv/pass_manager.py`：统一执行、名称注册及默认 IR 管线构建。
- `scratchv/assembly_passes.py`：适配已有汇编实现，不重写其算法。
- `scratchv/compiler.py`：组织编译阶段，兼容导出旧 `PassManager` 与工厂导入路径。

`PassManager.run(program)` 原地修改 IR，返回不可变 `OptimizationReport`，
保持 #48 调用契约。混入功能性 IR Pass 时必须返回同一个 Program。

`PassManager.run_pipeline(data)` 逐步传递每个 `PassResult.data`，返回
`PipelineResult(data, report, warnings)`。数据转换使用此接口；只读分析返回原数据、
零变更数及诊断。不能将汇编文本传给 IR 优化 Pass。

两个入口共享顺序执行、计时、非负整数变更数校验与异常处理。
失败时停止后续 Pass，异常带有失败名称、序号和已完成报告；不回滚已做的原地修改。
编译驱动在优化或汇编 Pass 失败时返回错误，不创建或覆盖目标文件。
统计分别位于 `CompileResult.stats["optimization"]` 和 `["assembly"]`。
调度和美化器没有原生改动数量，适配层以文本是否改变计为 0/1；不混称为节省指令数。

## 开关方式一：简单条件

```python
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.pass_manager import PassManager

manager = PassManager("optimizer")
if enable_constant_folding:
    manager.register(ConstantFolder())
report = manager.run(program)

# 等效简写：关闭时不加入执行管线。
manager = PassManager().register(ConstantFolder(), enabled=enable_constant_folding)
```

简写接收已经构造的对象，因此若构造本身昂贵，使用上面的 if 或工厂注册方式。

## 开关方式二：名称注册与管线选择

```python
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.optimizer.dead_code import DeadCodeEliminator
from scratchv.pass_manager import PassRegistry

registry = PassRegistry()
registry.register("constant-folding", ConstantFolder)
registry.register("dead-code-elim", DeadCodeEliminator)
manager = registry.build(
    ["constant-folding", "dead-code-elim"],
    disabled=["constant-folding"],
    pipeline_name="optimizer",
)
report = manager.run(program)
```

注册仅声明可用工厂，不代表默认启用。构建时先校验所有名称，再过滤禁用项，
最后构造对象。禁用项不会调用工厂；不同管线和重复的名称位置各有独立实例。
重复注册同一个名称、未知选择、未知禁用名称、错误工厂结果均报错。
工厂返回的 Pass 名称必须与注册名称一致。`registry.names` 可查询可用名称。

## 默认管线和 CLI

| 级别 | 顺序 |
| --- | --- |
| none | 空 |
| basic | constant-folding → dead-code-elim |
| all | basic → ir-peephole → muladd-fusion → licm |

```bash
python -m scratchv model.onnx --opt-level basic --disable-pass constant-folding
python -m scratchv model.onnx --passes constant-folding,dead-code-elim
python -m scratchv model.onnx --passes constant-folding,dead-code-elim,constant-folding
python -m scratchv model.onnx --passes ""
```

优先级：显式 `--passes` 替代优化级别预设，然后应用 `--disable-pass`。
未传 `--passes` 与显式空列表不同。重复名称按次执行；禁用移除该名称全部出现位置。
`--disable-pass` 可重复；名称拼写错误会报错，不静默忽略。`--optimize` 仍是优化级别别名。
Python 配置对应 `CompilerConfig.passes`、`disabled_passes`；默认值保持原行为。

汇编注册表独立于 IR 注册表，保持现有顺序和默认关闭策略：
`asm-peephole → const-merge → schedule → beautify → count-instr`。
这些项继续由现有 CLI 开关控制，`--passes`/`--disable-pass` 仅作用于 IR。
`constant-folding` 是 IR 算术常量折叠，`const-merge` 是汇编常量加载合并，二者不同。
前端解析、代码生成、运行时验证和 cycle estimation 仍由编译驱动组织。

## 常量折叠与兼容性

常量折叠复用上游实现，只承接 #48 的调用接口与单次计数改造。
本次不加入 #50 的数值语义扩展，不重写 ADD/SUB/MUL/DIV 算法。
仓库内优化调用点统一迁移，包含最新主分支新增的 Topic 17 CNN benchmark。
外部调用者需将 `ConstantFolder(program).run()` 迁移为
`ConstantFolder().optimize(program)`，或调用统一管线工厂。
`OptimizationReport`/`OptimizationPassError` 名称为兼容 #48 保留，功能性管线也复用。

## 验证与历史归档

行为测试位于 `tests/test_pass_manager.py`、`tests/test_pass_registry.py`：
真实 IR 折叠开关、默认预设、显式空管线、顺序和重复、工厂隔离与懒构造、
未知名称、功能性数据传递、只读分析、错误停止及输出文件保护。
集成验证覆盖既有优化器、编译驱动、汇编适配和 benchmark 调用。
完整命令、结果和平台限制见 [验证记录](Topic04-PassManager-验证记录.md)。

- [#48 课题说明历史归档](Topic04-IR优化器框架-历史归档.md)
- [#48 优化器框架历史归档](Topic04-Optimizer-Framework-历史归档.md)

历史材料原有的阶段目标和示例用于追溯，不作为当前功能完成度的断言。
