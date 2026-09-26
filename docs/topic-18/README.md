# Topic 18：指令调度

当前实现对寄存器分配后的汇编做局部换序，使用 LLVM MCA / SiFive E76 评估收益。仓库中的评测结果使用 LLVM 18.1.3；复现这些数值时使用该版本。日常运行和 CI 不锁定 LLVM 版本，报告记录实际工具版本。报告统计发射峰值、零发射周期和局部完成周期。

| 文件 | 用途 |
|---|---|
| [代码说明](18-指令调度器代码说明.md) | CPU 模型来源、调用配置、调度及验证方法 |
| [验证报告](18-指令调度器Review迭代报告.md) | 仓库 CNN 的 A/B 结果、寄存器副作用和覆盖限制 |
| [原始数据](cnn-scheduling-results.json) | 输入哈希、汇编、逐区域原始 LLVM JSON 和执行证据 |
| [TODO](todolist.md) | 完成事项与后续问题 |
| [设计文档](18-指令调度器设计文档.md) | 原文保留；历史设计，不等同于当前实现 |
| [SPEC Review](18-指令调度器SPEC-Review.md) | 原文保留；历史评审 |

主测试对象为 `models/graph/cnn.onnx` 的 standalone 编译结果，CompilerDriver 两种寄存器分配输出作为补充。检查沿用现有 CI jobs。

本次实现保留主分支的公共指令选择、寄存器分配和 SelectionDAG 行为。关闭调度的版本对照由 `benchmarks.compare_schedule_disabled` 执行；它与同一版本内的调度 A/B 是两项独立验证。standalone 默认汇编格式不变，评测显式启用 `symbolic_asm=True`。

```bash
PYTHONHASHSEED=0 python -m benchmarks.bench_cnn_schedule --llvm-mca llvm-mca-18
```

生成报告：`benchmark_reports/inst_scheduler_cnn.md`；数据：同名 JSON；汇编和二进制：`benchmark_reports/cnn_schedule/`。

上述命令用于复现 LLVM 18.1.3 的结果，运行前需确认 `llvm-mca-18 --version`。日常评测可省略 `--llvm-mca`，默认使用 `llvm-mca`，也可通过 `LLVM_MCA` 指定其他路径。
