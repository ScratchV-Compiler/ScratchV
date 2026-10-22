# W4：完整 Qwen3 的代码生成与 QEMU 单次前向

W4 使用固定 Qwen3-0.6B ONNX，沿用共享 ScratchV IR，通过 Tensor-C、Zig/LLVM 生成
RV64GC/LP64D ELF，在 QEMU virt 裸机中计算完整 logits。范围为 FP32、batch=1、L=256、
28 层、无 KV cache；阶段要求见[开发计划](../开发计划.md)。

W2 已由 [#106](https://github.com/ScratchV-Compiler/ScratchV/pull/106) 合入主干；
W3 [#96](https://github.com/ScratchV-Compiler/ScratchV/pull/96) 已于 2026-10-15 合入，
本分支同步主干 `8f726507c539607331f6b3599f2be811205a3e86`，保留 W4 扩展及共享修复。
代码依赖已合入；独立复现、Nightly 和团队验收仍按[验收与 CI](validation.md)分别确认。

本目录保存接口、复现和验收约定。原始运行产物保存在被 Git 忽略的 `output/` 中，
公开验收记录应引用固定源码、模型、工具和输入身份，并保留原始报告摘要；更新文档
不会改变已有运行的执行身份。

| 阅读顺序 | 内容 |
|---|---|
| [运行接口](runtime-contract.md) | 权重格式、内存布局、外置 ABI、加载和清理规则 |
| [复现步骤](reproduction.md) | 固定依赖与模型、完整运行、诊断和原始数组复核 |
| [验收与 CI](validation.md) | 分层门槛、测试入口、默认关闭的 Nightly、证据边界 |
| [阶段衔接与后续准备](evolution.md) | W1–W4 接口对应、W5 前置工作、未来 MLIR 转换边界 |
| [软仿优化实测报告](softsim-optimization-2026-10-22.md) | 多 hart、微内核、布局与运行器优化；完整模型对照和适用边界 |

## 验收标准

| 门槛 | 判据 |
|---|---|
| `build:riscv-full` | 完整模型的无大权重 ELF，文件大小严格小于 100,000,000 bytes |
| `unit:weight-loading` | 固定资产身份、名称/dtype/shape/偏移、容量、损坏与生命周期测试 |
| `smoke:qemu-full` | guest 真正完成前向，完成帧及完整输出合法，退出码 0，清理成功 |
| `numeric:qemu-full-qwen3` | 全部 `[1,256,151936]` FP32 输出有限，严格 `max_abs < 1e-3`，`rtol=0` |
| 团队出口 | E1/E3/E5 独立复现、接口/布局确认、实际 Nightly 成功记录 |

默认完整探测覆盖一个 `full_seed_0` 输入。七组输入入口不代表七组已经执行；前缀检查点、
Host C 诊断、离线 audit、构建成功、top-1 一致均不能替代完整 QEMU 数值门槛。
W5 的 token 循环、文本生成、多轮和部署脚本，以及 W6 的性能优化另行验收。

现有小模型默认 inline 权重和 sequential MatMul 保持兼容。W4 入口显式选择 external、
独立 kernel helper 和 blocked_fma，不改变既有 W1–W3 数值阈值。
