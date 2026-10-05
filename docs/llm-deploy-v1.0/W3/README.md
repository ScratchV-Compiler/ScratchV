# W3 完整 Qwen3 IR 验证与交付

本目录汇总完整 Qwen3 IR 数值验证、准备探测、诊断与复现交付。实现与本地硬门槛已有验证记录，团队正式验收仍待完成。本轮基于 W2 提交 `65616a8cde6a581661ac3734378da78a9fcbe5ed`，改动保留在独立的 `codex/w3-preparation` 工作树，尚未提交或推送。

初次预备阶段的数据见 [历史预备工作本地验收报告](W3-预备工作本地验收报告.md)。这些历史记录不会随代码修改自动更新；本次验收应使用当前代码实际生成的报告、源码指纹和数值模式。

本地交付入口：

- [W3 工作总结与验收进度](W3-工作总结与验收进度.md)：开发计划对照、改动分组、实测证据及待团队推进的事项。
- [独立复现与验收清单](独立复现与验收清单.md)：候选验收口径、源码快照、实际执行命令，以及 E1/E2/E5 各自填写的记录。
- [Linux CI 与 Nightly](Linux-CI与Nightly.md)：可复用的手动 CI、显式启用的定时编排及尚未实跑的边界。
- [保存证据的离线复核](../../../probes/w3_qwen3_full/EVIDENCE.md)：重新读取原始数组核对，不能替代另一位开发者执行模型。

未提交代码可通过 `scripts/package_w3_repro.py` 生成带文件哈希和 snapshot ID 的本地源码包；它不是新的 Git 提交，不包含模型、运行产物或环境。解压后先校验源码，再按清单准备已有资产和依赖。报告中的历史源码指纹保持原样。

完整 28 层的实际 ORT/IR 执行、内存受控模式和同输入局部诊断见 [完整模型入口](../../../probes/w3_qwen3_full/README.md)。完整入口默认采用显式版本化的 `reference` FP32 模式，解释器 API 仍默认 `native`；复现时必须记录并核对模式与 profile。统一 preparation 的五项 PASS 只代表准备范围通过，完整模型 gate 另行执行。

本轮完成可独立执行的数值与资源准备，不代表 W3 阶段出口通过。开发计划要求的完整 28 层 IR/ORT 数值 <1e-4、W1 人工验收前置、E1/E2/E5 三人复现及 Nightly 仍单独验收。

## 交付与范围

| 项目 | 本地入口 | 通过含义 |
|---|---|---|
| 完整 Qwen3 IR | [full](../../../probes/w3_qwen3_full/README.md) | 固定真实 28 层、reference-v2、七输入，普通/诊断完整 logits 含 padding，四项不变量；严格 <1e-4 |
| 六层 medium | [medium](../../../probes/w3_qwen3_medium/README.md) | 官方 Qwen3 类的六层随机小配置，7 输入、81 检查点、普通/诊断双图、none/basic/all；全部 <1e-5 |
| 真实维度子图 | [subgraphs](../../../probes/w3_qwen3_subgraphs/README.md) | 固定 checkpoint 的第一层预训练权重，14 个 L256 子图及 10 项不变量；Torch/ORT/IR <1e-4 |
| 组合 Attention 后端 | [attention](../../../probes/w3_attention/README.md) | Q/K RMSNorm、RoPE、GQA、因果/key-padding mask、Softmax 和 context 组合，6 图 × 2 优化等级，共 12 次实际 QEMU，<1e-4 |
| 逐层误差工具 | [layer-diff](../../../probes/w3_layer_diff/README.md) | 独立验证已有 NPZ/schema 的全部检查点或选定子集；部分比较始终 PARTIAL |
| 完整资产/尺寸预检 | [preflight](../../../probes/w3_full_preflight/README.md) | 固定完整产物哈希、ONNX 契约、静态逻辑字节数通过；不执行完整 ORT/IR |

medium 候选配置为 6 层、hidden64、FFN192、Q4/KV2、head_dim16、vocab128、L256，304128 个随机参数。开发计划尚未规定 medium 的通道尺寸，本 preset 是可复现的候选，需团队确认，不能当作完整尺寸或预训练六层模型。

真实子图采用 hidden1024、FFN3072、16Q/8KV、head_dim128；Q 投影宽度是 2048。其激活为固定种子的合成输入，部分 Q/K/V 由真实投影、归一化和 RoPE 生成，未执行完整预训练 decoder 层。BF16 checkpoint 权重提升到 FP32 计算，来源与哈希记录在报告中。W3 的 1e-4 不改变已有 W2 的 1e-5 门槛。

## 环境与统一复现

使用 Python 3.12、[固定 CPU 依赖](../../../requirements/qwen3-small-probe.txt)、Zig 0.14.1 和 qemu-system-riscv64。环境安装方法见 [Qwen3 小模型说明](../../../probes/w2_qwen3_small/README.md)。固定 HF snapshot 及 ONNX 发布目录均需真实完整资产，不能是 LFS 指针。统一入口离线校验已有资产，不会安装依赖或下载模型。

在这个 W3 checkout 的根目录执行；输出目录必须不存在：

~~~powershell
$w3Python = 'D:/cyq/code/ScratchV/.venv-qwen-export/Scripts/python.exe'
& $w3Python -B -X utf8 scripts/run_w3_preparation.py `
  --source-dir D:/cyq/code/ScratchV/models/qwen3-source/c1899de289a04d12100db370d81485cdf75e47ca `
  --model-dir D:/cyq/code/ScratchV/models/qwen3-0.6b-onnx `
  --output-dir output/w3-team-preparation-new `
  --cc D:/path/to/zig.exe `
  --qemu D:/path/to/qemu-system-riscv64.exe
$LASTEXITCODE
~~~

五项都实际执行并通过、源码指纹一致、必需报告/产物齐全时，才返回 preparation:w3 的 PASS/0。它不是 numeric:ir-full-qwen3 的通过。失败保留子日志和 FAIL 报告，后续复跑必须换新目录。

统一报告位于 output/w3-team-preparation-new/report.json、report.md、report.html；子目录中保存原始张量、模型、ELF、命令和各项报告，logs 目录保存原始执行日志。medium 的 traces/<case>/ 下保存原始 PyTorch、ORT、IR 张量，trace_schema.json 定义检查点；layer-diff 会独立读取本轮 short_17 的全部 81 个检查点进行核对。CI 配置上传报告、schema、日志、单测 XML、用于 layer-diff 的 actual/reference 原始数组和失败 case 证据；固定完整模型权重及其余成功大数组不上传。

准备阶段报告产物保留 30 天，选定诊断产物保留 7 天。诊断收集器默认单文件上限 256 MiB、累计上限 512 MiB；`retained-evidence.json` 记录实际复制的大小、SHA256 和省略原因，收集不完整时返回失败。它仅保留选定轨迹与失败证据，不代替完整 W3 数值验收。

子入口具有各自完整报告和退出码；layer-diff 选层、子图缩短序列的成功只产生 PARTIAL/2，不能用作上述完整准备验收。

## 内存和耗时口径

IRInterpreter.run 新增可选 collect_memory_stats=True，返回结果增加 memory_stats；默认关闭。测量输入/initializer 逻辑字节、每条指令边界的保留张量逻辑峰值、去重 NumPy 底层存储峰值，以及返回值拷贝。它不改变输入复制、张量生命周期、控制流、数值或 D8 异常语义。

- peak_live_logical_bytes 包含别名视图按各自形状计算的逻辑大小，可能重复。
- peak_numpy_storage_bytes 对底层 buffer 去重，包含返回结果的独立拷贝。
- process_peak_rss_bytes 是所在进程的全生命周期峰值，包括 Torch/ORT/IR 等；不是某个阶段的增量峰值，也不是 IR 工作区。
- NumPy kernel 临时 buffer、Python 对象和分配器缓存不在解释器的保留值计数中；其进程影响可能反映在 RSS。
- QEMU 时间包含启动、guest 计算、UART 传输和退出；成功与失败尝试都计入，不能当作目标硬件纯推理性能。

上述是预备阶段的默认内存统计。后续完整入口已显式启用只读权重映射、initializer 借用和 last-use 回收，旧调用默认行为保持不变；该容量准备不等于 W6 性能门禁通过，详见完整模型报告。

## CI 与后续阶段

新增 [.github/workflows/w3-preparation.yml](../../../.github/workflows/w3-preparation.yml) 保留手动准备验收入口，并允许 [Nightly 编排](../../../.github/workflows/w3-nightly.yml) 复用。它安装固定依赖，下载并校验固定 checkpoint 和 ONNX 资产，运行回归与五项入口。Nightly 定时任务需要仓库变量显式启用；手动触发 Nightly 则明确请求运行两套流程。本轮没有发布/运行远端 CI，也没有设置远端开关；本地配置存在不算 CI 已通过。

完整模型 preflight 仅验证资产与静态大小，峰值是 metadata-only 进程，不能替代完整 IR 的容量实测。完整 logits 必须包括 padding 位置；PyTorch/ORT 的导出诊断与正式 IR/ORT 门槛分别记录，不能用局部子图或有效位置通过覆盖。

后续顺序：
1. 团队完成 W1 人工验收、确认 medium 配置与 W3 输入/比较契约。
2. 使用完整 ORT/IR 分进程入口，在约定的数值模式下重新生成七组全部 logits 的 <1e-4 验收证据；保留依赖、CPU/BLAS 诊断和内存测量。
3. 并行安排 E1/E2/E5 独立复现和远端手动 CI；出现差异时使用局部诊断。Nightly 的配置和真正运行通过须单独确认。
4. W3 完整数值硬门槛通过后，推进 W4 权重独立装载和完整 QEMU 前向；生成循环属于 W5。
