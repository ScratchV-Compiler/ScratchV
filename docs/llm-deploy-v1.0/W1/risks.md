# 风险清单 v1.1 候选

> 上游：[开发计划.md](../开发计划.md) §5。
> 状态（2026-10-02）：**已按本地实测回填；责任人及处置路线待团队确认**。
> 对应 `docs:risks` 的结构检查：20 条风险，每条有触发信号和 Plan B。自动检查不替代人工评审。

## 0. 阅读约定

“旧路径已确认”指原标量选择器，不泛指新增 tensor-c 后端；“小模型通过”只覆盖两层缩小配置的固定随机权重；“本地通过”不代表新 Linux CI 已通过。历史失败仍保留，不因替代路线成功而删除。完整 28 层、预训练权重、内存与性能风险不得凭小模型结果排除。

## 1. 高影响风险

| # | 风险 | 概率 | 影响 | 当前状态 / 证据 | 触发信号 | Plan B | 责任 |
|---|---|---|---|---|---|---|---|
| R1 | 旧选择器 MATMUL 为标量整数乘占位 | 已确认 | 高 | `instruction_select.py::_select_matmul` 的旧路径未修复；新 tensor-c MatMul 7 用例通过 | 选错旧后端，或新后端 MatMul 与 ORT 不符 | 本次采用有显式矩阵循环的 tensor-c；旧路径保持范围限制。若回归失败，停止扩模，先定位 MatMul shape/累加/广播 | E3，当前路线确认 |
| R2 | 旧选择器缺张量 FP32 lowering | 已确认 | 高 | 不把助记符存在视为张量实现；新 RV64GC + musl 路线本地已通过 | 缺浮点目标、非有限值或绝对误差达阈值 | 保留 FP32目标、禁用 fast-math/FMA并逐层定位；是否另补旧选择器由团队另立项，不静默换 FP64/定点通过原 gate | E3 |
| R3 | 多后端 ISA / ABI 混用 | 已确认 | 高 | 新路线 RV64GC/LP64D/裸机；旧 RV32 CNN 配方另存，不能互用 | 用 qemu-riscv32、错误 ABI 或把 Linux triple 当成 Linux guest | 按 artifact/runtime 固定工具与 ABI；编译/加载失败即终止。团队确认路线；其他后端独立验收 | E2/E3/E4 |
| R4 | 不同执行器的算子支持范围被混淆 | 已确认 | 高 | 前端/IR 的 Transpose、Concat 已能执行，新 tensor-c 支持所需图；旧 selector 范围未修复 | 前端通过而选定后端拒绝/错误执行 | 每个后端显式拒绝未支持操作，新增能力时补后端数值用例；不能把 parser 成功当 backend 通过 | E1/E3 |
| R5 | 新 CI 工具链或网络不可用、门禁被跳过 | 待验收 | 高 | Windows 便携执行已通过；本轮新 Linux CI 尚无实际 run 证据。保留早期 runner sudo/HTTPS 失败历史 | 缺 Zig/QEMU/LFS 分片；job pending/skip；报告无实际执行 | 缺必需工具/输入显式失败；可先本地跑并标 CI 未验收。重型 ONNX 按模式分别展示结构与 ORT 状态，E5补真实 run/artifact | E5 |
| R6 | IR 不能表达 Attention | 小模型已验证 | 高 | 官方 Qwen3 基础算子图经 IR 及 QEMU 通过；不需要强制融合 Attention opcode | 新配置解析/执行失败或 checkpoint 首次偏差 | 保留基础算子基线，缩小到失败子图并修相应语义；若使用 host 分段必须标明路线变化 | E2 |
| R7 | 完整 28 层数值误差累积 | 待验证 | 高 | 两层 29 检查点通过不等于完整层数已通过 | 随层数增加误差越过既定阈值 | 逐步扩大层数/尺寸并比较首次偏差；先定位数学/累加/优化问题。任何更改精度或容差需独立评审，不能改 gate 掩盖错误 | E2/E5，W3 |
| R8 | 因果/padding mask 错误 | 小模型已验证 | 高 | 7 输入含 future/padding 改动；IR/QEMU 不变性通过。完整导出 padding query 数值差异仍记录 | 被屏蔽位置概率泄漏或有效前缀受未来/padding token 影响 | 固定加性 mask 与位置约定，检查 attention probabilities/首个偏差；把有效位置和全张量结果分开报告，不删 padding 失败证据 | E2/E5 |

## 2. 其余风险

| # | 风险 | 概率 | 影响 | 当前状态 / 证据 | 触发信号 | Plan B | 责任 |
|---|---|---|---|---|---|---|---|
| R9 | 完整 Qwen3 ONNX 导出/载入失败 | 本地已验证 | 高 | 历史导出通过；新入口verify已通过真实hash/checker/ORT两case，验证器峰值RSS约6.40 GB；新CI待运行 | 缺/损坏分片、版本漂移、shape 不符或 ORT 失败 | 固定 revision/依赖与哈希，从已批准来源重新准备产物；必要时重导出。子图只能作定位工具，不代替完整导出 gate | E1/E5 |
| R10 | 完整权重/激活超出内存布局 | 待验证 | 高 | 当前 guest 512 MiB、workspace 上限256 MiB；完整 logits 约148.4 MiB，权重约2.4 GB，未完整装载 | arena/ELF 与输入区冲突、容量超限、OOM | 先做完整容量预算，再设计大权重装载/分块/释放；若选择 mmap 须另有 Linux 路线，不能在现裸机上只增加 -m 就声称解决 | E4，W4 |
| R11 | L=256 无 KV Cache 前向耗时过长 | 待测完整规模 | 中 | 小模型 QEMU 时间只用于功能回归，不代表硬件性能 | 完整前向超时或生成循环不可用 | 记录真实耗时，按算子分块/优化；短 L 仅作诊断且另标配置，不降低固定 L=256 验收 | E3/E5 |
| R12 | 151936×1024 LM Head 计算/输出开销 | 配置已确认 | 高 | 小词表128不覆盖完整词表开销 | 完整 LM Head 超时或输出传输过大 | 保留完整 logits oracle；生产生成路径可另设计最后位置/分块接口，并与完整输出对照，不能替换 W1 完整形状 gate | E3/E4 |
| R13 | Q/K RMSNorm、全 head_dim RoPE 语义误解 | 小模型已验证 | 高 | 官方 Qwen3 为全 head_dim RoPE；旧草案 partial 说法已纠正。PR89 partial 图不作该项证据 | Q/K normalize 或 rope_q/rope_k checkpoint 出现偏差 | 对照固定版本官方实现和逐层输出；保留 hidden≠Qwidth 的小模型配置，禁止套用 PR89 partial 逻辑 | E1/E2 |
| R14 | GQA 头映射错误 | 小模型已验证 | 高 | 4 Q/2 KV 通过独立 NumPy repeat/context 检查；完整16/8尚需规模对照 | attention probability/context 数值或形状异常 | 对每组 KV head 重复映射做数值对照，再扩大到完整配置；不以比例相同替代完整验收 | E2 |
| R15 | 工程任务长期无进展 | 持续管理 | 中 | 无法由测试自动判断 | 负责人连续无法给出可复现进展/阻塞 | E2安排结对定位和明确小任务，必要时调整排期并记录原因 | E2 |
| R16 | 文档接口与实现不一致或未经团队确认 | 待确认 | 高 | 已更新候选接口：基础图、INT64、指针数组、RV64裸机；团队签字未完成 | 对接签名/dtype/shape不同或未经记录修改接口 | 消费方契约测试 + interfaces决议表 + PR review；冻结前逐项确认，冻结后版本化修改 | E2，全员 |
| R17 | 合并冲突或证据对应旧源码 | 持续管理 | 中 | 本地报告保留源码/模型/输入/ELF哈希；提交后须关联CI | 报告哈希与待评审实现不同或大改后未复跑 | 小步合并；对当前提交重新执行受影响gate，保留旧报告并标历史，不覆盖后复用旧PASS | 全员 |
| R18 | W3 numeric:ir-full-qwen3 硬门槛失败 | 待验证 | 高 | 两层随机模型和完整 ONNX ORT 不能替代完整 IR 数值 | 完整 IR 不可执行或逐层数值不通过 | E1/E2/E5联合定位，按原计划缓冲调整排期；阻塞明确写报告，不误写完整模型部署完成 | E2/E5 |
| R19 | 权重来源/许可/外部分片/版本不可复现 | 部分已落实 | 中 | 历史导出固定revision/checkpoint SHA并保留LICENSE；新clone仍需准备真实分片 | 下载失败、LFS pointer、哈希不一致或缺许可记录 | 从固定revision重建或按仓库LFS/artifact约定获取，校验每个分片和manifest；不要把多GiB普通Git文件塞进PR | E1 |
| R20 | 数值基准或工具版本漂移 | 部分已落实 | 中 | 小模型依赖固定，报告含哈希/工具版本；历史完整ORT threads=4，不能冒称单线程 | 同输入跨环境结果漂移或导出图变化 | 使用requirements固定环境，记录ORT线程、provider、seed、revision；新环境重新产参考并比较，不复用不明来源数组 | E5 |

## 3. 证据与状态维护

| 对象 | 证据 / 对应风险 |
|---|---|
| 旧后端边界 | `scratchv/backend/instruction_select.py` 与 `llvm_codegen.py`；R1–R4 不因新路径成功自动关闭 |
| 原两层 IR | `probes/w1_tiny_transformer/run.py`；基础表达能力，不覆盖官方 Q/K RMSNorm/全 RoPE |
| 官方两层 IR/QEMU | `probes/w2_qwen3_small/run.py`、`riscv.py`；R6/R8/R13/R14 在小配置范围缓解 |
| 本地 QEMU 结果 | `output/qemu-matmul-final/report.json`、`output/qwen3-riscv-final/report.json`，28次两层执行通过；忽略产物需独立复现或从CI artifact取得 |
| 完整导出 | 固定revision `c1899de289a04d12100db370d81485cdf75e47ca`；历史导出通过，本轮`output/qwen3-full-local/report.json`为新verify的真实hash/checker/ORT结果；R9/R19部分缓解 |
| 新 CI | 关联实际commit/run/artifact后更新R5；脚本存在、job启动、结构检查都不是完整数值证据 |

新增证据时记录模型范围、执行路径、环境、误差与源码指纹；失败保留首次偏差和日志。不要把所有状态统一改成“已排除”。

## 4. 评审与触发责任

- [x] 20 条风险均列出触发信号和 Plan B。
- [x] 回填新路线，保留旧选择器和完整模型的未覆盖边界。
- [ ] E1–E5确认各自责任、处置范围与排期；E2记录决议。
- [ ] 新 Linux CI及完整 ONNX 重型门禁取得真实执行证据，更新R5/R9。
- [ ] E4/E5完成第二人独立复现，按 [模板](README.md)记录。
- [ ] 完整权重加载、容量及28层数值验收另有计划和责任人。

## 5. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 原草案 | v0.1 | 原计划与静态勘察20项 |
| 2026-10-02 | v1.1 候选 | 回填本地结果，区分旧/新后端与小/完整模型；明确所有触发条件、替代路线和人工确认项 |
