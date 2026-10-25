# QEMU 软仿优化报告（2026-10-22）

本报告汇总 PR104 已有的软仿优化实测与取舍。固定完整模型用例为 Qwen3-0.6B、FP32、batch=1、L=256、28 层、无 KV cache。测试在 4 个逻辑 CPU 的 WSL2/Linux 环境下进行；目标为 RV64GC/LP64D、QEMU virt 裸机。它不代表真实 RISC-V 或 NPU 性能。

同配置完整模型单/四 hart 各执行一次，前向从 **1384.65 s 降到 410.53 s，3.373×**；CPU 总时间增加 **5.46%**，38,895,616 个输出逐位一致。小图的微内核、布局及等待优化收益分别统计，不能与该倍数相乘。

## 默认值与可选功能

默认值保持 `1 hart / sequential / 1x4 / original / padding none`。手动选择 `--harts 1/2/4` 或 `--harts auto`；auto 只在启动/构建时按进程可用 CPU 选择 1/2/4，上限 4，不是运行中的动态调度，也不处理 cgroup CPU 带宽配额。

| 方向 | 已完成内容 | 适用边界 |
|---|---|---|
| 并行与等待 | 门铃通知和 WFI；MatMul 及独立 Softmax 行并行 | 保留每行算术顺序、错误传播及工作区复用；真实平台需适配门铃 |
| Softmax | 同形状回调代码复用；有限性检查融合到逐行写回 | 每次调用上下文和状态独立；不移除有限性检查 |
| 微内核 | 显式 `--matmul-tile 1x4/2x4/1x8` | 默认 1x4；非默认 tile 与 blocked_fma 配套 |
| 数据布局 | 显式 `--matmul-rhs-layout column-panel` | 仅符合条件的常量 RHS/独占常量转置链；共享和动态 RHS 保留 original |
| 运行器 | UART 完成事件、QMP nodelay、poll 高 FD 等待 | 保留 stop 后帧复核、转储、退出与异常清理 |
| 验证链路 | 收敛重复单测，20 次清理循环仅 schedule/manual，普通 PR 保留一次 | 保留真实 RV64/musl、Qwen 小图、W4 必跑 JUnit 与最终验收 |
| 未默认启用 | 两缓冲 `two-buffer-pad-4k` 实验选项 | QEMU 专用、增加 4096 B，部分形状退化；下方主要性能对照均 padding none |
| 未采用 | 本轮 RVV 候选 | 正确性通过但 QEMU 实测负收益，不纳入正式实现 |

## 完整模型：同配置 1/4 hart

两边固定 `full_seed_0 / blocked_fma / 2x4 / column-panel / padding none`，按 4 hart、1 hart 的顺序各运行一次。编译器 Zig 0.14.1，QEMU 8.2.2（Ubuntu 包 1:8.2.2+ds-0ubuntu1.18），数学库为 Zig 自带 musl。

| 指标 | 1 hart | 4 hart |
|---|---:|---:|
| 前向耗时 | 1384.65 s | 410.53 s |
| QEMU 生命周期耗时 | 1385.97 s | 410.94 s |
| QEMU CPU 总时间 | 1385.80 s | 1461.40 s |
| 平均 CPU 占用（100% 为一核） | 99.99% | 355.62% |
| 观测峰值 QEMU RSS | 5.3957 GiB | 5.3903 GiB |
| 对 ORT 最大绝对误差 | 8.58306884765625e-05 | 8.58306884765625e-05 |
| 对照输出逐位一致 | 是 | 是 |

并行加速为 **3.372844654×**，前向耗时减少 **70.35%**；CPU 成本为单 hart 的 **1.05455 倍**。全部 logits 满足严格 `max_abs < 0.001`、`rtol=0`，12 项 FMA 检查及保留数组审计通过。

完整数值、测量范围、采样缺失和来源索引见 [1/4 hart 对照数据](reports/full-hart-comparison-2026-10-22.json)。

测量口径：

- 前向计时从 QEMU 启动到宿主确认 UART 完成帧，包含启动及通知开销，不是纯 guest 算子时间。
- CPU 由 getrusage 记录整个 QEMU 生命周期的 user+system，包含启动、转储和退出，不包含构建、FMA 探测或 ORT。平均 CPU 占用以 QEMU 生命周期耗时为分母。
- RSS 是进程观测峰值，不是 guest 预留 RAM；两侧分别有 3 次、1 次读取失败，均标为 partial，内存结论只称基本持平。
- 没有重复运行统计或置信区间；宿主负载、温度与文件缓存状态并未严格一致。没有使用早期跳时异常记录计算本次加速比。
- 执行时 Git 基线为 `af5768cbe49976896f88f10f8f0cc6ec08160f30`，工作树含未提交候选改动；不能仅凭该 commit 重建完整候选。两侧生产源码摘要比较通过，已有快照身份各自保留在对照 JSON 中。运行早于最终高 FD、Host C 清理等修复，不能视作当前 PR HEAD 的完整重测。
- 完整模型本次没有可靠的 QMP hart/线程映射，不能将其线程采样精确标为主 hart 与各 worker。

## 小图：微内核与布局

下表采用 2026-10-22 UART 事件计时后的同批对照中位数；每配置 3 次新进程运行，慢轮未剔除，padding 均关闭。微内核共 54 次、正式布局共 24 次性能运行，输出均逐位一致。

| 对照 | 1 hart：前向耗时 | 4 hart：前向耗时 |
|---|---:|---:|
| Dense：1x4 → 2x4 | 5.8311 → 3.6155 s（1.613×） | 1.7681 → 1.1188 s（1.580×） |
| Dense：1x4 → 1x8 | 5.8311 → 3.6519 s（1.597×） | 1.7681 → 1.1211 s（1.577×） |
| 直接常量 RHS：original → column-panel，固定 2x4 | 3.6063 → 1.5745 s（2.290×） | 1.1042 → 0.5320 s（2.076×） |
| 常量转置链：original → column-panel，固定 2x4 | 3.7616 → 1.5677 s（2.399×） | 1.2308 → 0.4813 s（2.557×） |

正式布局的 direct_dense 为 LHS `[256,1024]`、RHS `[1024,1024]`；batch_transpose_wide 为 LHS `[2,64,1024]`、存储 RHS `[2048,1024]`，转置后参与乘法。后者工作区由 9,437,184 → 1,048,576 bytes，减少 8 MiB。phase3 报告未直接记录 Dense 形状，本表仅按原报告用例名引用，不用旧阶段形状替代它。

Dense 的 1x4 → 2x4 对照中，CPU 中位数分别从 5.8332 → 3.6183 s、6.8124 → 4.2183 s。Attention 的 4 hart 2x4 为 0.999×，说明更大微内核不保证所有形状受益。

布局测试保留原布局对照与容量/重叠/溢出/错误恢复/不同输入及工作区复用检查。完整模型实际采用 196 个 panel，逻辑与物理权重均为 1,761,607,680 bytes；共享 embedding/lm_head 与动态 RHS 保留原布局。ELF 必须与对应物理布局的权重 bundle 配套。布局收益没有新的 SoftMMU 采样支撑，不能精确拆成硬件缓存、TLB 或地址生成各自贡献。

## 等待、Softmax 与计时链路

这些是不同阶段的诊断或实现对照，不能组合成整体收益。早期 WFI/回调复用及 padding 对照仍使用约 50 ms 完成轮询，须与后续事件计时区分。

- **空闲等待：** 在 4 hart、64 次串行 Softmax 的历史对照中，WFI 使 CPU 9.780 → 2.125 s（每配置仅 2 次）。它主要减少空转；后续 Softmax 已按独立行并行，此数字不是当前完整模型收益。
- **回调复用：** 64 个同形状回调变为 1 个；末轴回调机器码 17,920 → 280 B，非末轴 19,456 → 304 B，工作区仍为 2 MiB。单 hart text 未变，不宣称单核代码体积收益。
- **有限性检查融合：** 后续事件复测的 256×1024、4 hart 中位前向 0.676418 → 0.649905 s（−3.92%），CPU 2.281089 → 2.300152 s（+0.84%）。独立线程采样显示 worker 非运行时间减少，但该区间同时包含调度等影响，不等于纯 WFI/屏障等待。
- **UART 事件计时：** 观测后唤醒中位延迟 25.367 → 0.098 ms，减少轮询量化；这是宿主完成检测改进，不代表 guest 计算加速。
- **QMP 尾延迟：** nodelay 对照中 stop 中位耗时由 42.617 → 0.621 ms（1 hart）、42.227 → 0.760 ms（4 hart）。STOP 事件已先到达，服务端 NODELAY 与独立客户端 QUICKACK 对照强烈支持 TCP Nagle/ACK 交互造成回复尾延迟；没有抓包，不声称逐包证明，也不把这约 42 ms 计为模型计算收益。
- 正式运行器已采用 UART 事件与 QMP nodelay；高 FD 边界及 Host C 超时/取消进程清理随后补齐相关回归，没有借此重跑完整模型。

## 退化用例与剩余问题

- **Padding 有明确反例：** 256×1024 的实验中位耗时下降，但 255×1024 的 1/4 hart 耗时分别增加约 22.22%/23.14%；四 hart 收益组也保留慢运行。因此仅作默认关闭的 QEMU 两缓冲选项，不作为泛用硬件/NPU 缓存策略。
- **RVV 暂不采用：** 43 次边界正确性与 54 次性能运行逐位一致；本次 VLEN128/e32m1 一行四列候选，耗时为预选强标量方案的 2.31–2.92 倍，CPU 为 2.61–2.98 倍。标量与 RVV 的 tile/布局并非完全一致，这是实用候选比较，不是孤立的 ISA 因果试验，也不代表真实硬件或全部 RVV 实现。
- 4 hart 剩余波动和额外 CPU 尚未完全分解为宿主调度、频率、硬件缓存、QEMU 浮点、SoftMMU/TB 等因素。独立采样的比例不能换算为无采样运行的 CPU 秒数。
- 原始数组、ELF、权重、perf 采样与源码快照仍只保存在本地，公开摘要不等同于独立重建执行证据。E1/E3/E5、Nightly、团队验收和真实 RISC-V/NPU 部署另行完成。
- W5 token 循环、文本生成和会话部署不属于本报告的完成声明。

## 原始记录索引

以下路径相对于本地 ScratchV 开发根目录，均为 Git 忽略的产物索引；不是仓库下载链接。公开的对照 JSON 也使用相对路径，并明确这些源文件不随文档发布。

- 总报告：`output/w4-harts-short-20261021/comparison.html`、`comparison.json`（同目录）。
- 完整模型：`output/w4-harts-short-20261021/optimization-completion-20261022/full-hart-comparison/result.json`；两侧原始报告位于同级 `integration-full-1hart/run/report.json` 与 `integration-full-4hart/run/report.json`。
- 微内核：`output/w4-harts-short-20261021/optimization-completion-20261022/phase3/comparison.json`。
- 正式布局：`output/w4-harts-short-20261021/optimization-completion-20261022/phase4/formal/result.json`。
- RVV：`output/w4-harts-short-20261021/optimization-completion-20261022/phase5/comparison.json`。
- 等待/Softmax：总报告中的 `idle_wait_optimization`、`softmax_callback_reuse`、`softmax_finite_fusion` 与 `softmax_padding_option` 节。
- 计时与 QMP：`output/w4-harts-short-20261021/uart-event-timing-20261022/report-data.json`、`output/w4-harts-short-20261021/qmp-stop-diagnosis-20261022/report-data.json`。

本次文档整理仅引用已有测量和已有身份记录，不重新计算哈希、不重新运行模型。
