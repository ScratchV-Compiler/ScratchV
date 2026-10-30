# W5：端到端文本生成

## 本分支增加了什么

W5 将 W4 的单次 QEMU 前向接成文本生成：本机 Python 准备输入、选择 token、控制循环并校验，
QEMU 中的 RISC-V 程序执行模型。支持原始文本补全、三条固定参考用例及
[官方模板多轮文本](../../../scratchv/runtime/conversation.py)；
[命令行入口](../../../scripts/run_inference.py)、
[部署脚本](../../../scripts/deploy_w5.sh)与[Dockerfile](../../../deployment/w5/Dockerfile)
使用同一条生成流程。

<a id="maintainer-overview"></a>

**先看已经取得的结果。** 2026-10-23 的历史实跑中，完整 28 层 Qwen3-0.6B 使用显式
4 core 配置，三条用例各生成 8 个新 token，**24/24 步与冻结参考一致**；
最大绝对误差 **0.000303984**，每条端到端生成与校验约 **56～61 分钟**。
具体文字、计时范围和模型配置见[实际复现结果](#实际复现结果)。
这些是 `8ad3474` 功能基线的历史数据，不是后续命名和兼容修复的新测量。

**当前限制。** 无 KV cache，每个新 token 都执行一次完整前向并启动新的 QEMU。
固定 8-token 预算不代表自然回答完毕；完整模型三轮聊天、三例独立容器运行及第二人复现
尚无本报告可用的实跑记录，见[已验证范围与剩余问题](#remaining-scope)。

| 想了解什么 | 阅读入口 |
|---|---|
| 输入如何流动、Python 和 QEMU 各做什么 | [实现说明](#整体怎么实现)、[逐步流程表](#generation-flow) |
| 实际生成了什么、运行多快、如何判定正确 | [运行配置](#run-configuration)、[三条结果](#fixed-case-results)、[耗时表](#full-generation-results)、[误差判定](#max-absolute-error) |
| 数据与历史 CI 在哪里 | [报告入口](#report-locations)、[CI/benchmark 快照](#ci-and-benchmark) |
| 如何自己执行并核对 | [如何复现](#如何复现)，命令默认折叠 |

**默认配置保持不变**：单核、MatMul `sequential / 1x4 / original`、Mean `sequential`、
非线性 `libm`、padding `none`。可手动指定 `--cores 1/2/4` 或 `--cores auto`；
auto 按进程可用 CPU 在构建或启动时选择 1/2/4，上限 4，不在 token 间动态调整。
完整模型通过的记录使用下文显式参考策略，不代表默认算术策略也通过同一数值要求。
W5 没有 `--qemu-workspace-layout` 参数，padding 始终关闭。

旧名称与 Python 接口的兼容细节见[兼容说明](#core-compatibility)。

<a id="core-compatibility"></a>

<details>
<summary>展开 core/hart 名称与旧接口兼容说明</summary>

本文的 core 指 QEMU 模拟的 CPU 核。
[hart（hardware thread）](https://docs.riscv.org/reference/debug/introduction.html) 是 RISC-V 的正式术语，指硬件线程；
本项目当前 QEMU 配置中每个模拟核对应一个 hart，因此改用更直观的 core 提高接口可读性，
执行方式不变，旧名称仍兼容；在其他硬件上，hart 与物理核不一定一一对应。
当前命令使用 `--cores`，Python 参数和属性使用 `cores`。

旧 `--harts`、Python 的 `harts` 参数和属性仍兼容。CLI 解析结果的 `args.harts`
是 `args.cores` 的可读写别名，修改任一属性都会同步；默认值、手动值和 `auto`
在 `parse_args()` 返回时仍保持原字符串类型。命令行、普通构造和运行时 API 中，
新旧名称同时传入时必须一致，否则在构建或运行前报错。新报告用 `cores_requested` 记录请求、`cores` 记录实际数量，
并保留等值的 `harts_requested`、`harts` 兼容字段。旧构建记录只有 `harts` 时仍可读取；
两种核数字段均缺失时按原规则视为单核，两种字段不一致或值无效时拒绝复用。
历史报告不改写，复用已有 ELF/bundle 仍须通过原有配置和身份检查。

`dataclasses.asdict(executable)` 同时输出整数 `cores` 和等值的旧 `harts` 字段，
不输出私有核数字段；正式 `executable.json` 的保存和读取方式保持不变。
`dataclasses.replace(executable, cores=2)` 和旧 `harts=2` 写法都会同步两个字段。
`replace()` 会自动补入原值，因此一边为原值、另一边为新值时按新值更新；
两边都是不同于原值的新值且互不相等时拒绝。建议一次只用一个名称更新核数。

</details>

## 整体怎么实现

<a id="generation-flow"></a>

**从提示词到文字结果。** 本节按三条固定用例的实际路径说明：本机 Python 控制生成循环，
QEMU 内的 RISC-V 程序执行完整 Qwen3 模型，ONNX Runtime（ORT）在本机计算参考结果。
`token` 是 tokenizer 划分和处理文字的单位，`token ID` 是它对应的编号；
`logits` 是模型为各候选 token 给出的分数。
QEMU 返回的是这些分数，最终文字由 Python 转换得到。

**当前没有 KV cache（保存前面 token 的注意力中间结果的缓存）。**
每生成一个新 token，都把“提示词 + 已生成 token”作为输入，重新执行完整 28 层模型。
每条固定用例因此需要 **8 次完整前向和 8 次 QEMU 运行**，尚未采用基于缓存、
只计算新 token 的增量 decode，也不是只执行“一次 prefill + 一次 decode”。
这里的 8 次指完整模型前向，不包括运行前的浮点自检小程序。

| 步骤 | 执行位置 | 输入、具体处理与输出 | 实现入口 |
|---|---|---|---|
| 运行前准备 | 本机 Python + 编译器；自检会调用 QEMU | 新构建时，将 ONNX 解析成中间表示，生成 C 程序、写权重包并编译链接出 ELF；复用时检查已有 ELF、权重包和配置。浮点自检与 ORT 参考程序初始化也在生成循环之前完成 | [CLI](../../../scripts/run_inference.py)的 `prepare_executable` / `main` |
| 提示词编码 | 本机 Python | 原始文字经 tokenizer 转成 token 编号。三条用例是原始文本补全，不套聊天模板，不额外插入起止 token | [生成循环](../../../scratchv/runtime/generation.py)的 `generate` |
| 准备本步输入 | 本机 Python | 将当前完整 token 历史右侧补齐到 256 个位置，并构造限制可关注位置的注意力遮罩，得到模型输入数组；有效长度另行保留，供选词使用 | [输入处理](../../../scratchv/runtime/llm_inputs.py)的 `prepare_inputs` |
| 本步参考计算 | 本机 Python 调用 ORT | 固定用例先独立核对本步实际输入与冻结历史一致，再用这些输入运行同一份 ONNX 模型，得到 ORT 参考 logits | [CLI](../../../scripts/run_inference.py)的 `forward`；[参考输入检查](../../../probes/w2_runtime_model/run.py)的 `checked_inputs` |
| 模型前向 | QEMU 模拟执行 RISC-V 程序 | QEMU 装载 ELF、输入二进制和权重；其中的 RISC-V 程序调用生成的模型函数，计算全部输出分数并写入输出内存，通过 UART 串口发送完成状态及输出字节数 | [运行器](../../../scratchv/runtime/riscv_external.py)的 `guest_harness` / `run_external` |
| 读回与逐步校验 | 本机 Python | 验证 UART 成功完成帧后，用 QMP 暂停并复核完成帧，将输出内存保存为 `output.bin`，退出并回收进程；检查输出、退出码和清理结果，再与 ORT 比较，失败立即停止 | [QEMU 适配器](../../../scratchv/runtime/qemu_generation.py)；[CLI](../../../scripts/run_inference.py)的 `compare_step` |
| 选择下一个 token | 本机 Python | 从 QEMU 输出中取最后有效位置的分数，用 greedy 选择最高分 token；同分取较小编号，无额外 Softmax。把实际选出的编号加入历史，未停止则回到“准备本步输入” | [输入与选词](../../../scratchv/runtime/llm_inputs.py)的 `last_valid_logits` / `greedy_token`；[生成循环](../../../scratchv/runtime/generation.py) |
| 结束与结果输出 | 本机 Python | 满足停止条件后，把新生成的编号序列解码为文字；固定用例再核对完整编号、文字、停止原因与步数，保存报告并打印文字 | [生成循环](../../../scratchv/runtime/generation.py)；[CLI](../../../scripts/run_inference.py)的 `main` |

**QEMU 不直接接收提示词字符串，也不直接打印模型回答。** UART 完成消息不是文本生成结果；
模型输出是内存中的 logits，经 QMP 读回后，由 Python 选 token 和解码。
表中的“本机 Python”表示 QEMU 外的控制程序；容器运行时，这套 Python 程序与 QEMU 一起在容器内运行，职责不变。

同一条生成循环复用同一 ELF 和权重包，每步只更新输入；**没有逐 token 重复编译**。
但每步仍启动新的 QEMU 进程并装载权重，不能把复用构建文件理解为 QEMU 或 KV cache 常驻。
前置编译与初始化、生成阶段、最终报告保存各自的计时范围见[结果表说明](#full-generation-results)。

<details>
<summary>展开模型输入输出的具体格式、遮罩和停止规则</summary>

`generate(prompt, *, tokenizer, forward, max_new_tokens, cancelled=None, timeout_seconds=None)`
返回 `GenerationResult`，包含提示词编号、新生成编号、最终文字和 `stop_reason`。

| 项目 | 实际行为 |
|---|---|
| 提示词 | 原始文本模式直接编码，`add_special_tokens=False`；沿用 tokenizer 的 NFC 归一化 |
| token 输入 | INT64 `input_ids[1,256]`；前面放有效编号，右侧用模型的 `pad_token_id` 补齐 |
| 注意力遮罩 | FP32 `attention_mask[1,1,256,256]`；位置 `q` 只允许关注 `k <= q` 且 `k < valid_length` 的位置。允许处为 0，其余为 FP32 最小有限值；不是把补齐的 query 行全部屏蔽 |
| 模型输出 | FP32 `logits[1,256,151936]`；包含全部位置的候选分数，所有元素都必须有限，包括补齐位置 |
| 用于选词的输出 | 取 `logits[0, valid_length-1, :]`，即最后有效位置的完整词表分数；不是固定取第 256 行 |
| 有效词表 | tokenizer 为 151669，模型输出宽度为 151936；若最高分落在不可解码 ID，报错，不过滤后重新选择 |
| 停止顺序 | 新 token 生成后依次检查 EOS、上下文容量 256、新 token 预算；EOS IDs 为 `151645`、`151643` |
| 空间和预算 | 提示词非空且不超过 256，不隐式截断；零预算或初始上下文已满时不调用模型前向 |
| 失败与中断 | 不把异常、超时或取消的部分生成当作成功；适配器负责 QEMU 进程回收 |

形状、遮罩和停止顺序见[输入实现](../../../scratchv/runtime/llm_inputs.py)；
生成循环见[生成实现](../../../scratchv/runtime/generation.py)。

</details>

<details>
<summary>展开如何校验：相同输入、完整分数、逐 token 选择与最终文字</summary>

- **先确认模型程序成功执行。** UART 必须是完整合法帧，状态为 0、输出字节数正确；
  暂停 QEMU 后还会复核帧，并检查转储文件大小、输出有限性、退出码及清理结果。
  收到一条消息，或仅看到进程退出，都不能代替后续数值校验。
- **同输入的数值参考。** `--verify-onnx` 使用相同输入运行 ORT，比较整个输出数组，
  包括补齐位置；要求类型、形状正确、所有值有限、严格 `max_abs < 1e-3`、`rtol=0`，
  并且最后有效位置的 greedy ID 相同。top-2 margin 只作诊断记录，不是额外通过阈值。
  误差如何计算及能说明什么见[误差解释](#max-absolute-error)。
- **独立核对生成输入。** `--reference-suite` 从
  [冻结参考](../../../probes/w5_generation/references.json)维护独立的参考历史，
  `checked_inputs` 逐元素检查实际 token 输入和遮罩，再把核对后的实际输入交给 ORT。
  QEMU 仍接收生成循环准备的实际输入。
- **生成程序自己选择 token。** 每步要求 QEMU、ORT 和冻结参考选出的 ID 一致；
  只有通过后才推进独立参考历史。实际生成循环仍从 QEMU logits 中自行 greedy 选择，
  并把这个结果加入自己的历史，不能用冻结答案替代模型输出。
- **最终用例检查。** 比较提示词 IDs、新 token IDs、解码文字、停止原因，以及前向次数和校验次数；
  同时检查要求出现的文字。只出现 `Paris` 等城市名，或只有最大误差达标，都不足以判整例通过。
- **参考与实跑记录分开。** 冻结参考由[参考采集程序](../../../probes/w5_generation/run.py)运行 ORT 得到；
  其中 `baseline.qemu_e2e=NOT_RUN` 只表示采集参考时未运行 QEMU。之后的完整 QEMU 实跑
  见[实际结果](#fixed-case-results)和[历史数据摘要](reports/validation-summary.json)，不能混用两种记录。

`--reference-suite` 必须使用 `raw` 模式并同时提供 `--verify-onnx` 参考模型，
使用冻结的 token 预算，不接受另传 `--max-new-tokens`；它也不会自动选择算术策略。
完整三例历史实跑显式使用以下选项，不代表默认算术策略也完成了相同的完整模型验证：

```text
--cores 4 --matmul-policy blocked_fma --matmul-tile 2x4
--matmul-rhs-layout column-panel
--mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference
```

</details>

<details>
<summary>展开多轮对话、容器部署和失败处理</summary>

**多轮文本。** `Conversation.from_directory(tokenizer_directory, system_prompt=None)`
加载同一快照的 `tokenizer_config.json` 官方 `chat_template`，通过 Jinja2 不可变沙箱
渲染，固定 `add_generation_prompt=True`、`enable_thinking=False`。生产路径使用
tokenizers/Jinja2，不依赖 Transformers/Torch。仅支持文本 system/user/assistant，
system 只在构造时提供；每轮重新渲染完整历史并编码，不截断旧消息，不缓存 KV。

`turn(...)` 只有成功且至少生成一个 token，才同时保存 user 和 assistant；异常、取消、
超时、零预算或没有生成空间时历史不变。EOS-only 可以保存合法的空 assistant 回复。
`history` 是不可变快照，`reset()` 清除已完成轮次但保留初始 system。历史保存文本，
不承诺下一轮 token IDs 直接拼接；超过 256-token 容量会在前向前失败。单个会话拒绝
并发或重入 turn/reset。原始三例不使用聊天模板。

**部署与报告。** 容器只打包运行所需源码和依赖，资产只读挂载到 `/assets`，构建及报告
写入 `/output`，以调用者 UID/GID 运行。默认镜像 `scratchv-w5:local`、基础镜像
`ubuntu:24.04`；可显式设置 `SCRATCHV_CONTAINER_ENGINE`、`SCRATCHV_W5_IMAGE`、
`SCRATCHV_W5_BASE_IMAGE`、`SCRATCHV_W5_ASSETS` 和 `SCRATCHV_W5_OUTPUT`。
路径可含空格，挂载路径不能含逗号。脚本检查引擎和资产后构建、运行，不安装引擎，
不下载模型，不更改 WSL 或 Windows。依赖来自
[pyproject.toml](../../../pyproject.toml) 的 `llm-deploy`，不另维护一份版本锁。

部署预检和构建客户端各有独立进程组，SIGINT/SIGTERM 时转发终止并等待清理，必要时
5 秒后强制结束，返回 130/143。容器运行阶段保留 `exec ... run --init` 信号转发，
QEMU 适配器负责其子进程回收。CLI 最终保存报告失败时向 stderr 说明保存错误，
保留已有生成失败或取消的退出码 1/130/143；原本成功或仅构建成功时改为退出 1。
数值、参考文本等原始错误不会被保存错误覆盖，也不打印误导性的成功报告路径。
运行中的 SIGINT/SIGTERM 均记录为取消，分别保留退出码 130/143。

本分支的报告检查器支持 `--log-exit-code-file`：指定后要求生成与日志退出码均为 0，
日志状态缺失、不可读或内容无效也返回 FAIL。外层 Bash 分别保存
`runner-exit-code.txt` 和 `log-exit-code.txt`；生成失败优先保留原退出码，
生成成功而 `tee` 失败时返回日志退出码。下方固定 `8ad3474` 的历史复现命令保留原版
参数，仍须按其成功要求单独核对日志退出码，不向旧检查器传入新参数。

</details>

## 实际复现结果

下文使用**“本机直接运行”**和**“容器内运行”**区分是否通过 Docker；
具体执行链路、运行环境和计时范围见单次前向表后的说明。

<a id="run-configuration"></a>

**历史完整模型运行配置。** 以下数字来自 **2026-10-23 的实跑**，代码功能基线为
`8ad347477f7f5ba15b2da7c91178b3c6668456cc`。后续报告、取消、日志修复及 core 命名兼容
工作只做相关验证；本次整理文档没有重跑模型，不能把旧数字当作当前版本的新测量。

| 项目 | 这次三条固定用例的配置 |
|---|---|
| 模型 | `Qwen/Qwen3-0.6B`，完整 28 层，FP32 |
| 输入与生成 | batch=1，固定序列长度 256；原始文本补全，不使用聊天模板；每条生成 8 个新 token |
| 生成方式 | greedy，每次取最高分 token；无 KV cache，每步执行完整前向 |
| 模拟器与核心数 | QEMU 模拟 RV64GC，显式使用 4 core；三条用例复用同一 ELF 和权重包 |
| 算术与布局 | `blocked_fma / 2x4 / column-panel`；Mean `two-four-lane-last-axis`；非线性 `fp32-reference`；工作区 padding `none` |
| 参考与环境 | 本机 WSL2 Ubuntu，4 个可用逻辑 CPU；ORT 通过 `CPUExecutionProvider` 计算参考输出 |

<details>
<summary>展开模型快照和工具版本</summary>

- 模型快照：`c1899de289a04d12100db370d81485cdf75e47ca`，见
  [模型清单](../../../probes/w1_qwen3_export/manifest.json)及[冻结参考](../../../probes/w5_generation/references.json)。
- QEMU 8.2.2（Ubuntu 包 `1:8.2.2+ds-0ubuntu1.18`），Zig 0.14.1；
  目标为 `riscv64 / RV64GC / LP64D / QEMU virt` 裸机，数学库为 Zig 随附 musl。
- Python 3.12.3、NumPy 2.2.6、ONNX Runtime 1.22.1、tokenizers 0.21.4。
- 以上为历史记录中的版本；完整字段见
  [数据摘要](reports/validation-summary.json)的 `configuration`。不据此推断当前机器或最新依赖版本。

</details>

<a id="fixed-case-results"></a>

**三条实际输出与停止原因。** 下表来自实际 QEMU 运行报告，并已与冻结参考逐项核对，
不是只展示预设的期望答案。每条均为 8 个新 token、8 次完整前向，`reference_passed=true`；
全部 24 次前向的数值和生成结果通过，QEMU 进程及进程组均已回收。

| 用例 | 原始输入提示词 | 实际新生成文字 | 实际停止原因 |
|---|---|---|---|
| France | `The capital of France is` | ` Paris. The capital of Italy is Rome` | `max_new_tokens`（达到 8 个 token 预算） |
| Germany | `The capital of Germany is` | ` Berlin. The capital of France is Paris` | `max_new_tokens`（达到 8 个 token 预算） |
| Japan | `The capital of Japan is` | ` Tokyo. The capital of the United States` | `max_new_tokens`（达到 8 个 token 预算） |

三条输出开头均保留一个空格。**此次没有遇到 EOS，停止是因为预算用完，不代表自然回答完毕。**
例如 Japan 的输出停在 `the United States`，后续句子尚未补全；
这些是原始提示词的续写，不是聊天模板下的问答质量评测。

<details>
<summary>展开三条用例的实际输入及新生成 token 编号</summary>

| 用例 | 提示词 token IDs | 实际新生成 token IDs |
|---|---|---|
| France | `[785,6722,315,9625,374]` | `[12095,13,576,6722,315,15344,374,21718]` |
| Germany | `[785,6722,315,9856,374]` | `[19846,13,576,6722,315,9625,374,12095]` |
| Japan | `[785,6722,315,6323,374]` | `[26194,13,576,6722,315,279,3639,4180]` |

</details>

对应的预期结果见[冻结参考](../../../probes/w5_generation/references.json)，
实际结果与 24 步误差见[历史数据摘要](reports/validation-summary.json)的 `three_case_generation`。
[报告入口](#report-locations)说明哪些文件随仓库保存、哪些原始数据只保留在本机。

<a id="full-generation-results"></a>

**三条固定用例的耗时与数值误差。** 对应上述实际文字结果，均为本机直接运行。

**每行对应同一条提示词生成 8 个新 token：串行执行 8 次前向，每次使用 4 core。**
本次按固定 token 预算结束，不代表让模型自由回答到自然结束。
“QEMU 仿真总耗时”是这 8 次模拟器运行的实际经过时间相加。
**本节各表的耗时均按实际经过时间展示，不把同时工作的多个线程的 CPU 用时相加。**
累计 CPU 用时仍保留在[原始数据摘要](reports/validation-summary.json)中，不作为运行速度指标列入表格。

| 用例 | 端到端生成与校验耗时（秒） | QEMU 仿真总耗时（秒） | 最大绝对误差 |
|---|---:|---:|---:|
| France | 3375.024 | 3199.695 | 0.000303983688 |
| Germany | 3668.852 | 3482.999 | 0.000262737274 |
| Japan | 3489.655 | 3315.397 | 0.000197887421 |

<details>
<summary>表中名称说明：</summary>

- **端到端生成与校验耗时**：从开始处理这条提示词，到全部 8 个新 token 生成、转换为最终文字并完成主要校验，
  实际经过的时间。包含输入处理、每步 QEMU 启动与权重装载、模型计算、结果读回和进程清理，
  以及参考计算、逐步数值校验和最终文字与冻结参考的核对。
  **最终文字已经生成并经过校验，这部分时间已包含在内；不含前置编译，也不含最后的报告整理、保存和终端显示。**
  计时顺序是：**处理提示词 → 逐个生成 8 个 token 并校验 → 转换为最终文字并核对 → 计时结束**；
  随后才整理、保存报告并打印文字。因此，它不是纯模型计算时间，也不是从部署到显示结果的全部链路耗时。
- **QEMU 仿真总耗时**：同一条用例生成 8 个新 token 时，8 次 QEMU 仿真从启动到退出回收的
  **实际经过时间之和**。包含模拟器启动、装载权重、模型计算、输出转储和退出清理；
  不包含前置编译、QEMU 启动前的输入准备和文件检查，也不包含进程外的 ORT 参考计算和完整结果校验。
  **表中是 8 次仿真的累计耗时，不是单次前向耗时，也不是只执行模型算子的纯计算时间。**
  它已包含在“端到端生成与校验耗时”中，两者不能相加。
- **最大绝对误差**：该用例中，模型输出分数与参考结果之间最大的绝对偏差，计算和判定方法见下文。

</details>

<details>
<summary>8 次 QEMU 运行如何生成一条结果？</summary>

三条提示词初始都为 5 个 token。第 1 步用这 5 个 token 选出第 1 个新 token；
第 2 步用提示词和已选结果，共 6 个 token，选出第 2 个；依次继续，
8 次前向的有效输入长度为 **5、6、7、8、9、10、11、12**，每次都右填充到 256。
第 8 个新 token 产生后，历史长度变为 13，达到生成预算，不再执行第 9 次前向。

每步用 greedy 选出下一个 token，最终将 8 个新 token 转成同一条输出，
不是生成 8 个回答再择优。无 KV cache，复用的是同一 ELF 和权重文件；
Python/QEMU 的具体职责见[流程表](#generation-flow)。

</details>

<a id="max-absolute-error"></a>

<details>
<summary>最大绝对误差：和谁比、怎么算、能说明什么？</summary>

1. **和谁比较。** 同一份 Qwen3 模型、相同的 token 输入及遮罩，分别由 ScratchV 编译后在 QEMU
   中执行，以及由 ONNX Runtime（ORT）参考程序计算。对比的是模型输出的 **logits**，
   即选下一个 token 时使用的候选分数；这些原始分数不是概率，也不是最终文字。
   ORT 在这里提供数值基准，不是判断回答是否符合事实的裁判。
2. **怎么算。** 每个对应位置的 QEMU 分数减去 ORT 分数，去掉正负号，再选出最大的差值：
   `最大绝对误差 = max(|QEMU 分数 − ORT 分数|)`。
   例如参考分数为 1.2500、QEMU 为 1.2503，这个位置的误差约为 0.0003。
   每次前向比较全部 **38,895,616 个输出分数**（包括固定长度中补齐的位置），
   表中再取该用例 **8 次前向中的最大值**，不是平均误差，也不只检查最终选中的 token。
3. **为什么小于 0.001 才通过。** 这是[开发计划](../开发计划.md)为完整 QEMU 前向规定的数值容差，
   W5 沿用 W4 的 `ATOL = 1e-3`。它给不同浮点计算实现之间的偏差设定上限；
   FP32 舍入、运算顺序和数学函数实现差异可能使两个后端的数值不完全一样。
   代码要求严格 **`max_abs < 0.001`**，等于 0.001 也不通过；
   `rtol=0` 表示不再因分数较大而增加相对误差宽限。
   **0.001 是项目约定的分数偏差上限，不是“错误率低于 0.1%”，也不是普遍适用的正确性证明。**
4. **能否说明生成结果正确。** 单看误差小，只能说明这次计算的分数接近参考。
   当两个候选 token 的分数很接近时，小偏差也可能改变排名、选出另一个 token。
   因此，完整用例通过还要求：全部输出是有限数值，没有 NaN 或正负无穷；
   每一步实际选出的 token 与当步 ORT 结果及[冻结参考](../../../probes/w5_generation/references.json)一致；
   最终 token 序列、文字和停止原因也与冻结参考完全一致。不能仅凭误差小于 0.001 判整条用例通过。

上述计算和检查见[生成入口](../../../scripts/run_inference.py)的 `compare_step` 与
`reference_suite` 路径。**这里的三条固定用例共 24 步同时通过了数值与生成结果检查**，
说明所列历史版本和显式配置在这些用例上与参考一致；不代表所有提示词都会得到相同结果，
也不保证模型生成的内容在事实层面永远正确。本次只解释已有结果，没有重新执行模型。

</details>

<details>
<summary>展开三例合计耗时、内存和采样限制</summary>

整次测量 10540.190 s（约 2 小时 55 分 40 秒），包括准备、ORT 和数值比较。
24 次前向的**启动至完成通知耗时**合计 9990.355 s，平均 **416.265 s/token**；
**QEMU 仿真总耗时**合计 9998.091 s。这两个耗时指标的起止位置见[单次前向表说明](#container-results)。
**采样到的内存峰值**：模拟器 5.406 GiB、主控程序 4.254 GiB、同时运行的进程合计 7.644 GiB。
17/24 步各有一次内存采样读取失败，属于部分采样（`partial`），可能漏掉更高的瞬时占用。
同时进程合计是在同一采样时刻求和后取最大值，不能直接相加各进程在不同时刻的峰值；
按 RSS 求和也可能重复计算共享内存页。

</details>

<a id="container-results"></a>

**独立单次完整前向与干净容器 Paris。** 两次都只生成 `[12095]`、文本 ` Paris`，
完整输出分数的最大绝对误差同为 `0.000172138214`；比较对象和通过条件与
[上面的误差说明](#max-absolute-error)相同，这里每边只生成 1 个新 token。
容器内重新构建一次，没有重复三例套件；
155,582,464 字节完整输出与本机单次前向逐字节相同。

**两列采用相同的程序内部计时方式，但构建方式不同；这是运行记录，不是容器性能开销对比。**

| 测量项 | 本机直接运行（复用构建） | 容器内运行（重新编译） |
|---|---:|---:|
| 推理程序内部耗时（秒） | 444.416 | 547.454 |
| C 程序编译与链接耗时（秒） | 复用已有构建，本次未重新编译完整模型 | 93.868 |
| 启动至完成通知耗时（秒） | 418.954 | 415.058 |
| 模拟器进程总耗时（秒） | 419.383 | 415.478 |
| 采样到的模拟器内存峰值（GiB） | 5.398 | 5.398 |

**本机直接运行**表示整条推理链路在 Docker 容器外执行，**容器内运行**表示同一链路在容器内执行。
两边的模型计算都由 QEMU 完成。本次均在同一台电脑的 WSL2 Ubuntu 环境中，
不是 Windows 原生与另一台电脑的对比；也不表示每次都重新下载模型、安装依赖或编译。

**这张表的计时范围。** 以下时间项均为实际经过时间，包含关系不能相加。
本机复用构建，容器的 **93.868 秒编译已经包含在 547.454 秒内部耗时中**；
构建工作不同，不能把两列总时间之差视为容器性能损耗。

| 名称 | 从哪里计到哪里 | 包含与排除 |
|---|---|---|
| 推理程序内部耗时 | 程序开始准备 → 按需构建、自检、参考程序初始化 → 生成与校验 → 程序内收尾 | 使用 `run/report.json` 的 `elapsed_seconds`；只计本次实际发生的工作，不计此前构建，也不含内部计时开始前的 Docker 镜像准备或容器启动 |
| C 程序编译与链接耗时 | 启动编译工具 → 生成并检查模型 ELF | `compile_seconds` 包含编译器启动、C 编译、汇编、链接、日志保存及 ELF 检查；不含此前的模型解析、C 源码生成、权重打包和之后的推理 |
| 端到端生成与校验耗时 | 处理提示词 → 得到最终文字并完成主要校验 | 是内部耗时的中间阶段，不含前置编译、浮点自检、ORT 初始化及最后报告保存、终端显示；含每步 ORT 参考计算。三例各生成 8 个 token，本表每边只生成 1 个 |
| 启动至完成通知耗时（原 UART 耗时） | QEMU 启动 → Python 确认合法的成功完成帧 | 包含启动、权重装载、模型计算和通知接收；不是串口传输时间，也不是纯算子时间。输出读回和数值校验尚未完成 |
| 模拟器进程总耗时 | 同一次 QEMU 启动 → 进程退出并回收 | 包含上项，以及暂停、输出转储和退出清理；不含进程外编译、ORT 参考计算和完整结果校验 |
| 采样到的模拟器内存峰值 | 定时读取 QEMU 驻留内存 RSS，取成功采样的最大值 | 包含模拟器、权重和运行缓冲；不是模型文件大小、配置内存上限或整机占用。可能漏掉瞬时峰值；1 GiB = 1024³ 字节 |

UART 是串口通信接口；本项目用它发送完成状态，完整分数经 QMP 读回。
关于 QEMU 的输入输出和通知顺序，见[流程表](#generation-flow)。

<details>
<summary>展开单次前向的历史计时细节与可比性限制</summary>

- 本机表中采用内部计时 **444.416 秒**。此前外层测量的 **448.263 秒**仍在原始记录中，
  不混入本表；其中端到端生成与校验为 **441.033 秒**。
  容器没有独立记录同口径的生成与校验耗时，不能用其他计时项代替。
- 本机复用产物中记录的 **85.680 秒**是历史模型编译时间，不计入本次。
  小型浮点自检程序仍需编译，本机为 **0.110 秒**、容器为 **0.047 秒**，
  已包含在各自内部耗时中，与完整模型编译分开。自检用于检查工具链浮点乘加行为，不是重跑完整模型。
- 两列内部耗时相差 **103.038 秒**，容器本次模型编译 **93.868 秒**是主要的已测量差别；
  其他准备、缓存和运行开销未完整分段计时，不能把差值全归于编译或 Docker。
  模拟器进程耗时 **419.383 / 415.478 秒**接近，单次记录不能用于认定容器更快或更慢。
- 本机从完成通知到退出回收约 **0.429 秒**，是 `419.383 - 418.954`；
  这是进程总耗时的一部分，不能再次累加。
- 容器另有运行器计时 **430.481 秒**，包含权重传输、QEMU、后检查和清理，
  不含更早的输入打包、写入、ELF/bundle 检查及完整 ORT 参考计算，因此不是端到端生成与校验耗时。
- 本机主控程序/同时进程采样内存峰值为 **4.119 / 7.210 GiB**。
  容器 QEMU 有 **2071 个有效样本、1 次读取失败**，标为 `partial`；
  未清空本机 Linux 页缓存，不宣称冷缓存性能。
- 容器单 prompt 的 `full_model_gate=NOT_RUN` 表示没有运行三例套件，
  不能据此说没有运行完整模型。

</details>

<a id="small-graph-results"></a>

**小图与多轮行为。** 历史真实容器默认单核、手动 4 core、auto 解析为 4 的小图均
生成 IDs `[2,3,4,6]`、文本 `2 3 4`，EOS 停止，逐步完整 logits 对 ORT 的最大误差为 0，
padding `none`。不同上下文和中断后的再次生成可复用同一配置 ELF/bundle；
历史 QEMU 超时/取消及 Docker 构建 SIGINT/SIGTERM 后未遗留目标进程。
官方聊天模板的有/无 system 三轮渲染和 token 序列与 Transformers 一致；这些回复由
模板验证预设，未表示完整模型三轮聊天实跑。

<details>
<summary>展开 2026-10-25 小图验证细节：报告保存、Docker 取消、清理与复用</summary>

**2026-10-25 局部修复验证。** 基于 `e93233143d2323de7de6888ff7d73bdbf50784c3`
加入报告保存与取消处理修复后，注入权限、磁盘满、I/O 和序列化失败，CLI 均保留原始生成错误
或取消退出码；原本成功的运行在最终报告无法保存时返回失败，不误报成功。后续调用
仍可正常保存。2026-10-25 的该项验证没有重新执行完整模型，上方完整数值与性能数据仍属于历史基线。

真实 Docker 小图在默认 1 core 和手动 4 core 下，正常生成均得到 `[2,3,4,6]`、
文本 `2 3 4`，逐步 ORT 最大误差为 0。向 Docker 客户端发送 SIGINT/SIGTERM 后，
报告均为 `CANCELLED`，退出码分别为 130/143；容器外的客户端进程已回收、QEMU 进程组为空、
`--rm` 容器消失，0.3 秒后复查仍无残留。中断检查以临时 `-S` 包装器控制真实 guest
的暂停时机，不修改生产 harness；正常生成和恢复使用实际小图前向。

两种信号之后均复用原 ELF/bundle 恢复生成，IDs、文本及全部逐步 logits 仍与基线一致；
4 core 的恢复还实测 `auto` 解析为 4，两组 padding 均为 `none`。这些结果说明小图
取消、清理和复用正常，不扩大为完整模型重跑或完整多轮聊天结果。

</details>

<a id="report-locations"></a>

**报告放在哪里。**

| 要找的内容 | 固定入口 | 是否能随 PR 查看 |
|---|---|---|
| W5 功能、复现、数值、耗时、内存及未完成项 | 本 README；[完整结果](#实际复现结果) | 是 |
| W5 三例逐 token 数值和资源、本机/容器运行记录、小图与清理结论 | [validation-summary.json](reports/validation-summary.json) | 是，随仓库保存的历史数据摘要 |
| W4 软仿优化过程与单核/4 core 完整前向对照 | [W4 软仿报告](../W4/softsim-optimization-2026-10-22.md)及[汇总数据](../W4/reports/full-hart-comparison-2026-10-22.json) | 是，已有仓库文件 |
| CI 原始测试明细、benchmark 详细数据和诊断 | [固定历史 run 的附件](#ci-and-benchmark) | 在 Actions 中可查看/下载，有保留期限 |
| 本地交互页面 `comparison.html` 与大体积原始产物 | 忽略的 `output/`；[本地文件定位](#local-raw-artifacts) | 否；不能把本机文件链接当作 PR 报告链接 |

公开 JSON 是现有记录的白名单摘录，保留测量单位、历史版本、逐步数值及采样缺失；
不含权重、ELF、大数组、机器绝对路径或进程 ID。它不是本次重新执行的结果。
本地交互报告不再作为维护者查找 W5 结果的必经入口。

<a id="local-raw-artifacts"></a>

[公开数据摘要](reports/validation-summary.json)收录上述关键数值及逐 token 结果，可随仓库保存。
下面只是本机诊断文件定位，不是 GitHub 下载链接。路径均相对于执行该次记录的 W5
工作树根目录；原始产物在其 `output/` 中，不提交权重、ELF 和逐步 logits：

<details>
<summary>展开本地原始文件清单（仅供本机定位，不是 GitHub 下载链接）</summary>

- `output/w5-followup-20261023/qemu-reference-4hart/run/report.json`：
  完整三例；`run/` 的上一级目录内 `steps.json`、`numeric-steps.json`、`resources.json` 为计时与资源细节。
- `output/w5-followup-20261023/qemu-final-single-forward-4hart/`：本机独立单次前向。
- `output/w5-container-20261023/paris/run/report.json`：容器完整 Paris 单 token。
- `output/w5-container-20261023/small-summary.json`、`small-runs/` 与 `build-cancel/`：
  容器小图、构建取消和进程清理记录。
- `output/w5-hardening-20261025/docker-final/test_external_signals_reap_gue{0,1}/docker-cancellation-summary.json`：
  2026-10-25 的 1/4 core 小图外部取消、进程清理和恢复结果。

</details>

<a id="remaining-scope"></a>

完整三例的独立容器运行以及其他执行人的独立复现记录仍未取得；不能把本机三例、
容器单 token 或后续局部验证合计成这些结果。现有实现仍是无 KV cache、逐 token
重启 QEMU 与装载权重；常驻运行、多轮完整模型语义和其他算术策略的完整结果未在
上述实跑范围内。

完整 W5 专项成功记录也尚未收录于上述历史快照；本次未检查远端是否已有后续运行。
普通 CI 与 benchmark 的独立范围见下面折叠内容。

<a id="ci-and-benchmark"></a>

<details>
<summary>展开历史 CI、benchmark 与 W4 对照（不是本次完整模型运行结果）</summary>

**版本与验证范围。** 以下固定保留 2026-10-27 整理的历史记录，不代表后续推送的 CI 状态。
PR head `e8c94a7`、测试合并候选 `fa79ae4` 的普通 CI 为
**14 个 job 成功、7 个按条件跳过**。本轮 core 命名兼容的相关验证范围与结果见
[测试说明](../../../tests/llm-deploy/w5/README.md)。旧远端 CI 不能证明后续改动已通过验证，
局部验证也不表示当前版本重跑了完整模型。

**历史 CI 与 benchmark 快照（2026-10-27 核对）。** [普通 CI #40173064525](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525)
对应 PR head `e8c94a71becfae96c4171674e332ed1239335c4d`，
实际执行 `fa79ae4d0f697f968574d647347076a013429684`（与当时 main `735dd00` 的合并候选）。
下表是该次运行的快照，不是每次打开 README 都会刷新的实时状态。

**CI 各项结果与附件下载（完整 W4/W5 本次未运行）**

| 数据 | 该次结果与范围 | 直接查看 / 下载 |
|---|---|---|
| 通用测试 | **8321 passed、93 skipped**；额外模拟器检查 **7 passed** | [test job](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/job/121107820034)、[test-reports](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/artifacts/12591298436) |
| W4/W5 单测与真实小图 | **1769 passed、4 skipped**；不代表完整 28 层模型通过 | [小图 job](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/job/121107851526)、[w4-unit 附件](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/artifacts/12591112082)中的 `tests.xml` |
| 官方多轮模板 | **2 passed**；验证模板及 token 序列，不是完整模型三轮生成 | [w5-multiturn 附件](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/artifacts/12590439548) |
| 编译器 benchmark | job **SUCCESS**，但对照存在无效旧基线，不能表述为全项通过，详见下表 | [benchmark job](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/job/121107819853)、[benchmark-reports](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/artifacts/12590609469) |
| 课题 06 benchmark | job **SUCCESS**；全量报告实际为 **12 PASS / 11 FAIL**，CI 严格检查的范围更小 | [Topic 06 job](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/job/121107819972)、[topic06-benchmark-reports](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/artifacts/12591146751) |
| 完整 W4/W5 | **NOT RUN**：普通 PR 未显式启用完整前向和三例容器生成 | [范围汇总 job](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525/job/121108398602)；本次没有完整模型性能附件 |


**已核对的 benchmark 数字与限制。** 这些是上述 CI 附件中已有的结果，2026-10-27
只下载并整理，未重新执行。它们测试解释器/编译器等范围，不能替代 W5 完整生成计时。

**注意：Topic 06 为 12 PASS / 11 FAIL，单算子对照的旧基线无效；job 成功不代表所有基准通过。**

**benchmark 详细数字、失败原因与统计范围**

| 附件内的报告 | 实际数据 | 如何理解 |
|---|---|---|
| `benchmark-reports / dsl_bench` | **23/23**；总耗时 **0.088 s**，总指令数 **153** | 仅该 DSL 基准的结果 |
| `benchmark-reports / ir_interpreter` | **6/6**，最大误差 **4.6566e-10** | 仅解释器用例的数值对照 |
| `topic06-benchmark-reports / report.md` | **12/23 PASS（52.2%），11 FAIL** | 8 项 TinyFive 输入 ABI 不支持；3 项 `dynamic_add` 计数增加，即使总动态指令减少也被标为回退 |
| `benchmark-reports / single_op_bench.md` | 表中 **30 行 FAIL**；当前版本 **78/78** 组合有效，旧基线 **78/78** 无效 | 旧基线有标签解析错误，不能用旧版 0 指令数算加速比，也不能把表中 FAIL 都解释为当前代码失败 |

Topic 06 的 CI 只对 activation/elementwise/loop 三组使用失败退出选项；
全量报告未设置该选项，所以“job 成功”和“全量 11 项 FAIL”可以同时出现。
这些是当时核对发现的报告/基准限制，尚未定位其引入版本，不据此断言由 W5 改动造成。
当前仅整理可见结果，没有修改 CI 或修复这些基准。


GitHub 运行页的“Load 摘要”加载的是工作流写入 `GITHUB_STEP_SUMMARY` 的已保存 Markdown；
点击它不会重跑测试或模型。附件下载通常需登录 GitHub。上述小图/多轮附件预计
2026-11-08 到期，通用测试/benchmark 附件预计 2026-11-24 到期；平台删除或权限变化也会
使下载失效。本文保留核心结果，完整诊断日志仍以附件为准。

**历史 W4 单次前向对照。** 2026-10-22 的同配置记录中，1 core 为 **1384.65 s**，
4 core 为 **410.53 s**，约 **3.37×**；CPU 总时间 **+5.46%**，输出逐位相同。
见[W4 软仿报告](../W4/softsim-optimization-2026-10-22.md)及
[原始汇总 JSON](../W4/reports/full-hart-comparison-2026-10-22.json)。
这只有一组有序对照，未使用旧计时异常记录，不能当作 W5 三例端到端生成的加速比。

</details>

## 如何复现

<details>
<summary>展开完整复现说明、环境要求、命令与验收标准</summary>

**本地已有环境。** 从仓库根执行，激活项目 Python 3.12 环境并按
[Linux 开发说明](../LINUX_REPRODUCTION.md)安装依赖。已有固定资产时，原始补全命令为：

```bash
python scripts/run_inference.py \
  --model assets/qwen3-0.6b/onnx/model.onnx \
  --tokenizer-dir assets/qwen3-0.6b/tokenizer \
  --prompt 'The capital of France is' --max-new-tokens 1 \
  --verify-onnx assets/qwen3-0.6b/onnx/model.onnx \
  --cores 4 --matmul-policy blocked_fma --matmul-tile 2x4 \
  --matmul-rhs-layout column-panel \
  --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
  --timeout 1800 --total-timeout 3600 --output-dir output/w5-paris-new
```

这会执行一次完整 28 层模型。只构建可加 `--build-only`，其 `BUILD_ONLY` 不表示
完成生成。复用时以 `--reuse-build output/w5-paris-new/executable.json` 替换
`--model ...`，仍传相同 tokenizer 和所有原有算术/core 配置，并使用新的输出目录。
元数据可能含绝对路径，不直接把本机构建的产物搬进容器复用；跨环境从模型重新构建。

聊天模式将上例 `--prompt ...` 换为下列参数，其他模型、参考策略及输出目录参数保留：

```text
--mode chat --chat-turn 'Hello.' --chat-turn 'What is 1+1?'
--chat-turn 'Reply with one word.' --system-prompt 'Give brief answers.'
```

这会按顺序运行三轮，前两轮回答也计入后续 256-token 容量。模板一致性检查不等于
完整模型三轮生成已实跑。相关测试命令集中在
[W5 测试说明](../../../tests/llm-deploy/w5/README.md)。

**固定版本的独立容器复现。** 以下步骤在同一 Bash 会话依次执行，使用独立的
Ubuntu 24.04 x86_64 环境，不复制作者的虚拟环境、`output/`、ELF、bundle 或报告。
三例共用一次新构建、执行 24 次前向，不先另跑 Paris，也不重新录制 ORT 参考。
固定提交 `8ad347477f7f5ba15b2da7c91178b3c6668456cc` 是已完成完整数值实跑的功能基线；
该提交用于复现既有完整数值结果，不含后续报告保存、SIGTERM 分类及 core 名称兼容修复；
这些行为另用相关小图或 Python 测试验证，不将旧完整结果标为修改后的重跑。
该固定提交也尚不支持 `--cores`，所以下面的历史复现命令保留 `--harts`；
运行当前分支时优先使用 `--cores`。这次名称兼容修改没有重跑完整模型。

| 条件 | 固定内容或要求 |
|---|---|
| 主机工具 | Bash、Git/Git LFS、Python 3.12、Docker Engine/Buildx、GNU time、timeout、setsid、tar |
| 模型 | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，FP32 ONNX 及同版本 tokenizer |
| 基础镜像 | `public.ecr.aws/ubuntu/ubuntu@sha256:5ce7043d3beb84e91bfeecacb5fafc80602bf247a1dd6016bc0fe7d69d707eca` |
| 原实跑工具 | Python 3.12.3、QEMU 8.2.2（Ubuntu `1:8.2.2+ds-0ubuntu1.18`）、Zig 0.14.1、ORT 1.22.1、NumPy 2.2.6、Torch 2.7.1+cpu |
| 原容器引擎 | Docker Engine 29.9.2、Buildx 0.38.1 |
| 准备后的资源 | 至少 4 个可用逻辑 CPU、10 GiB 可用内存、10 GiB 剩余磁盘；初始建议预留 25 GiB 磁盘 |
| 时间预算 | 单步 1800 s，总预算 16500 s，外层 280 分钟，强制清理再留 30 s |

镜像 digest 只固定基础镜像；apt 与未逐项锁定的间接依赖仍可能变化。记录实际镜像、
工具和依赖版本，先核对差异是否影响结果比较，不默认为相同环境。这里沿用原资产
身份检查，不增加源码、输出或报告哈希。保留部署前后镜像 ID 作为运行记录，
不要求两者必须相等：Buildx 的缓存构建仍可能更新 provenance/attestation 元数据，
使 manifest ID 改变。ID 变化本身不足以说明运行内容改变，也不代表版本差异可忽略；
继续记录固定源码、模型和实际工具/依赖版本，复用既有 ELF/bundle 检查。

**1. 新 clone 并固定提交。**

以下命令在同一个 Bash 会话按顺序执行，从有足够空间的父目录开始。
目录名 `scratchv-w5-repro` 必须不存在；重试时改用新的目录名，保留失败记录。

```bash
set -euo pipefail
export W5_COMMIT=8ad347477f7f5ba15b2da7c91178b3c6668456cc
test ! -e scratchv-w5-repro
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout --filter=blob:none \
  https://github.com/ScratchV-Compiler/ScratchV.git scratchv-w5-repro
cd scratchv-w5-repro
git fetch --no-tags origin refs/pull/115/head
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach "$W5_COMMIT"
test "$(git rev-parse HEAD)" = "$W5_COMMIT"
test -z "$(git status --porcelain)"

export W5_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SCRATCHV_CONTAINER_ENGINE=docker
export SCRATCHV_W5_IMAGE="scratchv-w5:independent-$W5_RUN_ID"
export SCRATCHV_W5_ASSETS="$PWD/output/w5-independent-assets"
export SCRATCHV_W5_OUTPUT="$PWD/output/w5-independent-$W5_RUN_ID"
export SCRATCHV_W5_BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu@sha256:5ce7043d3beb84e91bfeecacb5fafc80602bf247a1dd6016bc0fe7d69d707eca
test ! -e "$SCRATCHV_W5_ASSETS"
test ! -e "$SCRATCHV_W5_OUTPUT"
mkdir -p "$SCRATCHV_W5_OUTPUT"
date -u +%FT%TZ > "$SCRATCHV_W5_OUTPUT/start-utc.txt"
{
  git rev-parse HEAD
  git status --porcelain
  git remote -v
} > "$SCRATCHV_W5_OUTPUT/source.txt"
{
  cat /etc/os-release
  uname -a
  lscpu
  free -b
  df -B1 .
  python3 --version
  git --version
  git lfs version
  docker version
  docker buildx version
  docker info --format 'OS={{.OSType}} Arch={{.Architecture}} CPUs={{.NCPU}} Memory={{.MemTotal}}'
  command -v setsid timeout tar
  /usr/bin/time --version
} > "$SCRATCHV_W5_OUTPUT/runner-resources.txt" 2>&1
docker info > /dev/null
test "$(uname -m)" = x86_64
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
printf 'Run output: %s\n' "$SCRATCHV_W5_OUTPUT"
```

检查 `runner-resources.txt`：容器外的 Linux 环境应为 Ubuntu 24.04，Docker daemon 也运行在该 Linux 环境中；
若使用远端 daemon、容器资源配额不足或没有引擎，先停止并回报，不用修改 WSL 来凑条件。
挂载目录不能包含逗号。不要把私人凭据写入报告。

**2. 构建镜像并记录实际环境。**

复用现有 Dockerfile；随后部署脚本会再检查构建并复用 Docker 缓存。
基础镜像和构建参数不变，不使用作者构建好的镜像。

```bash
docker build --file deployment/w5/Dockerfile \
  --build-arg "BASE_IMAGE=$SCRATCHV_W5_BASE_IMAGE" \
  --tag "$SCRATCHV_W5_IMAGE" . \
  2>&1 | tee "$SCRATCHV_W5_OUTPUT/image-build.log"
docker image inspect "$SCRATCHV_W5_IMAGE" \
  > "$SCRATCHV_W5_OUTPUT/container-image.json"
docker image inspect --format '{{.Id}}' "$SCRATCHV_W5_IMAGE" \
  > "$SCRATCHV_W5_OUTPUT/image-id-before.txt"
docker run --rm --entrypoint python "$SCRATCHV_W5_IMAGE" -m pip freeze \
  > "$SCRATCHV_W5_OUTPUT/dependencies.txt"
docker run --rm --entrypoint python "$SCRATCHV_W5_IMAGE" \
  -c 'import json; from probes.w1_qwen3_export.run import environment; print(json.dumps(environment(export=True), indent=2))' \
  > "$SCRATCHV_W5_OUTPUT/image-environment.json"
docker run --rm --entrypoint sh "$SCRATCHV_W5_IMAGE" -c \
  'python --version; python -m ziglang version; qemu-system-riscv64 --version; dpkg-query -W python3.12 qemu-system-misc' \
  > "$SCRATCHV_W5_OUTPUT/image-tools.txt"
cat "$SCRATCHV_W5_OUTPUT/image-tools.txt"
```

核对 `image-tools.txt` 与上表；环境差异应保留在报告中，部署脚本本身不锁 apt 补丁。

**3. 独立获取固定资产。**

下载只调用现有 `acquire_model`，沿用 W1 manifest 中已有的下载/分片完整性检查。
它只用 Python 标准库，不执行模型、不重新导出、不在容器外的 Linux 环境安装推理依赖。
Tokenizer 从同一提交的 Git LFS 获取。

```bash
git lfs install --local
git lfs pull --include="assets/qwen3-0.6b/tokenizer/*" --exclude="" \
  2>&1 | tee "$SCRATCHV_W5_OUTPUT/tokenizer-acquisition.log"
python3 - <<'PY' 2>&1 | tee "$SCRATCHV_W5_OUTPUT/model-acquisition.log"
import json, os, shutil
from pathlib import Path
from probes.w1_qwen3_export.run import acquire_model
assets = Path(os.environ["SCRATCHV_W5_ASSETS"])
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
result = acquire_model(assets / "onnx")
shutil.copytree("assets/qwen3-0.6b/tokenizer", assets / "tokenizer")
(out / "acquisition.json").write_text(json.dumps(result, indent=2) + "\n")
PY
test -z "$(git status --porcelain)"
```

不要另跑参考采集脚本或添加 `--record`：suite 内部已经逐步对照当轮 ORT 与冻结参考，
无需再运行一遍完整 ORT 生成作为“预热”。

**4. 资源检查。** 镜像和资产准备后核对可用资源，Docker 配额不能低于要求。

```bash
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
mem = {line.split(":")[0]: int(line.split()[1]) * 1024
       for line in Path("/proc/meminfo").read_text().splitlines()
       if line.startswith(("MemTotal:", "MemAvailable:"))}
evidence = {"stage": "after_image_and_assets",
            "available_cpus": len(os.sched_getaffinity(0)), "memory": mem,
            "disk_free_bytes": shutil.disk_usage(out).free,
            "minimum_available_cpus": 4,
            "minimum_available_memory_bytes": 10 * 1024**3,
            "minimum_disk_free_bytes": 10 * 1024**3}
evidence["passed"] = (evidence["available_cpus"] >= 4
                      and mem.get("MemAvailable", 0) >= 10 * 1024**3
                      and evidence["disk_free_bytes"] >= 10 * 1024**3)
(out / "resource-preflight.json").write_text(json.dumps(evidence, indent=2) + "\n")
if not evidence["passed"]:
    raise SystemExit("Insufficient W5 CPU/RAM/disk; generation NOT RUN.")
PY
```

资源不足时保留记录，不运行完整生成。

**5. 一次运行三条完整用例并保存报告。**

整块复制执行。`--reference-suite` 自带三条用例及每例 8-token 预算，
不要同时传 `--max-new-tokens`，不要传 `--reuse-build`。
代码从固定模型重新构建一次，跨用例自动复用同一 ELF/bundle。

```bash
test "$(git rev-parse HEAD)" = "$W5_COMMIT"
test -z "$(git status --porcelain)"
test ! -e "$SCRATCHV_W5_OUTPUT/run"
date -u +%FT%TZ > "$SCRATCHV_W5_OUTPUT/generation-start-utc.txt"

set +e
timeout --signal=INT --kill-after=30s 280m \
  /usr/bin/time -v -o "$SCRATCHV_W5_OUTPUT/process-time.txt" \
  bash scripts/deploy_w5.sh \
    --model /assets/onnx/model.onnx --tokenizer-dir /assets/tokenizer \
    --reference-suite --verify-onnx /assets/onnx/model.onnx \
    --harts 4 --matmul-policy blocked_fma --matmul-tile 2x4 \
    --matmul-rhs-layout column-panel \
    --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
    --timeout 1800 --total-timeout 16500 \
    2>&1 | tee "$SCRATCHV_W5_OUTPUT/runner.log"
pipeline_status=("${PIPESTATUS[@]}")
printf '%s\n' "${pipeline_status[0]}" > "$SCRATCHV_W5_OUTPUT/runner-exit-code.txt"
printf '%s\n' "${pipeline_status[1]}" > "$SCRATCHV_W5_OUTPUT/log-exit-code.txt"
date -u +%FT%TZ > "$SCRATCHV_W5_OUTPUT/generation-end-utc.txt"

# 此唯一 tag 只供本次使用；不停止其他人的容器。
{
  containers=$(docker ps --quiet --filter "ancestor=$SCRATCHV_W5_IMAGE")
  list_status=$?
  printf 'list_exit_code=%s\n' "$list_status"
  for container in $containers; do
    docker stop --time 30 "$container"
    printf 'stop_exit_code=%s\n' "$?"
  done
  docker ps --filter "ancestor=$SCRATCHV_W5_IMAGE" \
    --format '{{.ID}} {{.Status}}'
  printf 'final_list_exit_code=%s\n' "$?"
} > "$SCRATCHV_W5_OUTPUT/container-cleanup.log" 2>&1

docker image inspect --format '{{.Id}}' "$SCRATCHV_W5_IMAGE" \
  > "$SCRATCHV_W5_OUTPUT/image-id-after.txt"
python3 -B -X utf8 scripts/check_w5_ci.py \
  --report "$SCRATCHV_W5_OUTPUT/run/report.json" \
  --reference probes/w5_generation/references.json \
  --exit-code-file "$SCRATCHV_W5_OUTPUT/runner-exit-code.txt" \
  --output "$SCRATCHV_W5_OUTPUT/result-check.json" \
  > "$SCRATCHV_W5_OUTPUT/result-check.log" 2>&1
printf '%s\n' "$?" > "$SCRATCHV_W5_OUTPUT/check-exit-code.txt"
cat "$SCRATCHV_W5_OUTPUT/result-check.log"
{
  date -u +%FT%TZ
  git rev-parse HEAD
  git status --porcelain
  free -b
  df -B1 .
} > "$SCRATCHV_W5_OUTPUT/final-resources.txt"
set -e
printf 'Preserved output: %s\n' "$SCRATCHV_W5_OUTPUT"
```

这里为失败留证临时关闭 `errexit`，**脚本走到最后不等于 PASS**。
即使 suite 非零退出，仍运行容器清理和已有报告检查器；报告缺失时 checker 应失败，
不能补造成功字段。`result-check.json` 在本地的 `commit` 可为 null；实际 SHA 由
`source.txt` 和最终 Git 记录确认。

若手动 Ctrl-C 使整段 shell 中断，重新进入同一 clone，恢复上面输出目录和镜像变量，
执行本节从“此唯一 tag”注释之后的收尾段，再打包已有报告。不要直接重新跑模型。
主机掉电、daemon 失联或强制终止可能阻止清理；如清理日志有非零码或最后仍列出容器，
标为未完成并回报，不视为通过。

**三条用例与结果判定。** 输入、应得到的文字、token 编号及停止原因统一见
[三条实际输出与冻结参考对照](#fixed-case-results)。复现必须逐项匹配，不能只检查是否出现城市名。

成功要求：三例、24 次前向全部完成，prompt IDs/新 IDs/文本/停止原因匹配；每步全部
`256×151936=38,895,616` 个 logits 有限且严格 `max_abs < 1e-3`；28 层完整图检查、
FMA 12 行 binary32 位模式与 FCSR 检查通过；同一 ELF/bundle 与所选策略匹配、
padding `none`；每个 QEMU 正常退出并回收，没有清理错误。套件、日志管道、报告
检查器退出码均为 0，最终无本次容器残留，源码没有改动。
`container_reproduction=NOT_ASSERTED` 是内部 CLI 不会自行宣称容器复现的字段，
不能手改；容器身份由外层构建、运行和退出记录说明。

**回传报告。** 附 `handoff.txt` 说明执行人、UTC 时间、完整 SHA、独立 clone/构建、
环境差异、三例 IDs/文本/误差、计时与缺失采样、清理结果以及失败阶段，保留：

- 来源、主机资源、镜像/依赖/工具、下载记录、所有退出码与构建/运行/清理日志。
- `run/report.json`、`result-check.json`、`run/executable.json`、`run/build/` 的 JSON。
- `run/steps/step-0000…0023/run.json`、UART 完成帧 `uart.bin`、错误日志及
  `run/fma-conformance/` 的报告。

不传权重、ELF、原始 logits 或 Zig 缓存；失败时也回传已有报告并标明缺失项。
原始产物保留至复现结果确认；重试使用新目录。沿用原小文件筛选方式打包：

```bash
python3 - <<'PY'
import os, tarfile
from pathlib import Path
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
allowed = {".json", ".xml", ".txt", ".log", ".stderr", ".stdout"}
archive = out.parent / (out.name + "-reports.tar.gz")
if archive.exists():
    raise SystemExit(f"Keep earlier evidence; archive already exists: {archive}")
with tarfile.open(archive, "w:gz") as bundle:
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if (path.is_file() and not path.is_symlink()
                and ".zig-cache" not in relative.parts
                and (path.suffix in allowed or path.name == "uart.bin")):
            bundle.add(path, arcname=str(Path(out.name) / relative), recursive=False)
print(archive)
PY
```

`process-time.txt` 的 CPU/RSS 属于 Docker 客户端，不是容器或 QEMU 总量。逐步耗时、
RSS 使用 `run/steps/*/run.json`，partial 采样明确注明；未采集的 CPU 写“未独立采集”，
不填作者历史数据，也不据不同机器的记录计算加速比。

</details>
