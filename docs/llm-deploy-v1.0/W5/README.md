# W5：端到端文本生成

## 改动与功能

W5 把 W4 的“单次模型前向”接成了**逐 token 文本生成**：本机 Python 处理提示词、选择下一个 token 并校验；QEMU 中的 RISC-V 程序执行模型，返回用于选词的分数（logits）。

| 改动 | 实现的功能 |
|---|---|
| 生成循环与 QEMU 适配 | 提示词转为 token 编号 → 模型计算 → greedy 选最高分 token → 加入历史并继续；同一轮生成复用同一 ELF 和权重文件，不逐 token 编译 |
| 命令行与部署 | 提供单条提示词、三条固定参考用例、官方模板多轮文本和容器部署入口 |
| 校验与失败处理 | 逐步对比完整模型输出、选出的 token 和最终文字；支持超时、取消、进程清理及失败后复用；报告或日志写入失败也会报错 |
| 本轮命名与兼容修复 | 统一使用 `--cores` / `cores`；兼容旧 `--harts` / `harts`、`parse_args().harts`、dataclass 导出和更新 |
| 本轮文档与数据整理 | 集中说明实跑结果、计时范围和复现方法，补齐可随仓库查看的[数据摘要](reports/validation-summary.json) |

**默认行为不变：**默认单核，支持手动 `--cores 1/2/4` 和启动时 `--cores auto`，上限 4；padding 关闭。默认算术策略不变，下面完整模型结果使用显式参考配置。

**目前的限制：**没有 KV cache，每生成一个新 token 都重新计算完整模型、启动一次新的 QEMU 并装载权重。完整模型多轮聊天、三例独立容器运行和第二人复现，尚无本文可用的完整实跑记录。

## 实验结果

**完整模型数据来自 2026-10-23 的历史实跑，基线为 `8ad3474`；不是本轮命名、兼容或文档修改后重新测量。**

- **模型与输入：**Qwen3-0.6B，完整 28 层、FP32，batch=1、固定长度 256；原始文本补全，不套聊天模板。
- **运行配置：**QEMU / RV64GC，4 core；`blocked_fma / 2x4 / column-panel`，Mean 为 `two-four-lane-last-axis`，非线性为 `fp32-reference`，padding 为 `none`。
- **生成规模：**每条生成 8 个新 token、执行 8 次完整前向；三条共 24 次，复用同一 ELF 和权重文件。

| 输入提示词 | 实际新生成文字 | 端到端生成与校验耗时（秒） | QEMU 仿真总耗时（秒） | 最大绝对误差 |
|---|---|---:|---:|---:|
| `The capital of France is` | ` Paris. The capital of Italy is Rome` | 3375.024 | 3199.695 | 0.000303983688 |
| `The capital of Germany is` | ` Berlin. The capital of France is Paris` | 3668.852 | 3482.999 | 0.000262737274 |
| `The capital of Japan is` | ` Tokyo. The capital of the United States` | 3489.655 | 3315.397 | 0.000197887421 |

<a id="result-explanations"></a>

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

**结果：3/3 用例、24/24 步通过。** 每步完整 logits 与同模型、同输入的 ONNX Runtime（ORT）结果比较，均满足 `max(|QEMU − ORT|) < 0.001`，且全部数值有限；每步 token ID、最终文字和停止原因也与冻结参考一致。这说明所列用例与参考一致，不代表所有提示词的回答质量。

**停止原因：**三条都是 `max_new_tokens`，即达到 8-token 预算，未遇到 EOS；Japan 的续写尚未成句，不能理解为模型已自然回答完毕。表中生成文字开头均有一个空格。

**计时范围：**两列都是实际经过时间。端到端时间包括最终文字生成、ORT 参考计算和校验，不含前置编译、最终报告保存及终端显示；QEMU 时间是 8 次仿真从启动到退出回收的总和，已包含在端到端时间内，不能相加。

**干净容器单次前向：**本机与容器都生成 ` Paris`，最大绝对误差均为 `0.000172138214`，两边完整输出逐字节相同。本机内部计时 **444.416 秒**且复用构建；容器 **547.454 秒**，其中本次 C 编译与链接 **93.868 秒**。构建条件不同，这不是容器性能对比，也不代表容器三例全部完成。

本轮命名兼容修复另有真实 1/4 core 小图对照，逐步输出与 ORT 完全一致、最大误差为 **0**，取消和超时后的清理与复用通过；完整模型数据仍沿用上面的历史记录。

数据入口：[逐 token 数值与资源摘要](reports/validation-summary.json) · [冻结参考](../../../probes/w5_generation/references.json) · [本轮相关验证记录](../../../tests/llm-deploy/w5/README.md)。计时、误差、逐 token 生成和内存说明见表格下方的折叠项；输入输出细节与历史 CI 记录见[其他](#其他)。

## 如何复现

**下面是在当前代码上运行三条固定用例的最短路径，不是本次已重新执行的记录。** 从仓库根目录执行；使用新输出目录，以免覆盖已有结果。

1. **准备环境。** 按 [Linux 复现指南：W5 环境与固定资产](../LINUX_REPRODUCTION.md#w5-local-environment)完成准备；需要容器或第二人独立复现时，使用同一指南的[容器流程](../LINUX_REPRODUCTION.md#w5-container-reproduction)。已有环境可直接继续。
2. **一次运行三例。** `--reference-suite` 已包含三条提示词及每条 8-token 预算，不再传 `--max-new-tokens`。只构建一次，跨用例复用同一 ELF 和权重文件。历史三例整次测量约 2 小时 56 分钟，实际时间取决于机器与环境。

```bash
python scripts/run_inference.py \
  --model assets/qwen3-0.6b/onnx/model.onnx \
  --tokenizer-dir assets/qwen3-0.6b/tokenizer \
  --reference-suite --verify-onnx assets/qwen3-0.6b/onnx/model.onnx \
  --cores 4 --matmul-policy blocked_fma --matmul-tile 2x4 \
  --matmul-rhs-layout column-panel \
  --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
  --timeout 1800 --total-timeout 16500 \
  --output-dir output/w5-readme-repro-new
```

3. **核对结果。** 查看 `output/w5-readme-repro-new/report.json`：应有三条用例、24 次模型前向，文字与上表一致，逐步完整 logits 有限且最大绝对误差严格小于 0.001；token ID 与参考一致，停止原因均为 `max_new_tokens`，无进程清理错误。只出现城市名不能代替全部检查。构建描述在 `executable.json`，每步记录在 `steps/`。

已有构建可用 `--reuse-build <executable.json>` 替换 `--model ...`，保留原核数和全部算术配置，仍使用新的输出目录。上述完整三例命令不是小图测试，只有确实需要复现完整结果时才执行。

若要**复现表中的历史版本**，使用[独立容器复现流程](../LINUX_REPRODUCTION.md#w5-container-reproduction)中的固定提交 `8ad3474` 和旧 `--harts` 参数；不要把旧版本复现结果当作本轮兼容修复的验证。

## 其他

<details>
<summary>实现细节、接口兼容与补充数据</summary>

<a id="本分支增加了什么"></a>

### 接口与默认配置

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

<a id="整体怎么实现"></a>

### 输入输出与生成流程

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
<summary>多轮对话与失败处理</summary>

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

**容器运行。** Python 与 QEMU 在容器内仍承担相同职责；镜像、资产挂载及运行准备见 [Linux 复现指南](../LINUX_REPRODUCTION.md#w5-container-reproduction)。

部署预检和构建客户端各有独立进程组，SIGINT/SIGTERM 时转发终止并等待清理，必要时
5 秒后强制结束，返回 130/143。容器运行阶段保留 `exec ... run --init` 信号转发，
QEMU 适配器负责其子进程回收。CLI 最终保存报告失败时向 stderr 说明保存错误，
保留已有生成失败或取消的退出码 1/130/143；原本成功或仅构建成功时改为退出 1。
数值、参考文本等原始错误不会被保存错误覆盖，也不打印误导性的成功报告路径。
运行中的 SIGINT/SIGTERM 均记录为取消，分别保留退出码 130/143。

本分支的报告检查器支持 `--log-exit-code-file`：指定后要求生成与日志退出码均为 0，
日志状态缺失、不可读或内容无效也返回 FAIL。外层 Bash 分别保存
`runner-exit-code.txt` 和 `log-exit-code.txt`；生成失败优先保留原退出码，
生成成功而 `tee` 失败时返回日志退出码。环境指南中固定 `8ad3474` 的复现命令保留原版
参数，仍须按其成功要求单独核对日志退出码，不向旧检查器传入新参数。

</details>

<a id="实际复现结果"></a>

### 详细测量与验证记录

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

模型快照与工具版本见 [Linux 复现指南](../LINUX_REPRODUCTION.md#w5-measured-environment)。

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

### 运行示例

<details>
<summary>单条提示词、构建复用与多轮运行示例</summary>

**单条提示词。** 完成 [W5 环境与固定资产准备](../LINUX_REPRODUCTION.md#w5-local-environment)后，从仓库根目录执行：

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

<a id="independent-reproduction"></a>

**容器运行与独立复现。** 固定提交、环境准备、资产获取、三例生成、清理及报告回传步骤集中在 [Linux 复现指南](../LINUX_REPRODUCTION.md#w5-container-reproduction)，按同一 Bash 会话中的顺序执行。

</details>

</details>
