# W5 QEMU 生成入口与续作报告

日期：2026-10-23。分支 `codex/w5-generation`，基于已合并 PR104 的 `61bded3`。
所有变更仍在独立工作树内，未暂存、提交、推送。WSL 配置未修改；W5 CI 已接入既有流程并通过本地相关检查，远端尚未运行。
生成合同及参考用例见 [总览](README.md)。

当前进展：初始旧算术配置的三例首步 FAIL 作为历史记录保留；后续显式 Mean 与非线性
参考策略已完成 Host 与正式 4 hart QEMU **各 24/24 步、3/3 用例通过**，未放宽数值门槛。
对应套件 CLI 的 `status` 和 `full_model_gate` 均为 `PASS`。额外的独立完整模型单次前向
也已通过，完整输出与套件 France 首步逐字节一致；它单独记录，不计入这 24 步验收。
下文旧失败表不代表新策略的最终状态。

## 实现范围

- [命令行入口](../../../scripts/run_inference.py)：从 ONNX 构建一次，或通过 `--reuse-build` 载入经过现有 ELF/权重/布局/生成源码检查的 `executable.json`。
- [QemuForward](../../../scratchv/runtime/qemu_generation.py)：同一 ELF/bundle 跨 token、不同上下文复用；每次只更新输入并运行现有 QEMU stop/dump/quit 流程，完整输出后按有效位置 greedy。不会每 token 重新编译、写权重 bundle、运行 FMA 预检或准备 ORT session。
- [运行器](../../../scratchv/runtime/riscv_external.py)：保留 UART 事件通知、QMP 停止/转储/退出；增加等待期间取消检查，任何异常均经过原有进程清理。50ms 是有取消回调时的最长检查间隔，UART/QMP 数据到达会立即唤醒，不额外给完成计时加 50ms。
- [三轮历史](chat.md)：官方本地模板、显式 non-thinking、成功后提交 user/assistant；错误、取消、超时不污染后续历史。
- [部署文件](deployment.md)：Ubuntu24.04/Python3.12 镜像、固定依赖与薄启动脚本。真实容器小图、QEMU 中断复用、Docker 构建取消及完整 Paris 单 token 均已通过。

保持默认 `1 hart / sequential / 1x4 / original / padding none`。
Mean 默认仍为 `sequential`，非线性策略默认仍为 `libm`。
手动 `--harts 1/2/4` 和 `auto` 均保留，auto 在构建/启动时只解析一次。
复用 ELF 必须显式匹配其 hart、算术策略、tile 和 RHS 布局；不能因请求 auto 或省略参数
悄悄换成另一种编译配置。1 与 4 hart 各有自己的编译产物，每种配置内部复用同一个 ELF/bundle。

## 命令行用法

在已准备 W4 固定环境的仓库根目录使用以下参数。这里以符合 ABI 的小图为例：

```bash
python scripts/run_inference.py \
  --model output/w5-cli-small-20261023/model.onnx \
  --tokenizer-dir output/w5-cli-small-20261023/tokenizer \
  --prompt 1 --max-new-tokens 8 --harts 4 \
  --matmul-policy blocked_fma \
  --verify-onnx output/w5-cli-small-20261023/model.onnx \
  --output-dir output/w5-small-h4 --timeout 30 --total-timeout 120

python scripts/run_inference.py \
  --reuse-build output/w5-small-h4/executable.json \
  --tokenizer-dir output/w5-cli-small-20261023/tokenizer \
  --prompt 2 --max-new-tokens 8 --harts 4 \
  --matmul-policy blocked_fma \
  --verify-onnx output/w5-cli-small-20261023/model.onnx \
  --output-dir output/w5-small-h4-reuse
```

输出目录必须为新目录。小图资产由本轮测试准备，属于本地 `output`，不是预训练模型。
单独准备构建可用 `--build-only`，其状态为 `BUILD_ONLY`，不会伪装成生成 PASS。

`--mode chat --chat-turn 'first question' --chat-turn 'second question' --chat-turn 'third question'`
可在一条命令内进行三轮，另可传 `--system-prompt`；三轮共用同一个 backend/构建。
默认 `--prompt` 是 raw completion，不隐式插入聊天模板。

`--verify-onnx` 在同一个 ORT CPU session 上比较每步相同输入的完整 logits，并要求
最大绝对误差严格小于已有 W4 的 `1e-3`，greedy ID 也必须一致；保留 top-2 margin。
上轮小图全部逐位一致，实际 `max_abs=0`，并非仅靠误差阈值通过。
ORT 是可选诊断 oracle，不参与生成输出的替代计算。QEMU 仍执行全部客体算术。

真实三用例入口为 `--reference-suite --verify-onnx <固定onnx/model.onnx>`，使用冻结的
三个 prompt、预算和参考；同时要求可执行文件的完整图证据、ORT 的完整 28 层结构
以及固定资产接口。缺失图证据或只有小图时拒绝给出完整模型 PASS。W4 旧产物沿用
已有 metadata 身份链接，未新增每 token 的指纹或溯源体系。
完整模型验收示例显式传入 `--mean-policy two-four-lane-last-axis` 和
`--nonlinear-policy fp32-reference`，配合 `blocked_fma / 2x4 / column-panel / padding none`。
`--reference-suite` 不会自动启用这两项；复用该配置的 ELF 时也必须显式匹配。
上轮未执行完整模型；本轮实跑进展见下节。完整命令见 [部署说明](deployment.md)。

## 初始旧算术配置的完整模型 4 hart 验收（2026-10-23，历史结果）

**初始旧配置结果为 FAIL：三条固定用例均在首步完整 logits 数值检查失败，未完成 3×8 token 生成。**
France 首步使正式 `--reference-suite` 按合同退出。随后用同一入口分别运行 Germany、
Japan，并在 output 计量包装中执行相同冻结参考逐步检查；两例也在首步失败。
共完成 **3 次完整前向**，其余 **21 步没有执行**，没有可报告的完整生成文本或停止原因。
没有跳过 padding 行、放宽阈值或将候选 token 一致解释为生成验收通过。

三次均复用同一份既有的完整 Qwen3-0.6B **28 层** ELF 和权重 bundle，显式配置为
`4 hart / blocked_fma / 2x4 / column-panel / padding none`。未重新编译模型，默认单核、
手动/auto 选择、CI、WSL 配置不变。该初始验收阶段只新增 output 诊断产物和更新文档，未改正式实现；后续显式策略实现与验证见下文。
每个命令沿用一次小型 FMA 预检，不计入模型前向；由于失败后独立运行剩余用例，ORT
session 在各命令分别初始化，未声称三个命令共用同一个 session 或驻留 QEMU 进程。

### 三条用例实测结果

检查每步完整 `[1,256,151936]` 的 **38,895,616** 个 FP32 logits：
与同输入的本轮单线程 ORT 比较，要求 `max_abs < 1e-3`、`rtol=0`，greedy ID 必须一致。
冻结 JSON 没有保存全部 logits，因此另核对冻结 ID、有效长度和选中 logit；
最终文本、全部新 IDs 和停止原因属于用例末尾检查，本次因首步失败未到达。

下表均为**首步完整前向**，不是每例完整 8-token 生成的成本：

| 固定用例 | UART 前向耗时 s | QEMU CPU s | QEMU RSS 观测峰值 GiB | 完整 logits max_abs | 数值结果 |
|---|---:|---:|---:|---:|---|
| France | 432.607 | 1492.427 | 5.393 | 0.003272057 | FAIL |
| Germany | 400.287 | 1435.645 | 5.401 | 0.001770496 | FAIL |
| Japan | 399.097 | 1428.588 | 5.399 | 0.001197100 | FAIL |

三例首步候选分别为 **Paris / Berlin / Tokyo**，QEMU、当轮 ORT 和冻结 ID 一致。
QEMU 选中 logit 与冻结值的误差分别为 `1.335144e-5 / 1.716614e-5 / 1.144409e-5`；
当轮 ORT 选中 logit 与冻结值均精确一致。这不改变完整 logits 已失败的结论。

| 用例 | Host 生成调用耗时 s | Host CPU s | QEMU 完整进程耗时 s | QEMU 平均 CPU | Host RSS 峰值 GiB | 同时树 RSS 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| France | 453.789 | 20.722 | 432.969 | 344.7% | 4.068 | 6.699 |
| Germany | 421.424 | 20.245 | 400.650 | 358.3% | 4.100 | 8.372 |
| Japan | 422.524 | 22.998 | 399.460 | 357.6% | 4.099 | 6.399 |

- UART 前向时间从 QEMU 启动到完成帧验证，包含启动和权重装载；完整进程时间另含停止、转储和退出。
- QEMU CPU 为每步 `RUSAGE_CHILDREN` user+system 差值，该窗口唯一子进程为 QEMU；平均 CPU 以完整进程墙钟为分母，100% 表示一个逻辑 CPU。
- Host 生成调用时间另含 ORT、输入/权重传输和数值检查。Host CPU 与 QEMU CPU 分列；采样线程开销计入 Host。
- QEMU RSS 沿用运行器 0.2s 采样，Host/同时进程树 RSS 为 0.5s 采样，均为观测峰值。树 RSS 求和可能重复共享页，不能将独立的 Host/QEMU 峰值相加当作同时峰值。
- 三次命令含准备/FMA 预检分别耗时 458.113s、428.556s、430.072s。全部子进程 CPU 还包含 FMA 编译/运行，不能全称模型 CPU。

三次 QEMU 均 `qemu_reaped=true`，最终检查 PID 和 POSIX 进程组均不存在，成功 transport
副本已清理。数值失败在正常回收之后发生。采样中 perf 与 MONOTONIC_RAW、UTC 与 perf
的最大差均小于 **0.000362s**，未观察到历史跳时。本轮不计算加速比，也不将这些失败首步
外推为完整生成速度。

Germany 的命令准备/Host 阶段与一次 France 离线 ORT 诊断重叠约 10 秒；该诊断在首次
采样到 Germany 模型 QEMU 前约 7.4 秒已结束。没有证据显示模型 QEMU 窗口重叠，
但 Germany 全程数据仍不作为无干扰性能基线。诊断只执行一次，未重跑任何 QEMU 前向。

### France 错误定位：有效行通过，全输出失败

使用保留的 QEMU 输出和一次同输入 ORT 离线重放进行定位。实际 token IDs 和 attention
mask 与冻结合同的 `prepare_inputs` 逐位相同；ORT 仍为 intra/inter 1 线程、
`ORT_DISABLE_ALL`、关闭 CPU memory arena。

| 输出范围 | 最大绝对误差 | 大于等于 1e-3 的元素数 |
|---|---:|---:|
| 有效 query 行 0–4 | 0.000104904 | 0 |
| 用于选 token 的 row 4 | 0.0000572205 | 0 |
| padding query 行 5–255 | 0.003272057 | 20,641 |

超阈值只出现在 padding query 行 **23、101、250**，分别有 555、6、20,080 个元素。
最大坐标 `[0,250,45743]`：QEMU `-5.272073746`，ORT `-5.275345802`，
相对误差约 `0.0620%`。全部 256 行 argmax 一致，选取行 top-2 margin 为 `3.161660194`。
Germany/Japan 本轮未追加逐行定位，不能直接将 France 的行分布推广到另两例。

这里的 padding 指**序列补齐位置**，与本轮保持关闭的 **4KiB 工作区 padding 选项**不是同一件事。
既有 mask 允许这些 query 关注前面的有效 key；不能直接改成全遮挡以规避误差。
目前只确定误差位置，尚未确定首次产生/放大误差的算子，也未证明是否与 4 hart 并行有关。
后续补做的同配置 1/4 hart、同源码 Host 和逐层诊断见下节。

### 产物及发现的记录问题

本地路径相对当前 W5 工作树：

- `output/w5-full-reference-4hart-20261023/summary.json`：三例联合汇总，明确 FAIL、3/24 前向和 21 步未执行。
- `output/w5-full-reference-4hart-20261023/`：France；`output/w5-full-reference-4hart-20261023-germany/`、`output/w5-full-reference-4hart-20261023-japan/`：其余两例。
- 每个目录的 `run/report.json`、`steps.json`、`cases.json`、`numeric-steps.json`：原始 CLI 状态、逐步资源与数值结果；`resources.json`、`resources.jsonl` 为资源汇总/采样。
- 每个目录的 `run/steps/step-0000/output.bin`：完整 QEMU 输出；France 另有 `step-0000-error-localization.json`、`step-0000-ort-logits.npy`。
- `output/run_w5_full_reference.py`、`output/diagnose_w5_step0_error.py`：计量/离线定位工具。没有修改原始失败报告或冻结参考。

首次完整验收确认了两个 CLI 证据记录问题；原始报告保持原样，下节续作已修复这两项：

1. 正式 suite 失败后 `status=FAIL`，但 `full_model_gate` 仍保留初始化的 `NOT_RUN`。联合汇总根据实际前向和数值失败记为 FAIL，不能将该字段误读为本轮没运行完整模型。
2. 从旧 W4 构建载入的完整图审计只用于当次检查，没有带入新保存的 W5 metadata/report。对新副本离线调用 `require_full_executable` 会报缺少完整图证据；原始 W4 metadata/report 仍可复用，本轮三例均使用原始路径，因此该问题没有改变此次计算结果。

## 持续完善与数值定位（2026-10-23）

本轮从北京时间 **2026-10-23 09:35:57**（UTC 01:35:57）开始，持续处理 W5 的剩余问题。
截至最终输出重复性和进程清理核查的 **13:41:27**（UTC 05:41:27），已持续 **4 小时 5 分 30 秒**；
按分钟约为 09:35–13:41、4 小时 6 分钟。下文保留诊断过程和最终验收结果。
冻结合同和 `max_abs < 1e-3` 不变，没有忽略 padding query 行或使用参考输出替代 QEMU。
下列修改均未提交、未推送，CI、WSL、默认单核及手动/auto 选择不变。

### 已修复的边界与验收证据

- 完整生成开始时将 `full_model_gate` 从 `NOT_RUN` 改为 `RUNNING`，失败/取消分别记为
  `FAIL`/`CANCELLED`。准备失败和 build-only 保留 `NOT_RUN`，不会声称运行了模型。
- 通过已有身份检查载入旧 W4 构建后，把其完整图审计保存到 W5 元数据；真实 28 层构建
  连续转换、复用两次成功（均为 build-only，没有新前向）。
- 参考 suite 将冻结历史、实际前向次数、每步完整 logits 对照、独立选择行和冻结 token
  逐步绑定；缺步、额外步、错误历史/长度/mask/padding 或选错行不得只凭最终文本一致通过。
- QEMU 失败时，损坏、非对象、不可读或含非有限数的诊断 JSON 不再覆盖原始超时/取消异常。
  后续 backend 复用和会话历史回滚保持正常。
- 部署脚本对 info/build 客户端按进程组转发取消并回收，避免脚本退出后编译客户端残留。
  守护进程侧的构建取消仍需真实容器引擎验证。

本轮 CLI **66 项**相关单测通过；适配器边界及跨层历史/缓冲区复用 **27 项**检查通过（2 项真实 QEMU 测试未重跑）；
部署脚本 **14 项**检查通过。以上是各自相关验证，不与历史 216 项重复累加。
真实 CLI 另复用既有小图构建，分别完成默认 1 hart、手动 4 hart、auto→4 的两 token
调用：三组均产生 `[2,3]`、按预算停止，6 次前向完整输出与 ORT 逐位一致，全部 QEMU 已回收。
小图仅证明接口/边界，不用于性能加速结论。

产物：`output/w5-followup-20261023/cli-evidence-roundtrip/summary.json`、
`output/w5-followup-20261023/cli-regression/summary.json`。

### 三例错误分布补齐

复用三例原始 QEMU 输出；Germany/Japan 各进行一次相同输入 ORT 离线对照，没有重跑 QEMU。

| 用例 | 有效行最大误差 | 序列补齐行最大误差 | 超阈值 query 行 | 全 256 行 argmax |
|---|---:|---:|---|---|
| France | 0.000104904 | 0.003272057 | 23、101、250 | 全部一致 |
| Germany | 0.000104904 | 0.001770496 | 59、69、121、130 | 全部一致 |
| Japan | 0.000104904 | 0.001197100 | 23、55、58、122、167、177 | 全部一致 |

各例输入 IDs/mask 均与固定合同逐位相同。对应各例原目录下新增
`step-0000-error-localization.json` 和 `step-0000-ort-logits.npy`；首步失败记录没有被覆盖。
有效行通过仍不代表完整 logits 门槛通过。

### 同源码与逐层证据

相同 `2x4 / column-panel` 生成 C、权重和输入在 Host 顺序执行，完整 logits 与 ORT
最大误差为 **0.002274513**，与先前 4 hart QEMU 的最大差为 **0.001323700**。
因此存在独立于多 hart 调度的数值差异；Host 与 RV64 的数学库、目标架构不同，不能
直接把这两个差值归给某一种原因。Host 该次前向 20.819s、CPU 20.814s、RSS 峰值约 3.168GiB，
只作为诊断记录，不替代 QEMU 验收。

在同源码 C 增加 embedding、28 层输出、最终归一化共 30 个观测点，额外缓存 30MiB；
原工作区和 bundle 不变。Host 观测版与其原始 logits 逐位一致；ORT 观测版与已保存的
原始 ORT logits 也逐位一致，排除了这次增加观测点改变最终结果的影响。

embedding 完全一致，第一层补齐行最大误差约 `6.68e-6`。row 250 在第 10–13 层输出的
误差依次为 `0.000189 → 0.000610 → 0.001075 → 0.001394`，提示这一段对上游误差敏感。
第 2 层有较早超过中间诊断阈值的有效行，但对应约 6543 的激活，仅相差少量 FP32 ULP，
不能把“首次过阈值”直接当根因。随后用同一中间输入拆分第 12 层，区分局部计算与累积，结果见下文。

产物：`output/w5-followup-20261023/host-same-source/summary.json`、
`output/w5-followup-20261023/checkpoints/comparison.json`。
同配置 1 hart QEMU 完整前向已结束，与 4 hart 的 **38,895,616 个 FP32 logits 逐位一致**，
相对 ORT 的最大误差同为 `0.003272057`；这个输入上的失败不由并行引入。
单 hart QEMU 进程耗时 **1320.992s**、CPU **1318.380s**、RSS 观测峰值 **5.320GiB**，
平均约 99.8% CPU；进程及进程组已回收。该运行与 Host 诊断编译/执行有重叠，
仅记录实测成本，不用它和历史 4 hart 时间计算严格加速比。
产物：`output/w5-followup-20261023/qemu-1hart/summary.json`。

进一步把同源 C 编译为静态 x86_64-musl，确认无动态装载器，CPU feature 集合与原 Host
编译一致；所得全部 logits 与 QEMU 也逐位一致。其前向 21.622s、CPU 21.497s、RSS 约
2.951GiB，是本机 Host 诊断而非 QEMU 加速后的时间。它与 glibc Host 存在差别，
说明数学库路径值得继续拆分，但不能据此认定某一个 libm 函数已是唯一根因。
产物：`output/w5-followup-20261023/host-musl-same-source/summary.json`。

同一 ORT 第 11 层输出送入第 12 层切片时，切片 ORT 与原整图该层逐位一致，
Host 开/关观测点也逐位一致；该层独立注入的 padding 最大误差仅 `1.72e-5`、row250
仅 `2.15e-6`，远低于整图该行累计 `0.001075`。这表明主要存在上游差异和敏感放大，
不能把第 12 层指认为单一错误算子。后续平方、均值及非线性路径的完整 Host 单变量对照见下文。
产物：`output/w5-followup-20261023/layer12-same-input-v5/report.json`。

### 可选 Mean 策略与多 token 收益边界

已实现 `--mean-policy sequential|two-four-lane-last-axis`，默认仍为 `sequential`。
新策略只改变 FP32 末轴 ReduceMean 的求和顺序，对齐既有 W3 的 `mean_last` 参考；
它不启用 SIMD 指令，也不改变 MatMul、Softmax 或 Pow。构建 evidence、运行报告和
W4/W5 复用检查都携带该字段。旧 ELF 缺字段只能按旧顺序解释，新策略必须重新构建，
不得给旧产物改标签。helper 保留原有 MPL 许可证与第三方声明。

| 受控 France 首步 Host 候选 | 对 ORT 全 logits max_abs | 判断 |
|---|---:|---|
| 原顺序、静态 musl | 0.003272057 | FAIL，且与旧 QEMU 逐位一致 |
| 仅 Pow 精确平方 | 0.003275394 | FAIL；局部平方更准确不保证整图误差减小，未纳入正式修复 |
| 仅末轴 Mean 固定求和顺序 | 0.000959873 | 首步低于阈值，余量较窄 |

Mean 的 4 组真实算子输入和合成边界共 158 个诊断样本与既有 FP32 参考逐位一致。
正式实现新增 20 项 Host 测试并通过原有 5 项相关默认归约回归；真实 1/4 hart 小图
8 项通过，覆盖宽 13/128/1024、不同上下文、符号零、溢出报错后同构建复用。
追加的错误码/进程组断言两项通过，属于这 8 项的复测，不重复累加。
CLI/运行器合同相关三文件合计 271 项通过，替代本轮早先对应子集的计数。

真实新 CLI 对带 ReduceMean 的小图完成默认单核、手动 4、auto→4 的两 token 对照，
完整 logits 逐位一致；从新策略构建复用时若省略策略、退回默认，则在启动任何前向前拒绝。
这些结果只说明新选项与复用边界正确，不等于完整模型已验收。

随后复用同一静态 musl 候选、bundle 和一个 ORT session，按冻结历史进行三例多 token
Host 预检。**3/3 用例仍失败**，共检查 7 步、4 步通过，每例遇到严格数值失败立即停止：

| 用例 | 已通过 step（零基） | 停止 step | 停止步 full max_abs |
|---|---|---:|---:|
| France | 0、1 | 2 | 0.002007961 |
| Germany | 0 | 1 | 0.001288414 |
| Japan | 0 | 1 | 0.001218975 |

7 步中 France 首步复用先前已核对输入的成功输出，新增 6 次 Host 前向；没有重跑 QEMU。
所有已检查步的有效选择行和冻结 ID 一致，但不能绕过完整 logits 门槛继续生成或宣布通过。
因此此 Mean 选项只能作为显式的算术对照策略，单独启用尚未解除 W5 阻塞。
下一步以 France step 2 的同一输入分别对照三角函数、Sigmoid、Softmax，避免将多个改动
同时加入后无法归因。正式默认与 Softmax 行顺序当前都未改变。

资源口径：该多 token 诊断从持有 ORT 的父进程启动 native driver，Linux `ru_maxrss`
可能带入父进程高水位，不能当作 driver 独立峰值；原始 stdout 保留并加注，后续采用
独立轻量父进程或 `/proc` 观测。其 CPU/墙钟和数值判断不受这个 RSS 字段问题影响。

产物均相对当前 W5 工作树：

- `output/w5-followup-20261023/host-mean-candidate/summary.json`
- `output/w5-followup-20261023/host-pow2-candidate/summary.json`
- `output/w5-followup-20261023/mean-host-reference-suite/report.json`
- `output/w5-followup-20261023/mean-rv64-tests/`、`mean-rv64-overflow-final/`
- `output/w5-followup-20261023/mean-cli/summary.json`

### 非线性算子单变量对照（Host 诊断）

Mean-only 在 France step02（7 个有效 token）失败后，使用同一保存输入、原 bundle 和
静态 musl driver，分别只替换 Sin/Cos、Sigmoid 或 Softmax 的参考算术。每个候选的
原语先对照现有 FP32 reference；完整前向没有改变冻结参考或 `max_abs < 1e-3` 门槛。

| 显式诊断候选 | France step02 完整 logits max_abs | 耗时 / CPU 时间 | driver 峰值 RSS |
|---|---:|---:|---:|
| Mean-only（前述基线） | 0.002007961 | 见原逐步记录 | 不使用继承的高水位作独立峰值 |
| Mean + Sin/Cos 参考路径 | 0.000732422 | 21.330 / 21.327 s | 3,258,296 KiB |
| Mean + Sigmoid 参考路径 | 0.000707626 | 19.876 / 19.874 s | 3,258,368 KiB |
| Mean + Softmax 参考路径 | 0.000941038 | 20.056 / 20.048 s | 3,258,232 KiB |

三种单变量候选都让这一 Host 步骤通过，选中 ID 均为冻结参考的 `576`；这不代表
整套生成或 QEMU 已通过。继续运行 Mean + Sin/Cos 三例预检时，France 首步又出现
`max_abs=0.0010201931`，已按原门槛停止并保留失败结果，证明不能从单步外推收益。
上述单变量候选保留在 `output`。后续已将组合路径接为显式选项，见下节；默认实现未改变。参考路径使用标量 C；Softmax
的 16-lane/ldexp 描述固定的 FP32 运算顺序，不要求 RISC-V 目标具有 AVX512 指令。
Sin/Cos 仅在已验证区间使用参考多项式，区间外保留原 libm 回退。

原语和完整单步证据：

- `output/w5-followup-20261023/trig-candidate/primitive-validation.json`
- `output/w5-followup-20261023/host-mean-trig-france-step02/summary.json`
- `output/w5-followup-20261023/sigmoid-diagnostic/micro-summary.json`
- `output/w5-followup-20261023/host-mean-sigmoid-step02/summary.json`
- `output/w5-followup-20261023/softmax-candidate/validation.json`
- `output/w5-followup-20261023/host-mean-softmax-candidate/step02-summary.json`
- `output/w5-followup-20261023/mean-trig-host-reference-suite/report.json`（多 token 收益边界）

### 显式非线性策略与正式入口验证

正式后端新增 `nonlinear_policy="libm"|"fp32-reference"`，W4/W5 CLI 对应
`--nonlinear-policy`。默认仍为 `libm`；参考路径需显式选择，Mean 另由
`--mean-policy two-four-lane-last-axis` 控制。原始 MatMul/Pow/独立 Exp 和默认 hart
均不随选项改变。旧 ELF 缺字段按 `libm` 解释，策略不匹配在运行前拒绝；构建、运行
和复用报告记录实际策略，不增加新的哈希链。许可证与范围见
[运行合同](../W4/runtime-contract.md)。

- 新后端直接检查 **21 passed**，相关既有 Softmax/Mean 检查 **48 passed**；
  六份修改前默认 C 源码快照逐字节相同，包含 inline/external、回调与 Mean 配置组合。
- runtime/W4/W5 三份相关 contract 文件 **328 passed**，覆盖新策略、旧元数据、
  双向复用不匹配、默认/手动/auto 及 Mean 组合；本组包含前述重叠检查，不重复累计。
- 正式生成代码的真实 RV64 混合图 **2 passed**（1/4 hart）：含 Sin/Cos/Sigmoid、
  末轴与非末轴 Softmax、Mean，两个上下文全部与固定参考逐位一致；中途发生有限输入
  差值溢出时，仍传播 guest `status 2` 并回收进程，原 ELF/bundle 随后可正常复用。
- 真实 CLI 的默认 1 hart、手动 4 hart、auto→4 hart 各完成两步 `[2,3]`，
  共 6 次前向全部与 ORT 逐位一致，QEMU 进程及进程组均已消失。
- 此前 output-only Softmax 诊断另验证窄/宽行、非末轴、回调上下文与工作区恢复；
  注入零和/NaN 归一化错误后调用方输出不污染。它不计为完整模型验收。

正式验证产物为 `output/w5-followup-20261023/backend-nonlinear-validation.json`、
`nonlinear-rv64-tests/` 和 `nonlinear-cli/summary.json`。

组合 Host 三例预检已完成 **24/24 步、3/3 用例通过**；其中 France 首步仅按同一
驱动及逐字节相同输入复用一次，其余 23 次新执行。每步检查完整 logits、所选行及
冻结 ID，三条最终文本和停止原因完全一致，未改变误差门槛或过滤 padding 行。

| Host 诊断用例 | 全 logits 最大误差 | 有效行最大误差 | 8 次前向耗时 / CPU 时间 |
|---|---:|---:|---:|
| France | 0.000303984 | 0.000008583 | 158.853 / 158.753 s |
| Germany | 0.000262737 | 0.000020981 | 159.154 / 159.040 s |
| Japan | 0.000197887 | 0.000012875 | 158.336 / 158.239 s |

上表 France 含已完成首步的原始测量。预检这次实际总耗时 577.597 s（包含 ORT 与
对照，且不重复执行已复用首步），不能与上表求和混为同一计时口径。独立驱动通过
`/proc` 观测的峰值 RSS 最大 3,258,564 KiB，ORT 父进程累计峰值 5,290,672 KiB。
原 `getrusage` 子进程高水位继承父进程值的问题已明确标注，不用它冒充独立 driver 峰值。

证据为 `output/w5-followup-20261023/mean-trig-sigmoid-softmax-host-reference-suite/report.json`。
上述数据属于完整 Host 数值预检，**不能替代 QEMU 验收**。后续使用正式 codegen 的新
4 hart ELF、复用原 bundle 完成了三例 24 步 QEMU 验证；其独立实测结果见下节。

### 正式组合策略 QEMU 完整生成（3/3 用例、24/24 步通过）

正式 codegen 已构建新的完整 Qwen3-0.6B **28 层** ELF；三条冻结用例复用同一
ELF 和原权重 bundle，逐 token 只更新输入并启动 QEMU。显式配置为
`4 hart / blocked_fma / 2x4 / column-panel / mean two-four-lane-last-axis / nonlinear fp32-reference / padding none`。
没有将 Host 输出替代 QEMU 计算，也没有修改默认单核、手动/auto hart 选择或冻结参考。
24 份运行记录中的既有 ELF/bundle 身份字段完全一致，没有新增哈希检查。

三例各完整生成 8 token，**3/3 用例、24/24 步数值及冻结参考检查通过**，命令退出码为
`0`，正式 CLI 的 `status=PASS / full_model_gate=PASS / qemu_forwards=24`。
每步都比较完整 `[1,256,151936]` 的 **38,895,616** 个 FP32 logits，严格要求
`max_abs < 1e-3`、`rtol=0`，没有跳过序列补齐行。全套最大误差为
`0.000303983688`；这表示容差一致，不表示全部 logits 逐位相等。

| 固定用例 | 已完成步数 | 生成调用墙钟 s | QEMU CPU s | Host CPU s | 全 logits 最大误差 | 当前状态 |
|---|---:|---:|---:|---:|---:|---|
| France | 8/8 | 3375.024 | 11517.789 | 167.657 | 0.000303983688 | 整例通过 |
| Germany | 8/8 | 3668.852 | 12574.174 | 190.673 | 0.000262737274 | 整例通过 |
| Japan | 8/8 | 3489.655 | 11958.881 | 178.759 | 0.000197887421 | 整例通过 |

三例新 token IDs、完整文本和停止原因均与冻结参考一致，停止原因均为 `max_new_tokens`：

| 用例 | 新 token IDs | 生成文本 |
|---|---|---|
| France | `[12095, 13, 576, 6722, 315, 15344, 374, 21718]` | ` Paris. The capital of Italy is Rome` |
| Germany | `[19846, 13, 576, 6722, 315, 9625, 374, 12095]` | ` Berlin. The capital of France is Paris` |
| Japan | `[26194, 13, 576, 6722, 315, 279, 3639, 4180]` | ` Tokyo. The capital of the United States` |

各例最大完整 logits 误差分别位于本例 step 3、3、1；冻结所选 token 的 logit 最大误差
分别为 `1.907349e-6 / 5.722046e-6 / 5.722046e-6`。全部 24 个不同 QEMU 进程
均记录 `qemu_reaped=true`；套件结束后只读核查这 24 个 PID 和 POSIX 进程组均已消失。

| 每例 8 步资源口径 | France | Germany | Japan |
|---|---:|---:|---:|
| UART 前向耗时合计 / 每步均值 | 3197.261 s / 399.658 s | 3480.225 s / 435.028 s | 3312.868 s / 414.109 s |
| 单步 UART 前向耗时范围 | 395.824–407.598 s | 423.936–453.608 s | 406.769–428.952 s |
| QEMU 完整进程耗时合计 | 3199.695 s | 3482.999 s | 3315.397 s |
| QEMU 平均 CPU 占用（100% 为一个逻辑 CPU） | 359.97% | 361.02% | 360.71% |
| QEMU RSS 观测峰值 | 5.406 GiB | 5.406 GiB | 5.406 GiB |
| Host RSS 观测峰值 | 4.254 GiB | 4.102 GiB | 4.098 GiB |
| 同时进程树 RSS 观测峰值 | 7.565 GiB | 7.644 GiB | 7.411 GiB |

整次测量命令墙钟 **10540.190s（约 2 小时 55 分 40 秒）**，包含准备、ORT 与核验；
24 步 UART 前向合计 **9990.355s**、平均 **416.265s/token**，完整 QEMU 进程合计
**9998.091s**。模型 QEMU CPU 合计 **36050.844s**，平均 CPU 占用 **360.58%**；
Host CPU 全程 **540.717s**，其中三例生成阶段合计 **537.089s**。
所有子进程 CPU 全程为 **36050.974s**，另含准备/FMA 检查，不能全部算作模型 CPU。

计量口径沿用前文：UART 时间包括 QEMU 启动和权重装载，完整进程时间另含
stop/dump/quit；平均 CPU 用 QEMU CPU 总时间除以完整进程耗时之和。
生成调用墙钟和此表 Host CPU 包含本例 ORT 与数值核验，逐步 adapter 的 Host CPU
仅覆盖 adapter 调用，两者不能混用。QEMU RSS 为 0.2s 采样，Host/同时进程树为
0.5s 采样；全程 Host 与同时树峰值分别为 **4.254 / 7.644 GiB**。
不能将逐步峰值相加，也不能将独立 Host/QEMU 峰值相加当作同时峰值；树 RSS 可能重复共享页。
Host/同时进程树采样器全程 21,032 次采样、0 次错误。独立的 QEMU RSS 采样中，
24 步有 **17 步为 `partial`，各有 1 次读取错误**；5.406 GiB 仍只是观测峰值，
不能据此声称 QEMU 内存采样完全无缺失。`perf−MONOTONIC_RAW` 与 `UTC−perf` 的
最大绝对漂移分别为 **0.000773s / 0.000504s**，未观察到历史跳时。
本次不使用旧计时异常或不同算术策略的记录计算加速比。

本地产物路径相对当前 W5 工作树：

- `output/w5-followup-20261023/formal-reference-4hart/executable.json`：本次正式 ELF 的复用入口。
- `output/w5-followup-20261023/formal-reference-4hart/report.json`：构建与原 bundle/布局匹配记录；该构建报告本身不代表生成验收通过。
- `../w4-harts-short-20261021/optimization-completion-20261022/integration-full-4hart/run/weights/`：继续复用的原权重 bundle。
- `output/w5-followup-20261023/qemu-reference-4hart/metadata.json`：实际命令、环境与测量配置。
- 同目录 `cases.json`、`steps.json`、`numeric-steps.json`：三例结果、24 步资源及完整 logits/冻结参考对照。
- 同目录 `resources.jsonl` 和 `resources.json`：全程资源采样与最终汇总。该次命令使用原计量包装，不要求后来单次前向包装新增的 `measurement_passed` 字段；终态以 CLI、3 例输出和 24 步数值证据共同确认。
- 同目录 `run/report.json` 及 `run/steps/step-0000/run.json` 至 `step-0023/run.json`：CLI 最终状态与逐步运行证据。

### 独立完整模型单次前向（已通过）

三例通过后，另用相同正式 4 hart ELF/bundle、France 固定 prompt 完成独立的 **1 token /
1 次完整前向**计量，沿用相同算术策略和 `padding none`。命令退出码 `0`，生成
`[12095]`、文本 ` Paris`，按 `max_new_tokens` 停止，与冻结首 token、文本前缀及预算停止规则一致。
选中 logit 为 `17.498851776123047`，与当轮 ORT 和冻结值完全一致；完整
**38,895,616** 个 logits 对 ORT 的 `max_abs=0.000172138214 < 0.001`。

| 单次完整前向指标 | 实测 |
|---|---:|
| UART 完成耗时 | 418.954 s，约 6 分 59 秒 |
| QEMU 完整进程耗时 | 419.383 s |
| QEMU CPU 时间 / 平均 CPU 占用 | 1498.851 s / 357.39% |
| 生成调用墙钟 / Host CPU | 441.033 s / 21.604 s |
| 整次测量墙钟 / Host CPU（含准备） | 448.263 s / 25.332 s |
| QEMU RSS 观测峰值 | 5.398 GiB |
| Host RSS 观测峰值 | 4.119 GiB |
| 同时进程树 RSS 观测峰值 | 7.210 GiB |

本次启动新 QEMU 进程，但**没有清空系统文件缓存，因此不称为冷启动测量**。
完整 **155,582,464 字节**输出与先前三例套件 France 首步逐字节一致。
已有 ELF/bundle 身份也相同，没有新增哈希；最终确认套件与本次共 **25 个 QEMU PID
及其进程组全部消失**。单次采样 0 次错误，`perf−MONOTONIC_RAW` 与 `UTC−perf`
最大绝对漂移为 **0.000160s / 0.000144s**。

此命令范围为 `full-model single forward`，没有再次执行三例参考 suite，故正式 CLI
`full_model_gate=NOT_RUN` 是正常范围标记，**不是没有运行完整模型**。
计量结果 `measurement_passed=true`，`full_reference_suite_evaluated=false`；
三例 gate 的 `PASS` 来自前述独立的 24 步验收，不能把这一次重复前向计为第 25 个 suite 步骤。

本地产物目录：`output/w5-followup-20261023/qemu-final-single-forward-4hart/`。
其中 `steps.json`、`cases.json`、`numeric-steps.json`、`resources.json` 分别记录单步、
生成调用、数值和资源结果；`run/report.json` 保留正式 CLI 范围。
`repeat-and-cleanup-check.json` 记录同 ELF/bundle、完整输出逐字节一致和 25 个进程/进程组清理核查。

### 真实容器部署续验（本地通过）

本轮已用 Docker Engine 29.9.2 / Buildx 0.38.1 真正构建并启动干净镜像。
Docker Hub 认证 TLS EOF 后显式选择 Canonical 官方 ECR Ubuntu 24.04，默认基础镜像
仍为 `ubuntu:24.04`。引擎仅手动启动，验证后 Docker 服务、socket 和 containerd
均为 **inactive + disabled**；停止前无运行容器，临时权限已恢复，WSL 配置未变。
工具版本、完整命令与证据见 [部署记录](deployment.md)。

默认 1 hart、手动 4 hart、`auto→4` 的容器 CLI 小图各 4 次前向均通过，
IDs `[2,3,4,6]`、EOS 停止，完整 logits 与 ORT 逐位一致，padding `none`。
已有真实 QEMU 超时/取消及失败后复用测试在镜像内 **2/2 通过、0 skip**，
1/4 hart 共 16 次 guest 启动均完成进程/进程组回收。实际 Docker 构建
SIGINT/SIGTERM 另 **2/2 通过**，客户端组及引擎侧 RUN 进程均消失，延后复查无残留。

本轮只追加一次容器内完整 28 层 France 单 token 前向，现已 **PASS、退出码 0**：
输出 `[12095]`、文本 ` Paris`、预算停止，完整 **38,895,616** 个 logits 的
`max_abs=0.000172138214 < 0.001`；图审计确认 28 层，FMA 与 QEMU 回收通过。
容器从只读模型资产独立重新构建，CLI 总耗时 **547.454s**（含构建/FMA/ORT/核验），
UART 前向 **415.058s**、QEMU 完整进程 **415.478s**、运行器记录耗时 **430.481s**，
后者含权重传输、QEMU、后检查与清理，不含计时起点之前的输入打包/写入及 ELF/bundle 校验。
QEMU RSS 观测峰值 **5.398GiB**。RSS 有 2071 个有效样本，**1 次读取失败**，
可用性为 `partial`；本次未独立测量 CPU 总时间，不借用宿主或 Docker 客户端数据。
容器输出 **155,582,464 字节**与既有宿主独立完整前向直接比较逐字节一致，未新增哈希。

内部 CLI 的 `full_model_gate=NOT_RUN / container_reproduction=NOT_ASSERTED` 保留：
单 prompt 不重跑三例 gate，实际容器身份由外层 Docker 构建/运行和退出证据确认。
它与既有宿主三例 24 步及宿主独立前向分别记录，构成 **deploy:script 本地真实容器 PASS**，
不声称容器三例通过。**W5 本地功能出口已满足**；容器收尾时 CI 接入的 753 项轻量
检查通过，14 个新增必跑节点仅收集核对、未重跑；已有真实 24 步报告的严格检查通过。
后续提交前检查与修复见 [收尾记录](README.md#提交前检查与收尾2026-10-23)。
模式和资源门槛见 [CI 接入说明](README.md#ci-接入与远端执行边界)；
未提交/推送，远端 **NOT_RUN**。

证据相对仓库根：`output/w5-container-20261023/small-summary.json`、
`small-runs/real-qemu-tests.xml`（同目录）、`build-cancel/report.json`、
`build-cancel/delayed-recheck.json` 及 `paris.log`、`paris-summary.json`、`paris/run/report.json`、`services-final.json`。

## 上轮小图数值与清理验证

适配器使用真实 Gather + MatMul 小图：输入 `[1,256]` token IDs，输出完整 `[1,256,8]`
FP32 logits，随历史输入改变。合成图声明完整 mask ABI 但不消费 mask；mask 的精确
causal/key-padding 语义由已有 P1 测试验证，不能将该小图称为 Transformer 语言验收。

| 对照 | 1 hart | 4 hart |
|---|---|---|
| 同一 ELF/bundle 的多 token | `[2,3,4,6]`，EOS 停止 | 相同 |
| 独立上下文与预算停止 | `[4]`，`max_new_tokens` | 相同 |
| 每次成功前向的完整输出 | 2048 元素与 ORT 逐位相等 | 相同 |
| 正在运行的 WFI 测试 guest 超时 | 抛 TimeoutError；进程和进程组已回收 | 相同 |
| 正在运行时取消 | 抛 GenerationCancelledError；进程和进程组已回收 | 相同 |
| 中断后复用 backend/ELF | 再生成 `[6]`，EOS | 相同 |
| 独立真实启动数 | 8 次：6 成功、1 超时、1 取消 | 相同 |

这组 16 次真实启动全部核对 PID 与 POSIX 进程组消失，无清理错误。
失败运行按原有策略各保留 8192 字节 transport 诊断副本；这是故意保留的文件，不是
仍在运行的编译/QEMU 进程。成功运行的 transport 副本已清理。

另外从真实 CLI（本地小 tokenizer，并非 Python mock）完成四次命令：

| CLI 调用 | 新 token IDs | 前向次数 | 完整输出 max_abs | 结果 |
|---|---|---:|---:|---|
| 1 hart 首次构建，prompt `1` | `[2,3,4,6]` | 4 | 0 | PASS |
| 4 hart 首次构建，prompt `1` | `[2,3,4,6]` | 4 | 0 | PASS |
| 1 hart 复用构建，prompt `2` | `[3,4,6]` | 3 | 0 | PASS |
| 4 hart 复用构建，prompt `2` | `[3,4,6]` | 3 | 0 | PASS |

表中引用最终入口的复测产物；早期小图记录保留，不与最终矩阵重复累计。
跨命令的 ELF 和 bundle 路径确实相同，所有 QEMU 均回收。复用前沿用已有
report 到 executable metadata 的身份校验，缺失或漂移先于加载/编译失败。每条生成命令仅在准备阶段
做一次独立的小型 FMA 预检；它不算模型 token 前向。默认配置另做真实小图 build-only，
未启动 QEMU，记录默认 1 hart、sequential 和 padding none。

这些约 20ms 的 guest 是正确性冒烟。未采集 CPU 总时间，0.2s 的 RSS 周期只捕捉到
启动样本，不能据此宣称加速或峰值内存改善；未使用历史异常计时或完整模型速度作推算。

## 上轮相关检查与产物

| 检查 | 通过数量 |
|---|---:|
| 新 QEMU 适配/取消/QMP 边界与真实 1/4 hart 小图 | 18 |
| 既有 external runtime 与 W4 契约相关回归 | 112 |
| 三轮历史及官方模板文本/token IDs 对照 | 40 |
| CLI 参数、复用、数值、错误/信号与完整图门槛 | 37 |
| 部署脚本/打包/失败路径隔离检查 | 9 |

上轮共 **216 项相关检查通过**；与 P0/P1 的旧记录分开列出，不将重叠项重复累计。
另完成 12 个 Python 文件语法检查、部署脚本 Bash 语法检查、6 份相关文档的
53 个本地链接检查（含 33 个 Markdown/HTML 链接），未引入重复文档名称；
最终四份 CLI 报告与汇总、构建复用路径及进程回收记录离线核对一致。
上轮未运行完整模型；初始完整验收阶段仅进行上述完整验收和必要离线定位，未重跑这组 216 项单测或全量回归。后续显式策略的相关检查另列于持续完善部分；未追加 AI 审查。CI/WSL 未改，代码未提交推送。

本地诊断路径（相对当前 W5 工作树）：

- `output/w5-qemu-generation-summary.json`：适配器 1/4 hart 汇总。
- `output/w5-qemu-generation-real/`：真实逐步输出、超时/取消和回收证据。
- `output/w5-cli-final-20261023/summary.json`：四次 CLI 构建/复用对照。
- `output/w5-cli-final-20261023/{hart1,hart4,hart1-reuse,hart4-reuse}/report.json`：逐 token ORT/QEMU 记录。
- `output/w5-cli-small-20261023/default-build-only/report.json`：默认配置构建检查。

## 剩余问题和明确未执行项

1. **默认算术配置的数值边界仍需保留**：显式组合策略已经完成 Host 和 QEMU 各三例 24/24 步验收；初始 `mean sequential / nonlinear libm` 配置的三例首步仍有完整 logits 超过 `1e-3` 的历史失败，不能把显式策略 PASS 推广为默认路径 PASS。默认值未改变，完整验收需显式选择本文配置。独立单次前向也已通过，并已单独记录，不能与 24 步套件混算。
2. **本地功能与远端门槛分开**：真实容器镜像、小图、中断清理及唯一追加的完整 France 单 token 已通过，W5 本地功能出口已满足；没有重复容器三例套件。引擎及 socket 已停止并禁用自启，临时权限已恢复，WSL 配置不变。
3. **CI 本地实现与检查通过，远端 NOT_RUN**：提交前补齐现有 llm-deploy 的 W5-only 手动入口，811 项相关检查及最终 11 项重叠边界复查通过；保留 12 个真实 QEMU 小图、2 个官方模板及完整生成门槛。14 个真实节点本轮没有重新运行；远端资源和时限仍待授权提交/推送后验证，不把本地通过写成远端 Actions 已通过。详见 [提交前收尾](README.md#提交前检查与收尾2026-10-23)。
4. **同步边界**：QEMU UART/QMP 等待可响应取消并回收进程；构建/传输/可选 ORT 校验等同步工作仍在各自阶段边界检查总预算，Host deadline 不承诺硬中断任意 ORT 调用。SIGINT/SIGTERM 会经已有编译/QEMU清理路径退出，不返回成功的部分生成。

固定 256 长度、无 KV cache 意味着每个新 token 重算完整前向；当前实现复用构建，
不宣称 QEMU 进程或权重驻留。持久 guest、KV/RVV/量化等属于后续优化，不是本轮漏做的生成链路。
