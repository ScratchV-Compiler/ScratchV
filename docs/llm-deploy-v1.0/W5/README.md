# W5：Host 生成合同与端到端验收

合同版本：`w5-generation-v1`，冻结日期：2026-10-23。W5 从 PR104 已合入的
`main`（`61bded377097900f4c61323269fd6aed41f333d5`）开始，独立分支为
`codex/w5-generation`。阶段目标和责任分工沿用[开发计划](../开发计划.md)。

W5 的生成合同、Host 循环、QEMU 适配/命令行、三用例验收入口、三轮历史和容器部署文件
现已实现。P0/P1 的冻结 ORT 参考与上轮 1/4 hart 小图记录保留。初始旧算术配置的
4 hart 完整模型三例均在首步完整 logits 检查失败，这些历史结果仍完整保留。
后续显式启用 Mean 与非线性参考策略后，Host 三例预检与正式 4 hart QEMU 套件均已
**24/24 步、3/3 用例通过**。QEMU 每步完整 logits、冻结 token IDs、最终文本和停止
原因均通过，全部步骤最大绝对误差为 **0.000303983688**，保持严格 `<1e-3` 门槛。
通过结果限定于下述显式配置，未将默认算术路径标为通过，详见
[完整模型实跑报告](qemu-generation.md)。本轮干净容器的默认 1 hart、手动 4 hart、
auto 小图和真实超时/取消、Docker 构建取消检查已通过；完整 28 层 France 单 token
容器部署也已通过，实际输出 ` Paris`，完整 logits 最大误差为 **0.000172138214**。
**W5 本地功能出口已满足**；W5 CI 已接入既有流程并通过本地相关检查，远端运行 **NOT_RUN**；
WSL 配置未改，所有 W5 改动仍未提交、未推送。

使用入口：[QEMU 生成与本轮报告](qemu-generation.md)、[三轮历史](chat.md)、
[干净容器部署](deployment.md)。

## 固定生成合同

| 项目 | v1 约定 |
|---|---|
| 模型 | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| 资产身份 | 沿用 [W1 manifest](../../../probes/w1_qwen3_export/manifest.json) 中固定 ONNX、权重和 tokenizer；不另建校验体系 |
| 计算范围 | FP32、batch=1、固定序列容量 256、无 KV cache；每个新 token 重新执行完整前向 |
| 输入模式 | raw completion：使用 prompt 原文，不插入 BOS/EOS、系统提示或 chat 模板；tokenizer 按既有 NFC 规范化 |
| 输入张量 | INT64 `input_ids[1,256]`，右侧使用固定 `pad_token_id` 补齐；FP32 `attention_mask[1,1,256,256]`，沿用 W2 causal/key-padding 语义 |
| 有效长度 | 由实际 token 数确定；prompt 中出现 pad/EOS ID 不改变长度，不据此停止 |
| 前向输出 | 完整 FP32 `logits[1,256,151936]`；包括未使用 padding 行在内，所有值都必须有限 |
| 选择规则 | 取 `valid_length - 1` 行，直接 greedy argmax；平局选较小 ID，不额外计算 Softmax |
| 可解码范围 | 选中 ID 必须属于固定 tokenizer 的真实 ID 集合；不能仅以小于模型词表宽度为准 |
| 追加规则 | 每次前向最多追加一个合法 token；不可解码 ID 报错，不屏蔽、不替换、不重采样 |
| 停止顺序 | 新生成 EOS → 总长度达到 256 → 已生成数量达到 `max_new_tokens` |
| EOS | 使用固定 tokenizer 的 `generation_eos_token_ids`：`151645` 和 `151643`；只有新生成 token 触发 EOS |
| 输出文本 | 仅解码新生成 IDs，`skip_special_tokens=True`、`clean_up_tokenization_spaces=False`；完整新生成 IDs 是验证依据，EOS ID 保留在其中 |
| 返回数据 | prompt IDs、新生成 IDs、生成文本和停止原因组成不可变结果；无隐式会话历史 |

生成核心复用 [W2 输入与 greedy 辅助函数](../../../scratchv/runtime/llm_inputs.py)
和[本地 tokenizer](../../../scratchv/runtime/qwen3_tokenizer.py)。模型词表为 151936，
固定 tokenizer 实际包含 151669 个 ID；这两者不能混用。主循环不调整数值误差门槛，
也不会只检查用于 argmax 的那一行而漏掉其他输出中的 NaN/Inf。

`max_new_tokens` 是上限，不预留或扩展容量。prompt 非空且不超过 256 个 token；
空 prompt、超长 prompt、非法 token、负数/布尔预算明确报错，不静默截断。
0 预算不会调用前向；初始长度已为 256 时也不调用前向，按容量原因返回。
若一次追加同时触发 EOS、容量和预算，按上表顺序报告原因。生成 EOS 仍计入预算。

## Host 调用及失败语义

[Host 核心](../../../scratchv/runtime/generation.py) 的公开入口为：

```python
generate(prompt, *, tokenizer, forward, max_new_tokens,
         cancelled=None, timeout_seconds=None) -> GenerationResult
```

它接受一个 `PreparedInputs -> np.ndarray` 的同步 `forward` 回调。
回调可接 ORT 或已实现的 `QemuForward` 适配器；核心不编译模型、不管理 QEMU，也不为每个
新 token 重建 tokenizer。每一步从本次生成的 token 历史重新准备输入，复用后端输出
缓冲区不会改变已经选出的 token；不同调用不共享生成状态。

失败行为属于合同：

- 后端原始异常向调用方传播；非法形状、dtype、非有限输出及不可解码 ID 都使本次生成失败。
- 取消抛 `GenerationCancelledError`（`RuntimeError` 子类），超时抛 `TimeoutError`，不返回看似成功的部分结果。下一次调用从新的 prompt 开始，不能继承失败调用的 token。
- `cancelled` 是可选取消检查回调；检查发生在同步前向前后及返回前。它不强制中断正在阻塞的前向。
- `timeout_seconds` 是可选有限非负秒数，布尔值不合法；0 表示立即超时，即使 token 预算为 0。
- 超时从输入与配置验证完成后开始，以单调时钟作协作式检查；取消与超时同时触发时先报告取消。正在运行的同步后端由后端自身负责限时、取消和进程清理；Host 合同不声称能终止任意阻塞回调。

本轮不会为每个 token 新建线程来模拟可中断执行，也不会吞掉失败后继续选 token。
QEMU 适配已复用现有运行器，在 UART/QMP 等待中响应取消，并验证超时/取消后的进程及进程组回收。失败运行按既有策略保留 transport 诊断副本。

## 3 条固定参考用例

以下三条原文和预算在 ORT 执行前选定；Paris 原文与 W5 硬门槛保持一致。

| 用例 | prompt 原文 | `max_new_tokens` | 预先约定的内容检查 |
|---|---|---:|---|
| `france` | `The capital of France is` | 8 | 新生成文本包含 `Paris` |
| `germany` | `The capital of Germany is` | 8 | 新生成文本包含 `Berlin` |
| `japan` | `The capital of Japan is` | 8 | 新生成文本包含 `Tokyo` |

参考必须记录 prompt IDs、新生成 IDs、文本、EOS/停止原因和实际 ORT 版本；
不能只填写上述预期词。未来 QEMU 对照优先比较逐 token ID；若出现分歧，保留相同
输入的数值误差和 top-1/top-2 margin，定位原因，不用“语义合理”覆盖数值失败。
原始全量 logits 是运行中验证对象，无需将每一步约 148 MiB 输出都加入 Git。

[固定参考文件](../../../probes/w5_generation/references.json)保存三条输入及冻结后的
精确输出，检查入口为 [run.py](../../../probes/w5_generation/run.py)。在仓库根目录，
激活已按 [Linux 复现说明](../LINUX_REPRODUCTION.md)准备的环境后执行：

```bash
python probes/w5_generation/run.py \
  --assets assets/qwen3-0.6b \
  --output-dir output/w5-generation-reference
```

这条命令使用固定资产和同一 ORT CPU session 验证三条参考，并在输出目录保存
`reference-run.json`；不会启动 QEMU。默认检查已冻结参考，不自动覆盖它。
初次采集需要显式 `--record`，并按预先固定的内容标准确认结果后才冻结；不能为消除
失败任意重录参考。QEMU 用户入口已实现为 `scripts/run_inference.py`；与本节 ORT 参考工具用途不同，见 [QEMU 说明](qemu-generation.md)。
2026-10-23 使用同一个 ORT CPU session 实跑，三条均完成 8 个 token，停止原因为
`max_new_tokens`。独立逐步输入/greedy/停止 oracle 与 Host 核心一致，三个预先固定的
内容检查均通过，精确 token IDs 已冻结到上述 JSON。

| 用例 | 实际生成的新文本（保留前导空格） | 耗时 | CPU 时间 |
|---|---|---:|---:|
| `france` | ` Paris. The capital of Italy is Rome` | 44.672 s | 44.649 s |
| `germany` | ` Berlin. The capital of France is Paris` | 42.910 s | 42.894 s |
| `japan` | ` Tokyo. The capital of the United States` | 42.406 s | 42.394 s |

全轮耗时 **130.676 s**，CPU 时间 **130.611 s**；进程峰值 RSS **5.002 GiB**
（`ru_maxrss` 为整个进程累计峰值，不能当作每例独立分配或增量）。模型 session
加载耗时 0.307 s；这些是 ORT 参考实测，不是 QEMU 生成速度。

P0/P1 阶段只执行一次真实 ORT 采集并在每步使用独立 oracle 验证，随后冻结参考，完成离线
字段一致性检查；未重复执行“冻结后第二次 ORT 回放”。精简公开记录、逐 token
margin 与环境位于 JSON 的 `baseline` / `cases`，详细本地产物为
`output/w5-generation-reference/reference-run.json`。原始续写可以继续列举其他事实，
并因 8-token 预算停止；这三条只验证约定的 raw completion 内容，不宣称 chat 或多轮能力。

## 保留的执行配置

既有 RV64 路径保持默认 **1 hart**，显式接受 **1 / 2 / 4 / auto**。
`auto` 只在构建/启动时按进程可用 CPU 解析一次，上限为 4，不在 token 之间动态改数。
复用 ELF 的 hart 数必须与构建记录一致。Host 生成核心不覆盖这套配置，也不会把
ORT 的 CPU 线程数误当成 guest hart 数。

QEMU 工作区布局保持 `qemu_workspace_layout=none`，即 padding 关闭。
均值默认仍为 `sequential`；显式 `--mean-policy two-four-lane-last-axis`
可用于数值对照，但单独启用尚不能让完整三例通过，见续作报告。
非线性路径默认仍为 `libm`；显式 `--nonlinear-policy fp32-reference` 选择已有参考的
Sin/Cos、Sigmoid、Softmax 算术顺序，不改变 hart 选择，也不要求目标具有 x86 SIMD。
本轮完整 QEMU 三例通过的配置为 `4 hart / blocked_fma / 2x4 / column-panel`，同时
显式启用 `--mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference`，
padding 为 `none`；24 步复用同一 ELF/bundle。默认 `1 hart / sequential / 1x4 / original`、
Mean `sequential` 和非线性 `libm` 均未改变，不能把新策略的验收结果归给旧算术配置。
复用时必须显式匹配构建策略；实验 padding、RVV、KV cache、动态长度和量化不纳入本轮。

## 阶段与验收状态

| 阶段/门槛 | 本轮状态与后续要求 |
|---|---|
| P0：合同与 3 条参考 | **PASS（本地）**；合同冻结，3/3 真实 ORT/Host 参考与独立 oracle 通过，参考 IDs/text/停止原因冻结 |
| P1：Host token 循环 | **PASS（本地）**；核心 55 项及下述直接相关验证通过 |
| P2：QEMU 生成适配/CLI | **PASS（本地小图）**；构建一次，同一 ELF/bundle 跨 token 和跨命令复用；1/4 hart 数值与中断清理通过 |
| `e2e:qemu-paris` | **PASS（本地、显式新策略 4 hart）**；France 完整 8 token 的数值、IDs、文本和停止原因通过，输出包含 Paris。历史旧配置首步 FAIL 的 `max_abs=0.003272057` 保留于实跑报告 |
| `e2e:qemu-prompts` | **PASS（本地、显式新策略 4 hart）**；3/3 用例、24/24 完整 QEMU 前向及逐步数值通过，IDs/text/停止原因与冻结参考一致。历史旧配置初始轮仅检查 3/24 步且均 FAIL，余下 21 步当时未执行 |
| `unit:multiturn` | **PASS（本地）**；40 项会话测试含官方模板/IDs 六份对照；raw 与显式 chat 分开，失败不提交历史 |
| `deploy:script` | **PASS（本地真实容器）**；默认 1/手动 4/auto 小图、真实 QEMU 超时/取消复用与 Docker 构建取消通过；完整 28 层 France 单 token 实际生成 ` Paris`，数值与进程清理通过，不重复宿主三例套件 |
| W5 CI 接入 | **本地实现与检查 PASS；远端 NOT_RUN**；提交前 811 项相关检查通过，最终 11 项重叠边界复查通过；14 个真实必跑节点复用既有证据，本轮未重新执行模型；未提交/推送 |

W4 完整单次前向的已有证据不替代 W5 token 循环；本页 W5 结果来自上述独立完整生成验收。
W4 独立复现、Nightly 和团队签收仍按 [W4 验收说明](../W4/validation.md)分别确认。

完整 QEMU 三例验收已通过，原始结果位于
`output/w5-followup-20261023/qemu-reference-4hart/run/report.json`。
gate 失败状态、旧 W4 构建审计传递、逐步验收证据及取消清理已补强。
本地功能出口已有完整证据：宿主三例 24 步、宿主独立前向，以及干净容器镜像、
小图、中断清理和完整 Paris 单 token 均已通过，见 [部署记录](deployment.md)。
容器此次只补一次完整前向，不将其写成容器三例套件；CI 已完成本地接入和相关检查，远端
**NOT_RUN**，本地功能完成不等于已经取得远端 Actions 或发布验收记录。
不把候选 token 一致或静态部署检查写成实跑 PASS。当前每 token 启动 QEMU，但复用构建；
持久 guest、权重驻留、KV cache 等不属于 W5 v1 必需实现。


## CI 接入与远端执行边界

W5 复用 [w4-full-numeric.yml](../../../.github/workflows/w4-full-numeric.yml)，没有再建立
一套并行的完整模型工作流。手动 `workflow_dispatch` 的 `mode` 为互斥选择：

| mode | 执行范围 |
|---|---|
| `small`（默认） | 相关单测和真实小图，不运行完整模型 |
| `w4-forward` | 原 W4 完整单次前向 |
| `w5-generation` | 原 `deploy_w5.sh` 在真实容器中运行三例 3×8 token，再校验完整报告 |

已有 `workflow_call` 的 `run_full/case` 接口保留，新增 `run_w5=false`，不自动开启生成。
W5 选择与 W4 完整前向冲突，或在非手动事件请求 W5 时，先于模型执行明确失败。

**合并前的手动入口**使用 main 上已经存在的
[llm-deploy.yml](../../../.github/workflows/llm-deploy.yml)，显式传
`run_w5_validation=true`；该模式跳过旧 W1/W2 完整任务与 W3/W4 完整执行，
只调用同一提交的 W5 验证。W5 与 W3/W4 同时勾选会直接失败，不同时跑多套模型。
GitHub 要求手动工作流存在于默认分支，见
[官方手动运行说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)。
审查时远端 main 的 `w4-full-numeric.yml` 尚无 `workflow_dispatch`，因此不能把新文件的
直接手动入口当作仅推送 W5 分支后必然可用；不要为了建立入口先合并未验收的 W5。

下列命令仅供后续获得提交、推送和远端验证授权后使用，本轮没有执行；还需要仓库写权限：

```bash
gh workflow run llm-deploy.yml --ref codex/w5-generation \
  -f run_w5_validation=true -f run_w3_validation=false -f run_w4_validation=false
```

新 `w4-full-numeric.yml` 手动入口进入默认分支后，也可使用上表的 `mode`。
W5 完整 job/步骤/CLI 总预算分别为
**350 / 285 / 275 分钟**；镜像和固定资产落盘后才检查 **至少 4 CPU、10 GiB 可用 RAM、
10 GiB 剩余磁盘**，不足即明确失败，不通过缩小模型或跳过 token 凑齐门槛。
远端 runner 是否实际满足资源和时限，仍须授权提交/推送后实跑确认。

W4 unit 增加 **12 个真实 QEMU 必跑节点**（生成与取消 2、Mean 8、非线性 2）；
[llm-deploy.yml](../../../.github/workflows/llm-deploy.yml) 在现有 tokenizer 环境补 **2 个
官方三轮模板节点**。W5-only 跳过普通 llm-deploy job 时，使用已经构建的同一
W5 镜像、只读 tokenizer 与测试文件执行这两个模板节点，并保留必跑 JUnit 检查；
不会因此再安装一套依赖或重复生成。普通 W5 单测继续由既有通用 CI 承担。
最终汇总区分未选择的 `NOT_RUN` 与请求后失败/取消/skip，不能把缺失证据作为通过。

上一轮 CI 接入的 **753 项轻量检查通过（19.64s）**，覆盖工作流结构、旧 W4 接口、路径路由、
JUnit 缺失/skip/fail、资源短缺、退出码与汇总；其中严格报告 gate 为 44 项。
上述 **14 个必跑节点在 CI 接入阶段仅 collect-only 核对 ID**，该阶段没有再次执行这些节点或完整模型；
已有真实容器生成/取消 2 节点通过，Mean/非线性沿用此前真实小图证据，不把收集当实跑。
[严格报告检查器](../../../scripts/check_w5_ci.py) 对已有真实三例 24 步报告
`output/w5-followup-20261023/qemu-reference-4hart/run/report.json` 离线检查通过。
本地 CI 检查记录为 `output/w5-ci-integration-20261023/validation.json`，
同目录保存测试日志、14 个收集节点 ID 以及 98 段内嵌 shell 的语法检查结果。
这些都是本地实现/证据检查，**远端 W5 CI 仍为 NOT_RUN**，未提交、未推送。

## 提交前检查与收尾（2026-10-23）

已核对 W5 全部待提交改动：生成合同与冻结参考、Host/会话循环、QEMU 适配与 CLI、
显式数值策略、运行器清理、容器打包、第三方声明、CI 路由和报告。
**本地可执行的 W5 工作已完成，未发现尚未处理的本地阻塞；远端 CI 验收仍待执行。**
本轮没有重新运行完整模型或 QEMU 小图，没有暂存、提交或推送。

本轮修复和澄清：

- 两个有限 FP32 logits 的诊断差值原先可能溢出为 Infinity，导致严格 JSON 报告写入失败。
  CLI 和 Host 参考工具现先转为 Python float 再相减；只改变诊断计算，不改变模型、
  argmax、数值容差、冻结 IDs/text 或每行计算顺序。新增正负 FP32 最大值边界测试。
- 补齐合并前可用的 W5-only 手动 CI 入口：复用已注册的 llm-deploy，拒绝与 W3/W4
  同时选择，保留真实小图、官方三轮模板和完整生成门槛。job 预算增加到 350 分钟，
  给准备、清理和报告上传留余量，生成步骤与 CLI 的时限未放宽。
- 修正测量口径：宿主/进程树的 21,032 次采样无读取错误，但 24 步中有 17 步的
  独立 QEMU RSS 各缺一次读取；RSS 峰值是观测值。容器运行器记录的 430.481 秒
  不包含前置输入打包/写入和 ELF/bundle 校验，原始测量值没有改动。

| 本轮必要验证 | 结果与边界 |
|---|---|
| CLI 与 Host 参考工具 | 107 passed，0 skip；包含两处溢出复现及严格 JSON 边界 |
| 数值策略、external 运行器及 W4 契约 | 280 passed，0 skip；默认值、错误传播、上下文/工作区复用和取消清理 |
| CI、部署与打包相关检查 | 811 passed，0 skip；最终 11 项边界复查与此集合重叠，不重复累计 |
| 工作流与脚本静态检查 | actionlint 1.7.12 检查三份工作流、Bash 语法及 diff 检查通过；未运行外部 ShellCheck/Pyflakes |
| 既有三例 24 步报告 | 严格 CI 检查器离线核验 PASS；未重新推理 |

本轮证据在工作树下的 `output/w5-precommit-20261023/`：
`generation-audit.json`、`numeric-audit.json`、`ci-audit.json` 及对应 JUnit。
前后轮测试集合存在交叉，不将数量累加为独立覆盖总数。
默认 1 hart、手动 1/2/4 与启动时 auto 保持正常，W5 padding 强制为 none；
Docker 本轮未启动，WSL 配置未改。

剩余事项按性质区分：

1. **合入/发布前仍须取得远端 CI 证据**：后续获准推送后，使用上节 W5-only 入口；
   runner 资源不足、失败或取消必须明确失败，不能用本地 PASS 替代。
2. **数值验收配置限制**：完整模型 PASS 仅属于已记录的显式参考算术组合；
   原默认 Mean sequential/nonlinear libm 的历史完整 logits 超差仍保留，不能宣称默认路径也通过。
   本轮没有修改算术，因此未重复完整模型。
3. **已知运行边界**：同步构建、传输和 ORT 调用按阶段检查预算，不承诺硬中断任意 ORT 调用。
   KV cache、持久 guest/权重驻留、RVV 和量化属于后续阶段，不属于 W5 v1 未完成的出口项。

## 发布与首次远端 CI 反馈（2026-10-23）

代码、测试和报告已作为独立 [PR #115](https://github.com/ScratchV-Compiler/ScratchV/pull/115)
提交，初始实现提交为 259fc8e。四份报告链接已经验证可访问；没有合并。

首次 [PR CI](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/39917939519)
实际执行并通过 12 个 W5 相关 QEMU 小图必跑节点及 2 个官方三轮模板节点。
W4 unit JUnit 为 1762 passed、4 skipped，四个跳过为三种未提供模型资产的优化器对照及
Linux 上的 Windows junction 专用测试，不涉及上述 12 个必跑节点。完整 W5 门槛按普通
PR 路由未选择，汇总明确为 w5_generation_accepted=false。

通用测试发现 8 个失败：6 个来自旧 W3 测试求值器不支持新增 W5 互斥表达式；
另 2 个来自通用 checkout 未下载 tokenizer 的 Git LFS 内容，测试误把指针当作 JSON。
修复仅涉及测试：补齐 W3/W5 选择矩阵；默认可选资产缺失或为 LFS 指针时明确 skip，
显式指定 tokenizer 目录时仍必须有效，真正损坏的 JSON 不隐藏为 skip。
专用 CI 的两项模板必跑门槛保持不变。修复的直接验证为会话/部署 63 passed、
W3 工作流 54 passed、相关 W4/W5 入口边界 28 passed；均未运行完整模型。
修复推送后的最终 CI 状态以 PR 验证摘要为准，不把首次失败改写为成功。

W5-only 专项在 fork 调度时被 GitHub 以
HTTP 422: Actions has been disabled for this repository 拒绝，未创建 run。
仓库设置显示 Actions enabled=true、workflow active，重新启用指定 workflow 后仅重试
一次仍失败。该外部阻塞尚未解除；详见
[GitHub 官方说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)。
需要仓库所有者向 GitHub Support 核查，不能通过本地 PASS 或普通 PR CI 替代完整专项。
没有改 WSL、默认 hart/padding 或模型数值策略，也没有重复本地完整模型。
用户已接受仓库自动审查，没有额外手动发起审查。

## P0/P1 已有验证（2026-10-23，未重复执行）

| 验证 | 结果 | 覆盖边界 |
|---|---:|---|
| [Host 生成核心](../../../tests/test_generation.py) | 55 passed | 多步输入与结果、0/负数/布尔预算、EOS/容量优先级、非法输出/ID、第二步错误、取消/超时、不同上下文及输出缓冲区复用 |
| [参考工具](../../../tests/test_w5_generation_reference.py) | 16 passed | 独立逐步输入/greedy/停止 oracle、冻结字段差异、语义检查和错误识别 |
| [输入辅助函数](../../../tests/test_llm_inputs.py) / [tokenizer](../../../tests/test_qwen3_tokenizer.py) | 148 passed | 既有固定 ABI、mask、词表、解码和严格边界保持正常 |
| [hart 选择辅助函数](../../../tests/test_riscv_external_harts.py) | 28 passed | 默认/手动/auto、CPU 可用性回退、非法计数和栈规划；11 项不相关测试未选中 |
| [W4 CLI 配置边界](../../../tests/test_w4_full_probe_contract.py) | 10 passed | mock 构建覆盖默认单核、手动 2/4、auto 一次解析及复用 hart 匹配；未启动 QEMU |

P0/P1 共 **257 passed**；本轮新增的适配、CLI、多轮、部署及相关运行器检查见 [续作报告](qemu-generation.md)。不将前后轮重叠测试简单累加。
这组结果验证新 Host 核心及已有配置边界，不代表真实 QEMU 的多 token 端到端门槛通过。
文档检查确认本页本地链接和索引入口可解析，未增加 AI 审查。
