# W2：完整前端解析、基础后端与 Host 运行时

本目录记录开发计划中的 W2 验收：完整 28 层 Qwen3-0.6B 的前端解析与审计、四类模型模式回归、基础算子 QEMU，以及 Tokenizer/输入/mask/greedy。已有真实两层模型的 IR/ORT/QEMU 门禁继续保留。

W2 完整图的范围止于前端验收。完整 IR 与 ORT 的数值对照及真实维度预训练子图已在独立 [W3 交付](../W3/README.md) 中取得本地证据；权重独立装载、完整 QEMU 前向和生成循环仍按后续阶段推进。W2 不自动关闭 W1 的完整模型第二人复现、接口确认和出口评审。

**交付状态核对（2026-10-05）**：W1 修复 PR #93 已合并；W2 [PR #94](https://github.com/ScratchV-Compiler/ScratchV/pull/94) 尚待评审合并和第二人七项统一复现。初次 W2 交付 `65616a8` 已有三项远端工作流成功记录；集成 W3 的 [PR #96](https://github.com/ScratchV-Compiler/ScratchV/pull/96) 在 `b5a4f58` 上也已通过通用 CI、完整前端及 W1/W2 专项。上述 CI 不能代替第二人执行。初次独立分支的范围与实测见 [分支验收与发布说明](W2-分支验收与发布说明.md)，下方本地整合记录保留各自日期及源码归属，不改写成当前提交的结果。

## W2 独立技术验收入口

使用已有的 Python 3.12 固定 CPU 环境（[依赖](../../../requirements/qwen3-small-probe.txt)）、Zig 0.14.1、QEMU RV64，以及固定完整 ONNX 和官方 Tokenizer 资产。在仓库或完整源码副本根目录执行：

```powershell
$w2Python = 'D:/path/to/qwen3-venv/Scripts/python.exe'
& $w2Python -B -X utf8 scripts/run_w2_acceptance.py `
  --python $w2Python `
  --output-dir output/w2-acceptance-new `
  --tokenizer-dir D:/path/to/qwen3-tokenizer `
  --model-dir D:/path/to/qwen3-0.6b-onnx `
  --cc D:/path/to/zig.exe `
  --qemu D:/path/to/qemu-system-riscv64.exe
$LASTEXITCODE
```

Linux 使用相同参数，将解释器和工具替换为本机路径即可。程序不安装依赖；默认两个资产目录均使用 `verify` 离线校验。无资产时可显式传 `--tokenizer-mode download` 或 `--model-mode download`，后者需要下载约 1.24 GB 的完整模型发布包。所有结果使用本次全新的输出目录，不能复用已有目录，即使为空。

| 执行顺序 | 独立门禁 | 要求 |
|---|---|---|
| 1 | `frontend` | 22 项 RMSNorm/RoPE/SwiGLU/GQA 模式单测，不接受 skip |
| 2 | `backend` | 16 个小图 × none/all，32 次实际 RV64 QEMU |
| 3 | `runtime` | 官方 Tokenizer 20 组/480 对照，以及 5 组输入、mask、采样与停止条件 |
| 4 | `runtime-model` | 官方文本 ID 直入完整词表的真实两层小配置，Torch/ORT/IR none/all 单步接口集成 |
| 5 | `small-ir` | 从固定种子重新导出原两层普通/诊断图，7 组输入、29 检查点与不变性验证 |
| 6 | `small-qemu` | 使用本轮新导出的模型与报告，28 次两层实际 QEMU |
| 7 | `full-parse` | 固定 28 层完整图解析、逐节点/权重绑定审计与 IR verifier |

只有七项全部实际执行并通过才返回 **PASS / 退出 0**。`--gates frontend backend` 等子集支持开发期间定位，但最多返回 **PARTIAL / 退出 2**；失败或阻断返回 **FAIL / 退出 1**。small-qemu 自动包含 small-ir，runtime-model 自动包含 runtime。默认每门禁截止 1800 秒、每次 QEMU 截止 180 秒、完整解析 worker 截止 300 秒，可用 `--gate-timeout`、`--qemu-timeout`、`--parse-timeout` 明确调整。

总报告在 `output/w2-acceptance-new/report.json/md/html`，每项有独立目录和 `logs/<gate>.log`。总入口核对退出码、报告身份、必要计数、实际输出和源码指纹；运行期间源文件变化也会失败。报告记录 HEAD/dirty diff（可用时）及本次源码 SHA256；没有 `.git` 的源码包允许运行，但必须明确标为未知 Git 身份，不能借用父目录仓库的提交。

新 [单步模型集成探测](../../../probes/w2_runtime_model/README.md) 保留完整词表 151936，hidden32/FFN96/两层随机权重。原文本 ID 不裁剪、不取模、不重新映射。逐元素检查完整 logits（含 padding），校验最后有效位置、原始 greedy ID 和官方解码；不要求随机权重生成有意义的文本。本地会保存较大的参考数组（约 614 MiB），CI 仅上传摘要报告，不上传这些参考数组或权重。该探测不执行完整 0.6B、完整词表 QEMU 或多步生成。

当前本地 workflow 已加入单步集成、完整 28 层解析必跑任务和汇总检查。`W2 / required acceptance` 要求轻量/数值任务与完整解析任务均为 success；失败、取消、跳过、缺失结果均不通过。完整 ONNX 的 ORT 前向仍是独立的手动/夜间 W1 任务。配置随本 W2 分支发布，远端 CI 以对应提交的实际执行为准；分支保护规则未变更。源码隔离复现共用本机 Python/工具链时，只能证明不依赖旧源码目录与旧产物，不能代替第二人或另一台机器复现。

统一验收的 Linux 超时路径会先发 SIGTERM，让运行器清理独立会话内的编译器、QEMU 或解析 worker，再执行进程组兜底清理。Windows 继续使用进程树/Job 清理。Linux 专属进程集成回归在 Windows 上明确跳过，发布前须在 Linux 实跑；不能用 Windows 结果替代它。

## 本轮继续交付与最终实测（2026-10-02，本地未提交）

本轮新增完整词表的单步模型接口集成、七项统一验收入口及回归，并补齐原两层 IR 门禁的报告原子写入：MD/HTML 或 JSON 发布失败时必须返回失败，保留原数值错误，不遗留成功 JSON。此前四条报告路径的修复继续保留。

从当前工作树复制 **553 个源码文件**到全新的 `output/w2-independent-source-v2`，包含未提交改动，不复制 `.git`、历史 output、Python 环境或模型产物；从该目录真实运行统一入口，最终 **7/7 PASS，退出 0**。固定 ONNX/Tokenizer 资产重新验证哈希，Python 环境与 Zig/QEMU 工具仍共用本机现有安装。导入路径确认来自新副本，模型重新导出、ELF 重新编译、所有结果重新执行；总报告的 **134 项源码指纹与最终工作树一致**。

总证据：`output/w2-independent-source-v2/output/w2-acceptance/report.json/md/html`；每项明细、误差位置、日志和实际命令位于同一目录下对应子目录。快照来源及逐文件哈希在 `output/w2-independent-source-v2-manifest.json`，外层日志在 `output/w2-independent-acceptance-v2.log`。这些本机产物不进入版本库。

| 门禁 | 最终实测 |
|---|---|
| 前端四模式 | **22/22** 通过，未跳过 |
| 基础后端 | **32/32 QEMU**；最大绝对误差 **2.384185791015625e-7** |
| 运行时 | **20/20** 官方语料、**480/480** 编码/解码对照、**5/5** 输入与采样场景 |
| 单步模型集成 | **2 个原始文本、8 次模型调用**（2 Torch + 2 ORT + 4 IR）；完整输出 `[1,256,151936]`；最大绝对误差 **2.8312206268310547e-7**，采样 ID/解码一致 |
| 两层 IR | **7 组输入、29 检查点、6 项不变性**全部通过；所有数值比较最大绝对误差 **1.9073486328125e-6** |
| 两层 QEMU | **28/28**，普通/诊断图及 none/all 全通过；普通 logits 最大误差 **3.129243850708008e-7**，诊断检查点最大误差 **1.6689300537109375e-6** |
| 完整前端 | **28/28 层、7,847 节点、8,161 value、2,148 绑定**全部通过；5,910 条 IR 指令、verifier 无错误；绑定数据 **2,384,216,772 bytes** |

统一流程 **264.340 秒**；两层 QEMU 进程累计 **157.642 秒**，其门禁全流程 **172.264 秒**；完整解析门禁 **9.538 秒**，其中 parser **2.031 秒**，worker 峰值 RSS **5,064,372,224 bytes（约 4.72 GiB）**。QEMU 计时包含进程启动、guest 运算和输出传输，诊断图包含检查点工作；这些是单次主机观察，不是纯推理性能，也不能据此声称提速。

首次源码副本实跑保存在 `output/w2-independent-source`，总结果正确为 FAIL：前端 pytest 临时目录的父目录未创建，导致 22 个 fixture setup 错误，其余六项通过。修复为使用统一输出根下的临时目录，新增真实 pytest 子进程回归后从新副本完整重跑，没有复用第一次的 PASS。另修复源码包向上查找 Git 的身份错误：运行时、两层 IR 和完整解析入口检查自身 `.git`；缺失时保留源码指纹并明确未知提交。最终实跑不依赖 `GIT_CEILING_DIRECTORIES` 等外部补救设置。

专项回归先执行 **631 passed / 0 skipped**（`output/w2-closeout-combined-tests.xml`）。最终修正上述入口后，对受影响两组分别复验 **72 passed / 0 skipped**（`output/w2-source-identity-all-tests.xml`）与 **47 passed / 0 skipped**（`output/w2-acceptance-tests-d.xml`）；批次重叠，不相加。工作流 YAML、25 段 Bash 语法、文档链接和 diff 格式检查通过。本轮未执行全仓测试或远端 CI，未提交、未推送、未更新 PR #91。

本轮补齐的是 W2 的独立技术验收和单步接口集成。第二人独立复现、团队接口确认以及发布后对应源码的远端 CI 仍需完成；完整预训练模型的 IR 数值、真实维度后端、独立权重装载和多步生成仍属于后续阶段。

## 前一轮：拆分后的本地 Review（2026-10-02）

本轮审查完整解析、四模式前端、基础后端、Tokenizer、输入/mask/greedy，以及本地分支携带的两层 QEMU 修复。发现一类 P2 验收漏洞并修复四条报告路径：必需的报告写入失败时，可能留下可解析的 `passed=true` JSON。它会使仅消费 artifact 的脚本误判，不能以数值计算正确为理由忽略。

| 入口 | 复现与修复 |
|---|---|
| `probes/w2_backend_ops/run.py` | 原来先写 PASS JSON，随后 MD/HTML 失败；改为保存视图结果后发布 JSON，保存失败使门禁非零退出，保留原错误 |
| `probes/w2_qwen3_parse/run.py` | 同类先发布成功状态问题；保留验证阶段、failure_context 和主错误，报告保存异常单列，JSON 最后原子发布 |
| `probes/w2_runtime/run.py` | 虽然原来已最后写 JSON，完整内容写入后关闭/flush 抛异常仍可留下 PASS；改为同目录临时文件成功关闭后原子替换，失败不发布成功 JSON |
| `probes/w2_qwen3_small/riscv.py` | 既有两层 QEMU 门禁存在同类问题；视图与 JSON 原子写入，JSON 最后发布，保留原执行错误与时间证据。此修复只在本地，未同步 PR #91 |

新增 **30 项**失败注入回归，覆盖 MD/HTML/JSON 无法保存、完整写入后报错、JSON 发布失败、原探测已失败及 CLI 状态。没有修改数值算法、精度阈值或模型配置。在本轮实际核对的路径和样本中，未发现新的前端/MatMul/Tokenizer 计算错误；这不构成完整模型数值正确性的证明。

| 本轮验证 | 结果与证据 |
|---|---|
| W2 与两层报告相关专项 | **396 passed / 0 skipped**；`output/w2-review-combined-tests.xml` |
| 前端模式、C 后端与 RISC-V 运行时相关回归 | **138 passed / 0 skipped**；`output/w2-review-compiler-regression.xml`。与上一行有重叠，不相加 |
| 基础后端实际 RV64 QEMU | **32/32**；对 ORT/独立 NumPy 最大绝对误差均 **2.384185791015625e-7**；流程 **23.588 秒**，QEMU 累计 **5.800 秒**；`output/w2-review-backend-real/report.json` |
| 官方 Tokenizer 离线探测 | **20/20** 语料、**480/480** 对照、**5/5** 输入与采样场景；流程 **3.739 秒**；`output/w2-review-runtime-offline-fixed/report.json` |

已核对上述后端 30 项、运行时 5 项源码 SHA256 与修复后代码一致。额外 500 组随机 Unicode/特殊 token 文本编码和 4,000 次解码对照也通过；它们是本轮探索性检查，不算新增 CI 用例。本轮未重跑完整 28 层模型、整两层 QEMU 或全仓测试，也没有远端 CI；历史结果仍按原 SHA 归属。计时为单次观察，不用于声称性能改善。

### 当时可以继续的工作（历史规划）

以下清单保留本地审查时的规划，不作为 2026-10-05 的当前待办：独立 W2 PR #94 已提交，第二至四项已由 W3 PR #96 实现并取得本地证据。当前仍待 W2 第二人复现和评审合并，以及 W3 专项 Linux、Nightly 与团队验收；见本页顶部和 [W3 进度](../W3/W3-工作总结与验收进度.md)。

1. **W2 交付收尾**：第二人用自己的环境执行统一验收入口；未来按用户指示整理独立 W2 PR，明确哪些两层门禁修复应单独回补 W1。当前 PR #91 的绿色状态不是本地 W2 的验收；同机源码副本复现也不能代替第二人验收。
2. **真实维度与预训练子图**：先验证投影 MatMul、RMSNorm、RoPE、Attention、SwiGLU，保留 Torch/ORT/IR 和适当小子图的 QEMU 误差。固定源配置为 hidden=1024、head_dim=128、16 个 Q head/8 个 KV head、FFN=3072。现有两层缩小配置和 3-token GQA case 不能证明这些长累加维度满足误差要求。优先建立基线，再决定 MatMul 分块/布局优化。
3. **参数化中等模型与逐层定位（W3）**：将两层配置和固定 29 检查点推广成配置驱动，跑计划中的六层中等模型，记录首个偏差、最大绝对误差、耗时与峰值内存。Attention 可先沿用现有分解 IR，融合不是正确性前置。
4. **完整 IR 内存准备（W3）**：当前解释器会复制 initializer，并在 `values` 中保留中间结果，没有最后使用后回收。先测权重、活跃张量与 RSS，设计只读权重复用和按生命周期回收；再执行完整 IR/ORT，沿用计划中的 `<1e-4` 门槛。
5. **完整后端与生成集成（W4/W5）**：当前常量/工作区默认各 256 MiB，QEMU 为 512 MiB，权重展开为 C 字面量；仅真实 embedding 就约 593.5 MiB。需要代码与权重分离、装载 ABI、地址布局和容量验证，再接完整前向与生成循环。Host helper 与完整词表两层随机模型的单步调用现已集成；真实预训练模型调用、逐 token 循环、chat template 与流式解码仍未完成。

建议下一轮从第 2 项的真实维度数值基线开始，同时准备第 3 项的可配置探测，避免直接以完整模型试跑替代可定位的小范围验证。

## W2 基础能力补齐（2026-10-02，历史提交 ff193c7）

该次在已通过完整解析和两层数值探测的 `dfc9437` 上补齐三个独立入口，曾更新 PR #91，现已按 W1/W2 拆分要求撤出该 PR 并保留在本地。表中“通过”必须读取对应提交的报告或 CI；本页不把旧提交或跳过项计作当前本地修改通过。

| W2 门禁 | 当前入口与范围 |
|---|---|
| `unit:frontend-ops` | `tests/test_qwen3_frontend_patterns.py`：RMSNorm、RoPE、SwiGLU、GQA 四类模式，含错误 epsilon、旋转符号、门控分支与 KV 重复顺序反例 |
| `unit:backend-ops` | `probes/w2_backend_ops/run.py`：16 小图，MatMul/逐元素/Softmax/RMSNorm/RoPE/SwiGLU/GQA 七类；none/all 共 32 次实际 RV64 QEMU；独立 NumPy 公式和 ORT 双参考 |
| `unit:runtime` | `probes/w2_runtime/run.py`：固定官方七份 tokenizer/配置文件，20 组语料、fast/slow 双参考与 golden IDs/解码文本；mask/最后有效位置/greedy/EOS/容量/预算 |
| `numeric:ir-small` | 既有 `probes/w2_qwen3_small/run.py`：真实两层缩小配置随机权重，继续验收 IR 对照；另保留两层 QEMU 门禁 |
| `frontend:parse-qwen3` | 既有 `probes/w2_qwen3_parse/run.py`：完整 28 层固定产物解析与绑定审计；手动/定时完整模型 CI 实际执行 |

具体命令见 [基础后端门禁](../../../probes/w2_backend_ops/README.md) 和 [运行时门禁](../../../probes/w2_runtime/README.md)。生产新增 `scratchv/runtime/llm_inputs.py` 与 `scratchv/runtime/qwen3_tokenizer.py`；Tokenizer 只依赖可选 `tokenizers`，不在生产模块引入 Torch/Transformers。纯文本编码、完整模型输入准备与单步采样为可复用函数，尚未组合成端到端生成循环。

交叉 review 修复了非默认 prefix/special-token 配置被静默接受而与官方语义不同、缺省 unk token 偏离参考的问题；门禁也增加词表/特殊 ID 元数据与默认参数对照，并修正报告写入失败遗留 PASS 的路径。官方 NFC 规范化是保留行为，不把组合 Unicode 归一化误判成编译器错误。未知或空余 logits 行解码时失败，不静默丢弃或改变 greedy 候选。

本次 CI 增加明确的三个门禁、固定环境单测、独立 Summary 与 `w2-unit-gates-output` artifact。运行时下载约 15.2 MiB 的 tokenizer/config/许可证，按固定 revision 和既有 manifest SHA256 校验，不下载模型权重。源码和测试文件进入 Git，模型、下载缓存、ELF 与本地 output 不进入 Git。

### 本轮本地验收

| 验证 | 结果 | 本地证据 |
|---|---|---|
| 新增专项测试（固定 CPU 环境） | **228 passed / 0 skipped**，含四类前端模式 22 项、后端门禁 18 项、运行时 188 项 | `output/w2-final-unit-tests.xml` |
| 全仓回归（通用开发环境） | **2,732 passed / 41 skipped / 0 failed** | `output/w2-final-full-ready-tests.xml` |
| 基础后端真实 RV64 QEMU | **32 / 32** 通过；对 ORT 与独立 NumPy 参考的最大绝对误差均为 **2.384185791015625e-7** | `output/w2-backend-ops-final/report.json/md/html` |
| 官方 Tokenizer 下载与运行时门禁 | **20 / 20** 语料、**480 / 480** 比较、**5 / 5** 输入与采样场景通过 | `output/w2-runtime-final-download/report.json/md/html` |
| 同一执行身份离线复验 | 禁用 Hugging Face 在线访问，复用经过哈希校验的本地资源；上述 **20 / 480 / 5** 全部通过 | `output/w2-runtime-final-offline-same-user/report.json/md/html` |

41 个跳过项包括本机缺少 `qemu-riscv32` 的 38 项，以及通用环境没有 Torch/Tokenizers 的 3 个模块；新 Tokenizer 模块已在上述固定 CPU 环境执行。两个既有 Torch 模块在本轮未本地重跑，PR CI 的固定环境继续执行原两层数值探测。各批次测试有重叠，不能相加成通过总数。

首次全仓运行因子进程未获得 LLVM 工具路径失败，修正测试进程的 PATH 后得到上表结果；首次跨执行身份的离线复验因 Windows 目录读取权限失败，同一身份复验通过。失败报告仍保留，未通过修改数值门槛或跳过检查消除失败。

后端探测全流程 **22.194 秒**，32 次 QEMU 进程墙钟累计 **5.660 秒**；Tokenizer 下载门禁 **14.824 秒**，同一身份离线复验 **4.198 秒**。均为本机单次观察，不是性能改善结论。已核对后端 30 项、运行时 5 项源码指纹与本次源码一致。仓库指定的本地 L2 harness 文件缺失，未宣称执行；本轮以实际专项、全仓和真实探测结果交付。

## 完整前端解析的本地复现

详细环境、Windows/Linux 命令、下载模式和报告说明见 [探测入口 README](../../../probes/w2_qwen3_parse/README.md)。所有命令在仓库根目录执行；使用 Python 3.12 和固定 NumPy/ONNX/ORT/protobuf 环境，无需 torch、Zig 或 QEMU。

已有完整模型时：

```bash
python -X utf8 -B probes/w2_qwen3_parse/run.py --mode verify --model-dir output/qwen3-full-model --output-dir output/qwen3-parse --timeout 300
```

PowerShell 可使用实际虚拟环境解释器：

```powershell
& .\output\qwen3-parse-venv\Scripts\python.exe -X utf8 -B probes/w2_qwen3_parse/run.py --mode verify --model-dir output/qwen3-full-model --output-dir output/qwen3-parse --timeout 300
$LASTEXITCODE
```

没有完整模型时，将 `verify` 改成 `download`，仍使用同一固定 W1 manifest。模型必须包含真实 external-data 分片。每次使用新的报告目录，失败时保留已有文件与日志。

## 如何判断本阶段完成

| 项目 | 应保留的证据 |
|---|---|
| 源码身份 | checkout SHA、是否有本地改动、实际 diff 与新增文件 |
| 固定模型身份 | 发布产物和 external-data 文件校验结果 |
| 完整结构与层覆盖 | 本次结构审计摘要及 28 层检查结果 |
| 实际节点转换 | `nodes.json` 与审计摘要 |
| initializer 绑定 | `bindings.json` 与审计摘要 |
| 共享 IR | `ir.txt` 与 IR 验证器结果 |
| 本次执行 | 命令、退出码、`report.json/md/html`、实测时间及内存（如平台可测） |
| CI | 对应提交和实际 run URL；未执行时明确写未执行 |

退出 0 和 `passed=true` 必须来自本次执行。轻量 fixture 测试、历史 ORT 通过或一个可打开的报告文件都不足以代替完整图审计。失败发生较早时，某些明细可能尚未产生；按实际阶段和错误定位，不将缺失记录补成成功。

本地 W2 workflow 现在让 PR、手动和定时任务均执行独立的 `full-qwen3-frontend` 完整解析任务，无须等待完整 ONNX 的 ORT 前向。该任务下载约 1.24 GB 的固定发布包，解析时需约 5 GB RSS；手动/夜间同时运行 ORT 任务时，两台 runner 会各自下载。汇总任务拒绝失败或跳过的依赖任务。PR #91 已合并；未来 W2 远端结果须关联其独立提交和实际 run，不能继承旧提交的 PASS。

## 完整解析 Review 与实测（2026-10-02，历史提交 dfc9437）

该次同时复核本地完整解析门禁和 [PR #91](https://github.com/ScratchV-Compiler/ScratchV/pull/91)。核对时 PR head 为 `df18b02f152878b1ea838bbf6073def01d367456`，base 为 `d1c2b5f04cad817acf0b2f14d3bfa4ff56dbfa11`，当时差异涉及 52 个文件。审查覆盖编译入口、IR 优化、tensor-c 后端、RISC-V 运行时、数值探测、CI 和复现说明。下述完整解析与 Review 修复曾在 `dfc9437` 纳入 PR #91，现已撤出该 PR 并保留于本地；当时实际执行源码由报告中的 SHA256 确定。

### 完整解析门禁修复

| 问题 | 修复与验证 |
|---|---|
| 属性审计复用生产 parser 的属性解码，可能让相同错误在转换和审计两端同时通过 | 直接从原 ONNX protobuf 独立读取属性，拒绝重复属性；增加错误 Transpose 轴和错误 epsilon 的反例 |
| 未检查函数参数名称与常量标记 | 对照原 ONNX 输入校验名称、顺序、shape/dtype，并拒绝被标记为常量的输入 |
| 标量辅助 `LOAD_CONST` 校验不完整 | 对 initializer 和 Constant 两条路径核对 opcode、目标 Value、空操作数、shape/dtype、常量标记、常量值及指令属性 |
| 审计时隐式 `np.asarray` 可能掩盖错误的绑定类型 | 要求实际绑定是 `numpy.ndarray`；list 等错误类型直接失败，合法只读 ndarray 仍接受 |
| 日志写入失败可能覆盖 worker 原始错误 | stdout/stderr 独立尝试保存；保留原错误、阶段及退出码，附加 `log_errors`；成功路径缺少日志也不能报 PASS |
| 文档链接指向被忽略的本地输出，干净 checkout 中无法访问 | 改为明确的本地输出路径说明，并增加模拟缺少 output 目录的文档链接回归测试 |

进一步修正失败明细写入的同类问题：保存 `nodes.partial.json` 等失败证据时，写入异常记录到 `evidence_errors`，不覆盖原始解析或绑定错误；成功路径缺少必需证据仍失败。supervisor 在启动 worker 前保存 `attempt`，记录命令、模型路径、解释器、checkout 和源码指纹，便于定位过早超时、worker 未能启动等情况；该字段不代表验证已经执行。

### PR #91 修复

| 问题 | 修复与验证 |
|---|---|
| tensor-c 接受与 IR 标量常量值不同的外部绑定，可能与解释器行为不一致 | 校验绑定与常量声明一致；增加 FP32/INT32/INT64 错值、错误向量形状和合法标量执行的回归测试 |
| 编译或 QEMU 失败时，保存日志的异常可能掩盖编译错误、超时或 guest 错误 | 独立保存两路日志；保留主错误及原耗时，并附加日志错误。覆盖编译失败/成功与 QEMU 超时/失败/成功等组合 |
| RISC-V 报告不足以确定本次 ORT/IR 对照的实际执行环境和完整源码身份 | 顶层 `environment` 记录当前进程，`export_evidence.environment` 保留导出时环境；源码指纹覆盖 parser、IR、优化、后端和运行时，共 30 项 |

未发现本次审查范围内需要额外修改的 MatMul 实现问题；下列结果只证明已测试输入和路径。未调整精度门槛、模型权重或 D8 等团队待确认决策。

### 修复后的实测

| 验证 | 当前结果 | 本地证据 |
|---|---|---|
| 全仓回归 | 2,563 passed / 40 skipped / 0 failed | `output/review-fix-full-tests.xml`、同名 `.log` |
| 固定 CPU 环境最终相关测试 | 286 passed / 0 skipped，包含全部 114 项完整解析测试 | `output/review-fix-pinned-tests.xml`、同名 `.log` |
| 真实 Qwen3 结构两层小模型重新导出与 PyTorch/ORT/IR 对照 | 7 / 7 组输入通过；全部数值比较最大绝对误差 `1.9073486328125e-6` | `output/review-fix-qwen3-small/report.json` |
| RISC-V QEMU 两层探测 | 7 组输入 × 普通/诊断图 × none/all 优化，共 28 / 28 执行通过；QEMU 对照最大绝对误差 `1.6689300537109375e-6`，门槛 `atol=1e-5, rtol=0` | `output/review-fix-qwen3-riscv/report.json`、`report.md` |
| QEMU MatMul | 7 / 7 通过，最大绝对误差 `2.384185791015625e-7` | `output/review-fix-matmul/report.json` |
| 完整 Qwen3 前端解析 | 28 / 28 层；7,847 节点、23 类算子；5,910 条 IR；2,148 项绑定；IR verifier 0 问题 | `output/review-fix-full-qwen3-final/report.json`、`report.md`、`report.html` |

全仓回归中的 38 个 RV32 执行测试因缺少 `qemu-riscv32` 跳过，另有 2 个依赖 torch 的模块因通用开发环境不含 torch 而跳过；最终固定 CPU 环境相关测试已覆盖这两个模块。全仓测试收集后补充了 3 项失败证据回归测试，均包含在最后的 286 项测试内；没有把不同批次或重叠测试相加为一个通过总数。RV64 tensor 真实运行时集成测试和上述实际 QEMU 探测已执行。

本次 QEMU 进程墙钟时间累计 **160.450 秒**，探测全流程 **174.634 秒**，4 个成功构建累计 **6.586 秒**。普通图每次平均约 **1.143 / 1.147 秒**（none/all），诊断图约 **10.138 / 10.494 秒**。QEMU 时间包含进程启动、guest 计算、输出打包/UART 和退出；诊断图还包含中间张量输出，不能当作纯前向耗时或真实硬件性能。

完整模型本次纯解析 **3.636 秒**，入口至验证及日志保存结束 **12.772 秒**，worker 峰值 RSS **5,064,331,264 bytes，约 4.72 GiB**。完整模型输出静态契约为 FP32 `[1,256,151936]`，绑定共 **2,384,216,772 bytes**；这些是解析与绑定审计结果，本次未执行完整模型数值前向。以上耗时均为本机单次运行，期间存在并行测试，不用于声称性能改善。

已核对完整解析报告的 20 项、RISC-V 报告的 30 项源码 SHA256 与当前源码一致；完整解析的 `nodes.json`、`bindings.json`、`ir.txt` 的文件大小和 SHA256 均与报告一致。测试后仅补充说明文档，执行源码未再修改。输出目录被 Git 忽略，仅在本机保存；他人复现需运行入口生成自己的报告。

最终还实跑了 0.1 秒截止和模型缺失两个失败路径：均正确退出 1，分别记录 `timeout`、`hashes` 阶段，且保留启动前的 20 项源码指纹；报告在 `output/review-fix-final-expected-timeout/` 和 `output/review-fix-final-expected-missing/`。文档检查包含尚未跟踪的新文档，共检查 97 个文件、196 个相对链接，0 个失效；LLM workflow YAML 与 19 段 shell 命令语法检查通过，`git diff --check` 通过。

当前没有已确认且尚未修复的本轮阻断性问题。完整 IR/ORT 数值与容量验收、完整模型 QEMU 前向、权重独立装载、W1 第二人复现和接口签字仍未由本次工作完成。本页实测均为本地证据，远端 CI 状态及 run 链接以 PR 描述和 Checks 为准。

## 首次交付结果（2026-10-02，历史记录）

首次交付工作在独立本地分支 `codex/w2-full-qwen3-parse`，基于 PR #91 的 `df18b02f152878b1ea838bbf6073def01d367456`。实际工作目录为 `D:/cyq/code/ScratchV/output/w1-pr89-review`；当时修改未暂存、未提交、未推送，尚未更新 PR #91。上层原工作目录中的已有改动未纳入本轮。

首次交付的完整实物执行退出 0，结果 **PASS**。本地证据保存在 `output/w2-full-qwen3-parse-complete/`：`report.md`、`report.json`、`nodes.json`、`bindings.json` 和 `ir.txt`。这些是该次 checkout 的本地输出，`output/` 不进入版本库；其他机器需执行入口重新生成，不能依赖这些历史路径存在。

| 验收项 | 首次交付实测 |
|---|---|
| 固定模型 | Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`，FP32 / L=256 / 无 KV Cache |
| 完整层链 | 28 / 28 层，含最终归一化与共享 embedding / LM Head |
| ONNX 节点 | 7,847 / 7,847；23 类基础算子；全部可达 logits |
| ONNX value | 8,161 项静态 shape / dtype 审计通过 |
| 实际 IR | 5,910 条指令；IR verifier 0 个问题 |
| 真实绑定 | 312 个 initializer + 1,836 个 Constant，共 2,148 项；逐项内容哈希通过 |
| 绑定字节数 | 2,384,216,772 bytes |
| 合法别名 | 112 个 Identity，均核对实际对象映射 |
| 返回值 | `logits`，FP32 `[1,256,151936]` |
| 纯解析耗时 | 2.493 秒 |
| 全入口耗时 | 10.533 秒，包含检查、解析、审计与子进程开销 |
| 验证进程峰值 RSS | 5,063,430,144 bytes，约 4.72 GiB，Windows PeakWorkingSetSize |

耗时来自单次本地运行，不是平均值或性能承诺；本轮没有做解析性能优化。报告记录运行环境、实际源码 SHA256 和产物哈希。工作树有本地改动，不能将此结果仅归属于基准 HEAD。

首次交付的轻量测试 **91 / 91 通过**，在固定 CPU 环境和现有开发环境各执行一次，分别记录于 `output/w2-gates-pinned-delivery.xml` 和 `output/w2-gates-general-delivery.xml`。测试含真实两层 fixture 经文件哈希、结构检查、生产 parser、IR verifier 和绑定审计的贯通，以及漏节点、错权重、错 shape/dtype、外部数据损坏、解析异常、缺报告与超时等反例。当时全仓回归 **2,521 passed / 40 skipped**，记录于 `output/w2-regression-final.xml`；首次交付最后修正后重跑上述全部 91 项相关测试。当前修复后的验证以本页上方结果为准；跳过项不计为通过。

实际进程反例也已执行：模型缺失在 `hashes` 阶段退出 1；0.1 秒截止测试在 `timeout` 阶段退出 1，实测总耗时约 0.184 秒，均保存 FAIL 报告。记录分别位于 `output/w2-expected-missing-model/` 和 `output/w2-expected-timeout/`。

首次交付新增三个探测模块、三组测试与复现文档，并更新本地 LLM CI 配置。没有发现这份完整固定图需要新增前端算子，当时未修改通用 ONNX parser 或编译后端。首次互审修正了验收实现中的两个重点问题：控制参数必须从原 ONNX 独立求值，避免解析器错误自证通过；超时日志写入失败、进程异常退出及 verifier 失败时应尽量保留原始错误和节点证据。后续 Review 的更多修复见本页上方。

本阶段已具备完整前端解析的本地验收证据；远端 CI、第二人复现、完整模型 IR 数值及完整模型 QEMU 前向尚未由本轮验证。

## 后续衔接

1. 前端审计通过后，依据实际图和权重规模建立完整 IR 数值与峰值内存基线，逐步扩大预训练模型验证范围。
2. 设计代码与权重分离、权重绑定 ABI 和内存布局；当前 C 常量展开及小模型 guest 配置不能直接承担完整 0.6B 模型。
3. 完整 IR 数值验收通过后，再验证完整 QEMU 单次前向；随后将已有 Tokenizer、输入准备、最后有效 token logits、采样和停止条件组合成生成循环，并验证真实文本输出。

前端解析通过不能跳过这些阶段，也不能据此宣布完整 Qwen3 已在 ScratchV 上运行。开发范围见 [总开发计划](../开发计划.md)，W1 人工验收状态见 [W1 独立复现记录](../W1/README.md)。
