# 完整 Qwen3 IR 数值验证

复现统一采用 **Ubuntu 24.04 x86_64、Bash、Python 3.12**。先完成 [Linux 环境与资产准备](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md)，再在同一个 Bash 会话、仓库根目录执行命令；该指南设置 `SCRATCHV_PYTHON`、`SCRATCHV_CC` 和 `SCRATCHV_QEMU`。

该入口实际执行固定发布的 28 层 Qwen3-0.6B FP32 ONNX，比较 ScratchV IR 与 ORT 的完整 logits，包括右侧 padding query。它使用优化前 IR（none），不会把资产检查、两层模型或部分输入通过视为完整模型数值验收。

## 验证范围

默认运行七组输入：两组满长随机输入、单 token、17/255 个有效 token，以及改变未来 token、改变 padding token 的输入。固定形状为 INT64 input_ids `[1,256]` 和 FP32 加性 mask `[1,1,256,256]`，输出为 FP32 `[1,256,151936]`。输入是可复现的原始 embedding ID，用于数值测试，不作为 tokenizer 或语言生成质量证据。

每组输入分别启动 ORT 和 IR 独立进程；每个后端运行普通与诊断两次，要求两份完整 logits 逐元素完全一致。跨后端对普通、诊断 logits 分别执行严格的 `max_abs < 1e-4`、`rtol=0`，同时报告有效位置和 padding 位置的误差。因果性和 padding 隔离另有四项跨输入检查。

诊断保存 embedding、28 层 residual 输出、final norm 共 30 个检查点，按固定图的实际连接定位。中间张量的 `1e-4` 标记用于定位误差，正式数值门槛作用于完整 logits；某些中间激活很大，单个 FP32 ULP 已可能大于 `1e-4`。报告保留这些标记，不将它们隐藏，也不把中间误差与最终输出误差混为一谈。

## 运行

在 W3 工作树根目录，使用 [固定依赖](../../requirements/qwen3-small-probe.txt)。模型目录必须包含已发布且匹配 manifest 的完整 ONNX 与三份权重分片；入口不下载、不修改模型。

完整入口默认使用 `--fp32-mode reference`，当前已提交策略为 `numpy-fp32-reference-v4`；请在复现命令中显式写出模式。解释器 API 默认仍为 `native`，追加 `--fp32-mode native` 才复现原 NumPy 策略。两种模式、不同版本和不同 CPU 策略的 profile 必须分别记录与验收，不能将一份报告的通过结论用于另一种模式。该参考模式不是跨 CPU、BLAS 或 ORT 版本逐元素一致的保证。

环境变量 `SCRATCHV_FP32_REFERENCE_CPU` 只作用于 `reference` IR 的算术策略，允许以下取值：

| 值 | 行为 |
|---|---|
| `auto`（未设置时的默认值） | 从 NumPy 报告的可用 CPU 能力推断；具有 AVX2/FMA3 时，根据 AVX512F 是否可用选择下列一种策略 |
| `avx2-fma3` | 显式采用对应的参考算术顺序，profile 记录该值 |
| `avx512` | 显式采用对应的参考算术顺序，profile 记录该值 |

显式设置不会控制 ORT 的 CPU dispatch，也不是启用或禁用真实硬件指令的开关；两条参考路径仍由 NumPy 实现。跨机器重放保存的算术策略时，可以在启动 Python 前显式指定报告中的 `cpu_strategy`，但必须重新比较本机 ORT 输出。未知变量取值会报错；`auto` 模式无法识别所需 CPU 能力，或发现 `NPY_DISABLE_CPU_FEATURES` 遮蔽相关 AVX/FMA 能力时也会明确报错，要求显式选择并记录策略。

v4 把矩阵乘的分块与内存布局纳入版本化 profile；数值策略名字与 CPU 策略相同仍不足以替代完整字段核对：

| profile 字段 | `avx2-fma3` | `avx512` |
|---|---|---|
| `matmul_k_block` | 128 | 128；K≤128 时保留原 NumPy 路径 |
| `matmul_row_tile` | 2 | `null`，保留原宽矩阵路径 |
| `matmul_row_tail` | `zero-pad-to-two-then-trim` | `native` |
| `matmul_column_tail` | `zero-pad-single-column-to-two-then-trim` | `native` |
| `matmul_layout` | `contiguous-matrix-operands-per-k-block` | `native-strides` |
| `matmul_vector_policy` | `native-numpy-vector-promotion` | `native-numpy-vector-promotion` |

AVX2 矩阵路径按两个输出行计算，尾行和单列临时补零后裁回原输出形状，每个 K 分块的两个矩阵操作数使用 C 连续切片；向量提升继续遵循 NumPy。以上是参考计算契约，不改模型图、权重、输出形状或误差门槛，也不是性能优化验收。本节描述已提交的 v4 实现；第四轮 Linux 已取得同提交七组完整运行 PASS，普通/诊断完整 logits 及四项不变量最大误差均为 0；这是 v4 的新实测证据，历史 v3 或单例结果不能替代它。

~~~bash
set -euo pipefail
W3_MODEL=/absolute/path/to/qwen3-0.6b-onnx
export SCRATCHV_FP32_REFERENCE_CPU=auto
"$SCRATCHV_PYTHON" -B -X utf8 probes/w3_qwen3_full/run.py \
  --model-dir "$W3_MODEL" --output-dir output/w3-full-new \
  --fp32-mode reference --worker-timeout 1800 --max-worker-memory-gib 10
~~~

需要复现单个失败时追加 `--case short_17`。成功但仅覆盖部分 case 返回 PARTIAL/2，完整七组和不变量全部通过才返回 PASS/0；实际执行失败或数值超标返回 FAIL/1。中断返回 130。输出目录必须是新目录。

每个 worker 的超时和采样内存上限覆盖本次拥有的整棵进程树。Linux 使用独立进程组，采样整个进程树的 RSS，并在超时、超限或中断时清理本次拥有的进程。采样间隔为 0.2 秒，共享内存页可能按进程重复计数。它是超限终止措施，不是操作系统的硬内存配额，短暂峰值仍可能超过界限。失败保留阶段进度、日志、已完成产物和报告；中断、超时及超限会清理本次拥有的进程树。

## 证据与时长

统一 `report.json`、`report.md` 和 `report.html` 包含每组输入、每个 worker、完整比较和四项不变量。`cases[].comparison.logits` 是最终精度验收数据，`checkpoints` 及 `first_divergence` 定位最早超过诊断阈值的检查点。

Summary 将普通图与诊断图的最终 logits 分开，明确实际 `native/reference` 模式，并报告最大绝对误差、余弦相似度和相对 L2 误差；逐用例索引默认折叠，完整检查点和机器证据仅保存在 `report.json`，不重复嵌入摘要。后两项由实际输出和参考数组分块计算，使用 FP64 累积，不改变原有通过门槛。不能把参考兼容模式的通过结果归给默认 NumPy 模式。旧报告不补写或重新命名为本轮运行；离线重算保存数组也不代表重新执行了模型。

PPL（真实文本预测的困惑度）和 zero-shot（无示例任务评分）本轮未评测。这里的随机 token 输入用于数值对齐，不能作为语言质量语料；加入这类评测前需要固定真实文本、Tokenizer、L=256 的分窗及计分方式。

每组目录内有 `inputs.npz`、ORT/IR 日志，以及两个后端各自的完整 logits、诊断 logits、检查点 NPZ、schema、报告和阶段进度。报告包含固定模型、输入、输出及源码哈希；统一入口重新校验原始数组与报告一致性。

`cases[].workers[].stages_seconds` 可查看模型解析、普通执行、诊断执行和写盘时间；统一 `elapsed_seconds` 包括所有 worker 及文件核对。`process_peak_rss_bytes` 为该 worker 的全生命周期峰值，`resource_monitor` 是父进程采样观测。两者测量口径不同。

每个 IR worker 的 `ir.fp32_profile` 记录实际数值策略，包括 `name=numpy-fp32-reference-v4` 与已解析的 `cpu_strategy`；`auto` 是选择方式，不是保存后的策略名称。`environment.numeric_runtime` 补充 CPU 架构、NumPy 构建/运行诊断和线程环境。`blas_environment.OPENBLAS_CORETYPE` 保存环境变量原值，未设置时为 `null`，不是实际生效内核的证明；`numpy_openblas` 保存可选诊断的总体 `status`、`libraries`，可用库项包含 `path`、`status`、`core_name`、`config`，不可用时保存 `reason` 或 `error`。该诊断只读取已装载的 NumPy 私有 OpenBLAS 库，系统 BLAS/MKL 等可明确为不可用。

CPU SIMD 能力、IR 的 `cpu_strategy` 与实际 BLAS 内核须分别看待；上述字段也不证明 ORT 使用了哪一种内部内核。可选诊断不可用或旧证据缺少该字段，不能掩盖数值失败，所有门槛照常执行。独立复现应一并保留这些字段、环境变量原值、依赖版本与源码哈希。

离线 audit 按生产报告中保存的 CPU 策略核对整个 profile，不从审计机器 CPU 推断生产策略，也不会重新运行模型。当前 v4 工具严格验证 v4 契约；历史 v2/v3 原始证据应在其对应源码版本下核对，不能把报告字段改为 v4 或直接沿用旧版本的 PASS。每个提交和策略的完整七组以对应运行结果验收。

完整执行使用只读映射权重、显式借用 initializer、最后一次使用后释放中间 binding；默认解释器 API 仍复制和保留原有数据。只读映射文件在调用期间必须保持不变。解释器的 `peak_numpy_storage_bytes` 计入当时已绑定的借用 buffer，排除已解除绑定但仍由调用方持有的数组、内核临时缓冲和观察回调的拷贝；请同时参考实际进程 RSS。observer 是受信任的同步回调，只读 view 用于防止误写，不提供隔离沙箱。

需要继续定位失败时，使用 [同输入逐层与原语诊断](LOCALIZE.md)。该工具读取已保存的完整执行证据，保存层内检查点和局部 FP64 参考误差；诊断完成不代表完整数值验收通过。

## CI 与阶段结论

[完整模型手动 CI](../../.github/workflows/w3-full-numeric.yml) 已随 PR #96 发布。2026-10-05，作者在提交 `1c49e8acbcd9174491b2d0c2a5a0072178ae5f9a` 的[第四轮 Linux 手动编排](https://github.com/yuki-328/ScratchV/actions/runs/37299235103)中通过 preparation、完整七组 full 和汇总；v4/avx2-fma3、OpenBLAS Haswell 环境下普通/诊断 logits、30 检查点与四不变量最大误差均为 0，完整 runner 用时 `1477.314971525 s`（约 24.6 分钟）。reports 已下载并核对，完整 raw 已下载验真、离线 audit 通过。前三轮失败及旧 v2/v3 记录保留在 [Linux 实测记录](../../docs/llm-deploy-v1.0/W3/Linux-CI与Nightly.md)。实际完整数值失败仍使 job 失败；没有自动忽略 padding、调大阈值、启用低精度或回退到小模型。

该流程同时支持 `workflow_call`，供显式启用的 [Nightly 编排](../../.github/workflows/w3-nightly.yml) 复用；两套自动检查均成功才能通过汇总。Linux 运行、定时启用和资源限制说明见 [Linux CI 与 Nightly](../../docs/llm-deploy-v1.0/W3/Linux-CI与Nightly.md)。收到完整保存目录后，可按 [离线证据复核](EVIDENCE.md) 重新检查数组；这不执行模型，也不替代独立复现。

CI 将报告、哈希和 schema 放入 `w3-full-numeric-reports`（保留 30 天），完整原始输入、logits 和检查点另放入 `w3-full-numeric-raw`（保留 3 天，也包含报告）。需要独立复核原始数值时，应在过期前下载 raw 产物；其解压目录中保留 `w3-full/` 前缀，所以 `--evidence-dir` 应为 `下载目录/w3-full`。下载命令、报告 SHA256 获取与完整 audit 示例见 [产物与失败定位](../../docs/llm-deploy-v1.0/W3/Linux-CI与Nightly.md#产物与失败定位)。仅有摘要和哈希不足以重新比较数组。失败或中断运行的产物可能不完整，须先核对报告状态与覆盖范围。

即使本地数值 PASS，`w3_exit_accepted` 仍为 false：W1 人工前置验收、团队确认、E1/E2/E5 独立复现及 Nightly 状态由团队单独核对。[Linux 完整执行记录](../../docs/llm-deploy-v1.0/W3/W3-完整模型执行与数值诊断报告.md) 保留固定提交、模式和数据，不代表之后版本已经重跑。当前结论以本次完整运行生成的 `report.json`、实际 profile 和源码指纹为准。
