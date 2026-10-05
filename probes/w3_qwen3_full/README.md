# 完整 Qwen3 IR 数值验证

该入口实际执行固定发布的 28 层 Qwen3-0.6B FP32 ONNX，比较 ScratchV IR 与 ORT 的完整 logits，包括右侧 padding query。它使用优化前 IR（none），不会把资产检查、两层模型或部分输入通过视为完整模型数值验收。

## 验证范围

默认运行七组输入：两组满长随机输入、单 token、17/255 个有效 token，以及改变未来 token、改变 padding token 的输入。固定形状为 INT64 input_ids `[1,256]` 和 FP32 加性 mask `[1,1,256,256]`，输出为 FP32 `[1,256,151936]`。输入是可复现的原始 embedding ID，用于数值测试，不作为 tokenizer 或语言生成质量证据。

每组输入分别启动 ORT 和 IR 独立进程；每个后端运行普通与诊断两次，要求两份完整 logits 逐元素完全一致。跨后端对普通、诊断 logits 分别执行严格的 `max_abs < 1e-4`、`rtol=0`，同时报告有效位置和 padding 位置的误差。因果性和 padding 隔离另有四项跨输入检查。

诊断保存 embedding、28 层 residual 输出、final norm 共 30 个检查点，按固定图的实际连接定位。中间张量的 `1e-4` 标记用于定位误差，正式数值门槛作用于完整 logits；某些中间激活很大，单个 FP32 ULP 已可能大于 `1e-4`。报告保留这些标记，不将它们隐藏，也不把中间误差与最终输出误差混为一谈。

## 运行

在 W3 工作树根目录，使用 [固定依赖](../../requirements/qwen3-small-probe.txt)。模型目录必须包含已发布且匹配 manifest 的完整 ONNX 与三份权重分片；入口不下载、不修改模型。

完整入口默认使用 `--fp32-mode reference`，当前策略为 `numpy-fp32-reference-v2`；请在复现命令中显式写出模式。解释器 API 默认仍为 `native`，追加 `--fp32-mode native` 才复现原 NumPy 策略。两种模式和不同版本的 profile 必须分别验收，不能将一份报告的通过结论用于另一种模式。该参考模式不是跨 CPU、BLAS 或 ORT 版本逐元素一致的保证。

~~~powershell
$w3Python = 'D:/cyq/code/ScratchV/.venv-qwen-export/Scripts/python.exe'
& $w3Python -B -X utf8 probes/w3_qwen3_full/run.py `
  --model-dir D:/cyq/code/ScratchV/models/qwen3-0.6b-onnx `
  --output-dir output/w3-full-new `
  --fp32-mode reference `
  --worker-timeout 1800 --max-worker-memory-gib 10
$LASTEXITCODE
~~~

需要复现单个失败时追加 `--case short_17`。成功但仅覆盖部分 case 返回 PARTIAL/2，完整七组和不变量全部通过才返回 PASS/0；实际执行失败或数值超标返回 FAIL/1。中断返回 130。输出目录必须是新目录。

每个 worker 的超时和采样内存上限覆盖本次拥有的整棵进程树，包括 Windows venv Python 启动器创建的实际计算进程。Windows 在恢复启动器执行前将其加入 Job Object，累计该 Job 内的 RSS 与私有提交内存；Linux 使用独立进程组并检查 RSS。采样间隔为 0.2 秒，共享内存页可能按进程重复计数。它是超限终止措施，不是操作系统的硬内存配额，短暂峰值仍可能超过界限。失败保留阶段进度、日志、已完成产物和报告；中断、超时及超限会清理本次拥有的进程树。

## 证据与时长

统一 `report.json`、`report.md` 和 `report.html` 包含每组输入、每个 worker、完整比较和四项不变量。`cases[].comparison.logits` 是最终精度验收数据，`checkpoints` 及 `first_divergence` 定位最早超过诊断阈值的检查点。

每组目录内有 `inputs.npz`、ORT/IR 日志，以及两个后端各自的完整 logits、诊断 logits、检查点 NPZ、schema、报告和阶段进度。报告包含固定模型、输入、输出及源码哈希；统一入口重新校验原始数组与报告一致性。

`cases[].workers[].stages_seconds` 可查看模型解析、普通执行、诊断执行和写盘时间；统一 `elapsed_seconds` 包括所有 worker 及文件核对。`process_peak_rss_bytes` 为该 worker 的全生命周期峰值，`resource_monitor` 是父进程采样观测。两者测量口径不同。

每个 IR worker 的 `ir.fp32_profile` 记录实际数值策略；`environment.numeric_runtime` 补充 CPU 架构、NumPy 构建/运行诊断和线程环境。可用时运行诊断包含 SIMD 与实际 BLAS 信息；它不证明 ORT 使用了哪一种内部内核。独立复现应一并保留这些字段、依赖版本与源码哈希。

完整执行使用只读映射权重、显式借用 initializer、最后一次使用后释放中间 binding；默认解释器 API 仍复制和保留原有数据。只读映射文件在调用期间必须保持不变。解释器的 `peak_numpy_storage_bytes` 计入当时已绑定的借用 buffer，排除已解除绑定但仍由调用方持有的数组、内核临时缓冲和观察回调的拷贝；请同时参考实际进程 RSS。observer 是受信任的同步回调，只读 view 用于防止误写，不提供隔离沙箱。

需要继续定位失败时，使用 [同输入逐层与原语诊断](LOCALIZE.md)。该工具读取已保存的完整执行证据，保存层内检查点和局部 FP64 参考误差；诊断完成不代表完整数值验收通过。

## CI 与阶段结论

[完整模型手动 CI](../../.github/workflows/w3-full-numeric.yml) 在本地准备，尚未提交或在 GitHub 执行。默认完整数值失败会使 job 失败；没有自动忽略 padding、调大阈值、启用低精度或回退到小模型。

该流程同时支持 `workflow_call`，供显式启用的 [Nightly 编排](../../.github/workflows/w3-nightly.yml) 复用；两套自动检查均成功才能通过汇总。Linux 运行、定时启用和资源限制说明见 [Linux CI 与 Nightly](../../docs/llm-deploy-v1.0/W3/Linux-CI与Nightly.md)。收到完整保存目录后，可按 [离线证据复核](EVIDENCE.md) 重新检查数组；这不执行模型，也不替代独立复现。

CI 将报告、哈希和 schema 放入 `w3-full-numeric-reports`（保留 30 天），完整原始输入、logits 和检查点另放入 `w3-full-numeric-raw`（保留 3 天，也包含报告）。需要独立复核原始数值时，应在过期前下载 raw 产物；仅有摘要和哈希不足以重新比较数组。失败或中断运行的产物可能不完整，须先核对报告状态与覆盖范围。

即使本地数值 PASS，`w3_exit_accepted` 仍为 false：W1 人工前置验收、团队确认、E1/E2/E5 独立复现及 Nightly 状态由团队单独核对。[历史本地验收报告](../../docs/llm-deploy-v1.0/W3/W3-完整模型执行与数值诊断报告.md) 保留当时的模式和数据，不代表之后版本的最新状态。当前结论以本次完整运行生成的 `report.json`、实际 profile 和源码指纹为准。
