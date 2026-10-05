# 完整模型的局部数值定位

`localize.py` 将同一个完整 ORT 层输入送入原图切片的 ORT 和 ScratchV IR，帮助区分当前层局部误差与上游累计误差。它保留完整 FP32 运算、真实权重和所有 padding query。不会接受 W3 完整数值 gate，也不会修改固定模型或门槛。

先用本目录的 `run.py` 生成完整模型证据，再运行：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:SCRATCHV_FP32_REFERENCE_CPU = 'auto'
python -B -X utf8 probes/w3_qwen3_full/localize.py `
  --model-dir /path/to/qwen3-0.6b-onnx `
  --case-dir output/full-run/short_17 `
  --output-dir output/full-localize-new `
  --layer 0 --layer 21
```

依赖使用完整探测的固定 CPU 环境。输出目录必须不存在；不指定 `--layer` 时仅诊断第一层，层编号从 0 到 27，可重复指定不同层。原始 case 必须包含 `inputs.npz` 和 `ort/`、`ir/` 下完整 worker 的报告、logits、检查点及 schema。工具重新核验固定资产 SHA256、输入/mask、报告记录的数组哈希、普通/诊断 logits 一致性和固定图的 30 检查点结构。历史 IR/ORT 的源码指纹必须彼此一致，但不要求与新增诊断工具后的源码集相同；旧报告和当前诊断的源码指纹分别保留。

诊断沿用原 IR worker 记录的 `fp32_mode`，并核对完整 reference profile；无模式字段的旧报告按 `native` 解释。当前已提交 reference 版本为 `numpy-fp32-reference-v4`，`SCRATCHV_FP32_REFERENCE_CPU` 可设为 `auto`（默认）、`avx2-fma3` 或 `avx512`。本机自动解析出的策略必须与原报告 `ir.fp32_profile.cpu_strategy` 一致；若跨 CPU 诊断，先读取原报告，在启动 Python 前显式设置同一策略。否则工具会拒绝不一致的 profile，不会静默换用当前机器策略。

复现时另外保留原报告与本机 `environment.numeric_runtime` 中的 `blas_environment.OPENBLAS_CORETYPE`、`numpy_openblas.status/libraries` 及可用库项的 `core_name/config`。CPU SIMD 支持、参考策略和实际 BLAS 内核不同；环境变量原值不证明库已采用该内核。可选诊断不可用时保留原说明，不把它作为跳过数值检查的理由。

这个环境变量只改变本次 IR reference 算术策略，不控制本机 ORT dispatch；局部诊断会实际执行本机 ORT/IR，因此不能把“重放保存的 IR 策略”当作两后端必然一致的保证。这与只读取原始数组的 [离线 audit](EVIDENCE.md) 不同，后者无需在当前机器选择或执行生产 CPU 策略。

未知、互相矛盾或不受当前实现支持的 profile 会失败，不能自动换成最新策略。历史 v2/v3 的局部诊断应使用对应的代码版本，或重新生成 v4 完整证据；原报告的模式、profile 和源码哈希不得改写。

v4 的 profile 还包含 MatMul 分块、尾行/单列处理与布局字段；即使 `cpu_strategy` 相同，也必须与保存证据的完整契约一致。当前 v4 实现不能直接用 v3 证据运行并宣称是在复现原算术策略。

所选层按固定图的依赖关系切片；计算节点与权重不重写，以保存的前一层 ORT hidden state 作为共同输入。每层记录 13 个检查点，包含 Q/K RMSNorm、RoPE、Attention probabilities/context、残差及 MLP。第一次归约的 `Pow/ReduceMean/epsilon/rsqrt` 在选择第 0 层时额外记录。

选择第 0 层时还自动运行五项相同输入原语实验：input RMSNorm 的 ReduceMean、Q/K 投影 MatMul、Attention QK MatMul、Softmax。FP64 仅用于估计原语的数学误差，不参与 IR/ORT 执行、不替换任何输出或正式参考。Softmax 另记录不可见 key 概率和行和偏差。

`report.json`、`report.md`、`report.html` 保存误差、最差索引、有效/padding 分项、耗时、进程峰值 RSS 和源码/资产指纹。`layer-XX-values.npz` 保存原始检查点，`op-*-values.npz` 保存原语输入和双方输出，`.onnx` 是本次使用的实际 FP32 局部图。大文件仅在本地输出目录生成。

退出码 0 与 `status=COMPLETE` 仅表示诊断完成；局部误差超过 `1e-4` 会照实记录为比较不通过，不伪装成工具执行失败。资产、输入、schema、执行或有限值检查失败时退出 1。无论诊断完成与否，`numeric_gate_passed=false`、`w3_exit_accepted=false`，原完整模型失败报告保持原样。

不能单凭一个大值中间张量超 `1e-4` 判定语义错误。例如 6000 量级 FP32 数值的一个 ULP 已可能超过该阈值。应同时查看同输入对照、原语误差、值域和原始 mask 概率，最终仍以完整 runner 的全部 logits `<1e-4` 验收。

当前已提交 v4 的[第四轮 Linux 完整运行](https://github.com/yuki-328/ScratchV/actions/runs/37299235103)（`1c49e8acbcd9174491b2d0c2a5a0072178ae5f9a`）已七组完整执行通过；raw 数组 audit 也已通过。该结论来自完整 runner，局部诊断和本说明不替代完整 gate 或团队验收。
