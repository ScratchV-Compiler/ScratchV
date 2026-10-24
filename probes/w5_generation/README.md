# W5 Host 生成参考

本探针验证 Host token 循环；实际推理后端是固定完整 Qwen3-0.6B ONNX 的 ONNX Runtime CPU。它不启动 QEMU，不能代替 W5 的 QEMU Paris / 三 prompt 验收。生成合同见 [W5 合同](../../docs/llm-deploy-v1.0/W5/README.md)。

## 固定参考

[references.json](references.json) 固定以下三个原始 completion prompt；它们在第一次运行前选定，每条预算 8 个新 token，分别检查输出包含 Paris、Berlin、Tokyo。

- `The capital of France is`
- `The capital of Germany is`
- `The capital of Japan is`

prompt 按 UTF-8 原文编码，`add_special_tokens=False`，不套 chat template。模型身份引用既有 [W1 manifest](../w1_qwen3_export/manifest.json)，不增加另一套哈希清单。运行前检查 ONNX 文件长度，避免把 LFS pointer 当模型；资产完整性沿用已有下载/验收流程。本探针不宣称长度检查等价于 SHA256 校验。

参考保存 prompt 字节及 token IDs、所有生成 IDs、文本、EOS 集、停止原因和逐步 top-1/top-2 间距。使用 full FP32 logits 的最后有效行；检查完整输出（包括 padding 行）有限性。argmax 覆盖整个模型词表；采样到 tokenizer 不支持的 padded ID 会失败，不做过滤。

## 运行

先按 [Linux 复现约定](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md) 安装选定环境并获取固定模型、tokenizer 资产。在仓库根目录运行：

```bash
python probes/w5_generation/run.py \
  --assets assets/qwen3-0.6b \
  --output-dir output/w5-generation-reference
```

正常运行只读冻结参考，写 `output/w5-generation-reference/reference-run.json`。`--assets` 可指向已有完整资产目录，避免工作树重复下载。默认模式要求生成 token IDs、文本和停止原因与冻结参考完全一致；浮点 top-2 间距用于诊断，不把它当跨后端数值误差标准。公开 CLI 的 `--record` 只向 output 写候选参考，**不会覆盖 references.json**；更换冻结参考必须单独审查 prompt、模型身份及验收条件的变化。

一个 ORT session 顺序处理三个用例，沿用 W4 参考设置：CPU provider、intra/inter-op 均为 1、禁用 graph optimization 和 CPU memory arena。每次 forward 都由独立 oracle 检查：

1. 当前 token 历史、最后有效位置、右 padding 和 causal/key padding mask。
2. 完整 logits 的 FP32 ABI 与有限性，直接索引最后有效行并用 NumPy argmax 计算 token。
3. 独立的 EOS → 容量 → token 预算停止规则、生成 IDs、解码文本和调用次数。

这样一次真实 ORT 运行同时验证 Host 循环和参考输出，不再重复跑同一套完整参考。结果中 `qemu_e2e: NOT_RUN` 明确区分本轮覆盖范围。

## 计时与内存

每步同时记录 ORT forward wall/CPU 时间；每条用例另记包括 Host 输入准备、检查、采样及解码的 wall/CPU 时间。session 加载单独计时。CPU 时间含进程内所有线程；Linux `ru_maxrss` 是进程截至当时的累计峰值，不能作为每个用例的独立内存增量。这些是 ORT/Host 数据，不能与既有 QEMU 计时混用或推算 QEMU 加速比。

本探针不读写 hart、padding、CI 或 WSL 设置。QEMU 的默认 1 hart、手动 1/2/4、启动时 auto 和默认关闭的 padding 仍由既有后端负责。
