# Qwen3-0.6B KV cache ONNX

来自 `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，原始 checkpoint SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。采用 FP32、batch=1、固定缓存容量 256、opset=18。许可见 [LICENSE](../LICENSE)。

`prefill.onnx` 和 `decode.onnx` 共用 `weights-*.data`。每片不超过 1 GiB，可由 Git LFS 存储。图和全部权重分片必须保存在同一目录；各文件 SHA-256 见 [manifest.json](manifest.json)。

两个图分别接收 `[1,256]` 和 `[1,1]` 的 token 输入。Decode 还接收位置、mask 和 56 个历史 K/V 张量；两图均返回 logits 及 28 层 K/V，共 57 个张量。每个 K/V 为 `[1,8,256,128]`。运行时使用父目录 `tokenizer/` 中的官方分词器，pad token ID 为 151643；manifest 的 export.layout 中 pad=0 是原导出示例的输入准备参数，不是图中固化的 token 值。

## 下载

在仓库根目录运行：

```bash
git lfs install --local
git lfs pull --include="assets/qwen3-0.6b/kv-onnx/*,assets/qwen3-0.6b/tokenizer/*" --exclude=""
```

## 运行

从仓库根目录、在固定 `.venv-llm` 环境中执行。输出目录必须是新目录。

```bash
.venv-llm/bin/python scripts/run_inference.py \
  --kv-cache --model assets/qwen3-0.6b/kv-onnx/prefill.onnx \
  --decode-model assets/qwen3-0.6b/kv-onnx/decode.onnx \
  --tokenizer-dir assets/qwen3-0.6b/tokenizer \
  --output-dir output/kv-published-run --cores 4 \
  --matmul-policy blocked_fma --matmul-tile 2x4 --matmul-rhs-layout column-panel \
  --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
  --prompt 'The capital of France is' --max-new-tokens 4
```

## 重新打包

先通过 `scripts/export_qwen3_kv_onnx.py` 导出两个图，再将它们整理到新的发布目录：

```bash
.venv-llm/bin/python scripts/package_qwen3_kv_onnx.py \
  --source-dir output/kv-qwen3-full-20261110 --output-dir output/kv-onnx-package
```

打包器只修改 external-data 的文件名和偏移。相同名字、类型、形状及 SHA-256 的张量共享一份存储；全部原始张量字节和图中运算保持一致。它会重新读取保存后的图和分片，核对所有外置张量，并生成带文件校验和的 manifest。原始导出不被修改。

本次发布分片已核对全部外置张量字节及图结构；使用 `The capital of France is` 的 Prefill 和首步 Decode，各 57 个 ONNX Runtime 输出与分片前完全一致。

完整模型的既有数值和四核 QEMU 验证范围见 [验证说明](../../../probes/w6_kv_cache/README.md)；普通 PR 的 KV CI 仍执行小模型验证。
