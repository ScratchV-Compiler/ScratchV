# Qwen3-0.6B 固定验收资产

来源：`Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`。
来源文件和 ONNX 发布产物的唯一校验清单为
[manifest.json](../../probes/w1_qwen3_export/manifest.json)，许可见 [LICENSE](LICENSE)。

| 目录 | 内容 | 版本管理 |
|---|---|---|
| `onnx/` | 完整 28 层 FP32 ONNX 与三个 external-data 权重分片，约 2.24 GiB | Git LFS |
| `tokenizer/` | `tokenizer.json`、`vocab.json`、`merges.txt` | Git LFS |
| `tokenizer/` | 模型配置、生成配置、Tokenizer 配置与许可证 | 普通 Git |

ONNX 固定 batch=1、sequence=256、opset=18，无 KV cache。完整预训练权重已包含于
三个分片中；本目录不重复保存原始 Hugging Face `model.safetensors`，重新导出时另行获取。

Git LFS 的初始化和拉取方法见 [Linux 复现约定](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md)。
W2 验收入口只离线读取此目录并检查固定 SHA256；未拉取的 LFS pointer、缺文件和损坏文件均失败。
生成的小模型、参考输出和报告写入 `output/`，不提交到此目录。

更新资产时必须同时更新来源 revision、manifest 与相关验收证据，并保持 external-data
文件名与 ONNX 引用一致。LFS 中的每个权重版本独立占用存储，CI 按需拉取并缓存对象。
