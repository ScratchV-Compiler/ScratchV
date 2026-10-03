# W2 Runtime → Model 单步接口探测

把已实现的 Tokenizer、输入准备、模型调用、最后有效位置选择及 greedy 解码真正串起来。模型使用 `transformers==4.51.3` 官方 `Qwen3ForCausalLM`，两层、hidden=32、FFN=96、Q/KV heads=4/2、head_dim=16、随机种子 0、绑定 embedding/LM head。保留**完整模型词表 151936**；官方 Token ID 直接进入模型，不取模、不重新编号、不裁剪词表。

这是 W2 的接口集成证据；权重随机，不代表预训练 Qwen3-0.6B 的语言能力。执行单步前向与一个 greedy token，不执行生成循环、KV cache 或 QEMU。

## 本地复现

使用仓库 [固定 CPU 环境](../../requirements/qwen3-small-probe.txt)，Python 3.12。首先安装对应 CPU Torch，再安装固定依赖。Tokenizer 资产复用 [W2 Runtime 探测](../w2_runtime/README.md)下载的目录，或固定模型源目录；每次运行都会核验 [W1 manifest](../w1_qwen3_export/manifest.json) 的七个资产 SHA256，只读取小文件，不加载预训练权重。

```bash
python probes/w2_runtime/run.py --mode download --tokenizer-dir output/qwen3-tokenizer --output-dir output/w2-runtime
python probes/w2_runtime_model/run.py --tokenizer-dir output/qwen3-tokenizer --output-dir output/w2-runtime-model
```

输出目录必须不存在或为空。既有证据不会被覆盖；版本不匹配、资产缺失/损坏、数值或采样不一致、报告写入失败均返回非零。

## 验收内容

- 两个固定原始文本：中文混合文本，以及包含官方特殊 token 和 `endoftext` 的输入。后者包含 pad ID，仍必须按原始 token 数计算有效长度。
- 生产 `Qwen3Tokenizer` 的编码与官方 fast、slow Tokenizer 相同。生产 `prepare_inputs` 的 ID、dtype、shape、因果 mask 和 padding 由独立标量循环核验。
- 只导出普通 logits ONNX：INT64 `[1,256]` input_ids、FP32 `[1,1,256,256]` attention_mask，FP32 `[1,256,151936]` logits；FP32、opset 18、固定形状、无 KV cache、eager attention。
- Torch、ORT、共享 IR `none` 和 `all` 真正执行相同输入。ORT 对 Torch、IR 对 Torch 和 ORT 均检查所有位置，含 padding，严格要求有限 FP32、shape 完全相同、`max_abs < 1e-5`、`rtol=0`。
- 每条路径选择 `valid_length - 1` 的 logits，greedy ID 必须与独立索引的 Torch argmax 相同；解码必须与官方 fast、slow 相同。即使精度误差小于阈值，argmax 改变仍会失败；若抽中模型词表的未定义 Tokenizer 行，明确失败，不能修改候选 mask 来掩盖。

## 输出及资源

`report.json`、`report.md`、`report.html` 包含结果、阶段、首次失败、源码 SHA256、实际 checkout、环境版本、资产 SHA256、模型参数与权重 hash、输入 token IDs/hash、ONNX hash、全/有效/padding 误差、最差位置、greedy ID/解码/top-2 margin、构建/导出/解析/各引擎执行耗时。JSON 在完整写入并关闭后原子发布；报告生成失败不能留下 PASS。

无 `.git` 的源码快照也可复现：checkout HEAD 明确标为未知，身份由运行前记录的源码 SHA256 确定，不借用上层仓库的 Git 信息。报告同时记录实际导入的 parser、IR interpreter、Tokenizer 和输入模块的 `__file__`；若这些模块来自当前源码树之外则失败，防止误用旧安装。

报告中的引擎 `seconds` 仅计该次主机前向调用；总 `seconds` 还包含环境加载、资产校验、导出、报告前的数据保存与数值比较。都不是 QEMU 时间，也不是目标硬件性能数据。

`model.onnx`、`config.json`、每个文本的 `*.inputs.npz`、`*.torch.npy`、`*.ort.npy` 供独立复核。一个完整输出约 148.4 MiB；参考输出保存后通过 mmap 顺序读取，ORT 会话在 IR 前释放，各次完整输出按顺序释放，比较按 8 个 token 的块进行。两种参考、两个文本的持久化输出合计约 594 MiB，另加约 20 MiB 模型；预留至少 1 GiB 磁盘和数 GiB 空闲内存。这里没有声称限制了解释器内部内存峰值。

负例回归：

```bash
python -m pytest tests/test_w2_runtime_model.py -q
```

负例覆盖 ID 映射、padding/因果 mask、错误行选择、argmax 翻转、非法采样 ID、非有限/错误 shape/dtype、只在 padding 超差、解码差异及报告失败。单测不冒充真实模型运行；必须同时执行上面的集成 CLI 才能得到接口集成 PASS。
