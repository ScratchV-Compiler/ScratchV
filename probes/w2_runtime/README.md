# W2 Host 运行时验收

`unit:runtime` 验证文本 tokenizer、固定输入准备、最后有效位置 greedy 采样及停止边界，不执行完整模型或生成循环。生产模块只依赖 NumPy 与可选的 `tokenizers`；Transformers 只用于此门禁的官方对照。

## 固定资产与环境

模型为 `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`。复用 [W1 manifest](../w1_qwen3_export/manifest.json) 的七个 `source_files` 哈希：配置、tokenizer JSON、merges、vocab 与许可证，合计约 15.2 MiB。下载模式只获取这些文件，不下载模型权重。已有资产必须全部匹配哈希；损坏目录保留并失败，不静默覆盖。

使用 [两层模型指南](../w2_qwen3_small/README.md) 中的 Python 3.12 固定 CPU 环境；其中已包含 `numpy==2.2.6`、`tokenizers==0.21.4`、`transformers==4.51.3`。仅使用生产 tokenizer 时可安装 `pip install -e ".[llm]"`，无需 Torch；复现 gate 仍使用固定版本和官方参考依赖。

正式复现使用 Ubuntu 24.04 x86_64 / Bash；先按 [Linux 复现约定](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md) 激活环境，在仓库根目录运行：

```bash
.venv-linux/bin/python -B -X utf8 probes/w2_runtime/run.py \
  --mode download --tokenizer-dir output/qwen3-tokenizer --output-dir output/w2-runtime
```

已有七个文件时使用 `--mode verify`，该模式不访问网络。输出目录必须为空或不存在，每次复现选择新的目录。CLI 退出 0 且本次 `report.json` 的 `passed=true` 才代表通过。报告写入不完整、依赖缺失、下载或哈希校验失败均为失败。

## 验收内容

- 固定 20 组语料涵盖中英混合、空白、换行、代码、数字、CJK/RTL、emoji、组合字符和特殊 token；保存固定 IDs 与解码文本作为 golden corpus。
- 每组均对照官方 `Qwen2TokenizerFast` 和独立 Python `Qwen2Tokenizer`，覆盖默认参数、`add_special_tokens`、`skip_special_tokens` 与空格清理组合。官方 JSON 的 NFC 规范化会把 `e` 加组合重音转换为 `é`；“100%”指与官方输出完全一致，不是任意原始 Unicode 字节可逆。
- 分开核对基础词表 151643、含 added tokens 的词表 151669、模型输出维度 151936；模型中空余行不等于可解码 ID。未知 ID 报错，不丢弃；不静默屏蔽 logits 或改变 greedy。
- tokenizer EOS 为 151645，generation EOS 为 `(151645,151643)`，pad 为 151643。普通文本编码不注入 BOS/EOS，不渲染聊天模板。
- 检查有效长度 1/17/255/256、右 padding、INT64 IDs、FP32 加性 mask、最后有效位置和 greedy 平局。mask 仅当 `key <= query` 且 `key < valid_length` 时为 0，其余为 FP32 最小有限值。
- 验证生成 EOS、256 容量和 token 预算停止。空 prompt、超长输入、非法 ID、非有限 logits 等反例在单测中必须失败。

## 生产接口

```python
from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer
from scratchv.runtime.llm_inputs import prepare_inputs, greedy_next_token

tokenizer = Qwen3Tokenizer.from_directory("output/qwen3-tokenizer")
ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
inputs = prepare_inputs(ids, pad_token_id=tokenizer.pad_token_id,
                        vocab_size=tokenizer.model_vocab_size)
# outputs = your_verified_model_runner(inputs.as_feed())
# next_id = greedy_next_token(outputs, inputs.valid_length,
#                             vocab_size=tokenizer.model_vocab_size)
# tokenizer.validate_token_ids([next_id])
```

`generation_stop_reason` 供后续生成循环在前向前判断 EOS/容量/预算。模型输出仍按原 ABI 完整校验，最后有效位置选择是 host 操作。这里未实现聊天模板、多轮历史、完整模型前向或端到端文本生成；也不能将官方 tokenizer 的 IDs 输入词表只有 128 的随机小模型。

## 报告与 CI

`report.json/md/html` 保留逐语料结果、对照次数、词表/特殊 ID、资产与执行源码 SHA256、checkout、环境、阶段及错误。报告目录不含模型或 tokenizer 大文件；失败保留已完成的记录，不把未执行项计作通过。

LLM workflow 的 `unit:runtime` 在固定环境运行，下载固定的七个资产，随后全程离线调用官方参考；`w2-unit-gates-output` artifact 保存报告和单测 XML。`tests/test_llm_inputs.py`、`tests/test_qwen3_tokenizer.py` 和 `tests/test_w2_runtime_*.py` 覆盖输入、适配层和门禁的反例。具体运行结果见 [W2 交付记录](../../docs/llm-deploy-v1.0/W2/README.md)。
