# W6 Attention 与 KV 生成测试

从仓库根目录运行，使用已安装 `.[llm-deploy]` 的 Linux Python 3.12 环境。测试输入由 NumPy 构造，不需要完整模型权重。大部分测试使用本机 C 编译器和 pthread；`test_attention_kv_integration.py` 还包含一项真实 4 核 QEMU 小图测试，需要 RV64 编译工具链和 `qemu-system-riscv64`。

```bash
python -m pytest tests/llm-deploy/w6 -q
```

| 测试内容 | 怎样算通过 |
| --- | --- |
| 两个查询头直接读取同一组 K | 长度为 1 和 255 时，每个头、每个位置的分数都与按位置构造的参考值一致。 |
| 连续追加两个位置的 K、V | 第一步只读取 255 个位置，第二步读取 256 个位置。原转置实现、直接读 K 和按头分工得到的上下文均符合相同的手算参考，绝对容差为 2×10⁻⁵，相对容差为 3×10⁻⁵。 |
| 工作区不足、无效输入和计算溢出 | 工作区少 1 字节返回状态 1。第二步 V 含 NaN，或最后一组的有限 Q、K 相乘溢出，均返回状态 2，两步的输出都保持调用前的值。修正输入后，同一工作区可再次使用。 |
| V 加权求和的 FP32 顺序 | 专门构造会发生舍入抵消的输入，核对原实现和按头分工都是每 128 个位置先累加再合并，所有输出与预期逐位一致。 |
| KV Decode 图接入 | `test_attention_kv_integration.py` 使用 16 个 Q 头、8 个 K/V 头、128 维和 256 位置。有效长度 1/5/17/255/256 的上下文及全部 K/V 输出符合 ORT，优化前后逐位一致；屏蔽位置、无效输入、溢出和失败后复用也有检查。共享中间结果或其他形状沿用原实现。 |

只运行包含工作区和第二步溢出检查的完整 Attention 用例：

```bash
python -m pytest tests/llm-deploy/w6/test_attention_decode.py -k normalizes_the_valid_prefix -q
```

只运行 KV Decode 图接入检查，包含一项真实 QEMU 小图：

```bash
python -m pytest tests/llm-deploy/w6/test_attention_kv_integration.py -q
```

## KV 生成命令行与冻结用例

相关 CLI 测试沿用 [`tests/test_kv_cli.py`](../../test_kv_cli.py)，不启动模型、编译器或 QEMU。使用小型 tokenizer 和模拟的 QEMU 会话，执行真实 token 循环：三条用例各生成 8 个 token，共用同一 ELF、权重和会话；每例重置缓存，执行 1 次 Prefill 和 7 次 Decode。

检查全部词表分数的最大绝对误差严格小于 `ATOL`（等于也失败），每步 greedy token、输入历史、最终文字和停止原因均符合冻结用例。还检查三例缺失或预算改变、额外前向、超时、取消、退出与报告保存失败；任一失败不能报告整组三例通过。

默认单核、手动/自动核数及旧 `--harts` 继续可用。旧构建缺少新增策略字段时按 `original` 读取；复用构建时拒绝不同的策略。`--attention-decode-policy kv-heads` 只传给 Decode，Prefill 保留原实现；`--lm-head-policy vocab-blocked` 传给两张图，仍只构建一个 ELF/权重包。

```bash
python -m pytest tests/test_kv_cli.py -q
```

旧 Python 输出接口的兼容检查在 `tests/test_kv_multi_output.py`：
ABI v1 使用 `dataclasses.replace(artifact, output=...)` 时采用新输出；
ABI v2 使用有序的 `outputs`，并保持 `dataclasses.asdict()` 的结果可读取。
`tests/test_tensor_c_lm_head.py` 另检查：当转置结果本身也是模型输出时，分块优化不能删掉它。

```bash
python -m pytest tests/test_kv_multi_output.py tests/test_tensor_c_lm_head.py -q
```

普通生成入口和报告保存的相关检查：

```bash
python -m pytest tests/test_w5_inference_cli.py tests/llm-deploy/w5/test_report_persistence.py -q
```

无 KV 和 KV 路径都记录每例 `generation_and_validation_s`，包含生成、ORT、数值比较及最终文字/停止原因检查，排除编译、保存报告和终端显示。`ort_s`、`numeric_comparison_s`、`result_validation_s` 是其中各部分，不能再与总计相加。KV 会话启动在每例计时外，另列 `session_start_s` 和首例含启动的 `first_case_with_session_start_s`；`resident_session_s` 包含启动至退出。`checkpoint_save_s` 只统计已完成的中间报告保存，不包含最后一次写入。

## 短提示词的 Prefill

`test_prefill_contract.py` 检查短 Prefill 的输入长度、因果 mask、256 位置的缓存输出，以及构建记录与实际输入大小是否一致。旧构建未记录长度时仍按 256 处理；提示词超出短图的长度时拒绝执行，不截断输入或修改会话缓存。

```bash
python -m pytest tests/llm-deploy/w6/test_prefill_contract.py -q
```

`test_prefill_export.py` 使用固定随机权重的两层 Qwen3 小模型，对比 256 位置图与 5 位置图。核对每个有效位置的全部词表分数、各层有效 K/V，以及接上原 Decode 后两步的结果；短图补齐的缓存尾部必须为零。另检查 Dynamo 导出的短图能通过 ORT、ONNX 解析和 Tensor-C 代码生成。无需下载完整模型。

```bash
python -m pytest tests/llm-deploy/w6/test_prefill_export.py -q
```

真实四核 QEMU 的交替计时与数值比较使用 `probes/w6_performance/prefill.py`；这是小图性能实验，运行方式和实际结果见 [W6 README](../../../docs/llm-deploy-v1.0/W6/README.md)。单测通过不代替实际 QEMU 测量。
