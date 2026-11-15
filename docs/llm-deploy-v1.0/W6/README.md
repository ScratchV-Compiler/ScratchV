# W6 · 文本生成性能优化

[详细实验报告](EXPERIMENTS.md) · [完整模型数据](reports/kv-cache-integration-4core.json) · [短 Prefill 数据](reports/prefill-short-4core.json)

## 本分支增加了什么

在 PR121 已有 KV cache 和常驻 QEMU 的基础上，W6 增加两项可选优化：LM Head（词表输出层）按词表分块读取权重；单 token Attention 直接读取 K，按 K/V 头分工，减少转置和中间数据读写。模型仍返回完整词表分数（logits）。

另支持固定长度的短 Prefill：提示词只有 5 个 token 时，可以只计算 5 个位置，再接原来的 Decode。该功能目前完成了两层小模型验证，尚未测量完整模型。

生成入口补齐 KV 模式三条固定用例的逐 token 校验，并修复旧 Python 输出参数和模型额外输出的兼容问题。默认仍为原实现、单核、256 位置 Prefill；保留 `--cores 1/2/4/auto` 和旧 `--harts` 用法。

## 实际复现结果

以下均使用 4 核、FP32、QEMU 8.2.2、工作区 padding 关闭。完整模型与小图分开测量，不混用加速比。

### 完整 Qwen3-0.6B：France 生成 8 个 token

| 实现 | 生成与校验耗时（秒，含启动） | 相对 KV 原实现 |
| --- | ---: | ---: |
| 无 KV，原实现 | 3726.334 | — |
| KV，原实现 | 436.024 | 1.000× |
| KV，W6 优化 | 346.226 | 1.259× |

W6 在已有 KV 基础上缩短耗时 20.59%；后续 7 次 Decode 的平均请求耗时由 16.558 秒降到 4.750 秒。无 KV 到 KV 的收益还包含缓存复用、单位置计算和 QEMU 常驻，不能全部算作 W6 优化。

优化版 France、Germany、Japan 三例共 24 步均通过：每步全部 151,936 个 logits 与 ONNX Runtime（ORT）比较，最大绝对误差为 **2.48×10⁻⁵**，小于 0.001；token、最终文字和停止原因符合固定参考。[完整数据](reports/kv-cache-integration-4core.json)

<details>
<summary>生成文字、计时范围与精度说明</summary>

| 用例 | 生成与校验（秒，不含首次启动） | 最大绝对误差 | 生成的 8 个 token 对应文字 |
| --- | ---: | ---: | --- |
| France | 332.726 | 0.000024796 | ` Paris. The capital of Italy is Rome` |
| Germany | 336.269 | 0.000023931 | ` Berlin. The capital of France is Paris` |
| Japan | 333.339 | 0.000024796 | ` Tokyo. The capital of the United States` |

- 三例均达到 8-token 上限后停止，未到自然结束；例如 Japan 的续写尚未成句。
- 上面的 France 对照包含本次启动、生成文字和 ORT 校验，不含前置编译、每例结束后的报告保存和最终终端显示。三例表在同一会话中运行，启动只发生一次，单独记录。无 KV 和原 KV 的冻结文字另在运行后核对；三组生成计时内都已完成文字转换。
- 原 KV 与 W6 的 France 全部 1,215,488 个 logits 逐位相同。无 KV 检查每步全部 256 行，最大误差为 0.000303984；它与 KV 输出不要求逐位相同。KV 只读回当前有效位置的 logits，此次完整生成没有逐步转储全部 K/V。
- 编译单独计时：原 KV 70.628 秒、优化版 65.409 秒；生成时复用各自构建。两项 W6 优化合用，尚未分别关闭其中一项测量贡献。
- 数据来自 `79d2063` 整合 W6 `811b86c` 及当时集成改动的实测，完整模型每组只跑一次。本次整理发布复用已有数据，版本、资源采样及逐步误差见数据摘要。

</details>

### 短 Prefill：两层小模型、5 个输入 token

相同随机权重和 Decode 图，各预热一次，再交替运行 6 对：

| 测量项 | Prefill 256 位置 | Prefill 5 位置 | 变化 |
| --- | ---: | ---: | ---: |
| 调用耗时中位数（毫秒） | 234.688 | 23.243 | 10.097× 加速 |
| 其中：等待 QEMU 完成（毫秒） | 232.089 | 21.182 | 10.957× 加速 |
| Prefill 工作区（KiB） | 2465.000 | 134.898 | 减少 94.53% |

调用耗时包含本机输入准备、QEMU 执行及 logits 读回，不含启动、重置、额外 K/V 转储和 ORT 比较。加速比使用两组中位数相除。

前缀长度 1–5 的全部 logits、各层有效 K/V，以及接着两步 Decode 均通过比较，最大绝对误差为 **0**；token 相同，未使用的缓存位置为零。**这只是两层小模型结果，不能用 10.097× 推算完整文本生成时间。** [逐次数据](reports/prefill-short-4core.json)

<details>
<summary>小模型规模与内存范围</summary>

模型为固定随机种子的两层 Qwen3，隐藏维度 32、词表 128；缓存容量保持 256。6 对调用加速比为 8.96–11.72×。

Prefill 与 Decode 复用工作区，所以会话实际分配由 2465.000 KiB 降为 323.570 KiB，降幅 86.87%；Decode 自身工作区不变。工作区大小不等于整个 QEMU 进程的内存峰值。

</details>

仍未完成：原独立 Attention 分数矩阵乘法只有 1.285×，未达到 1.5×；短 Prefill 的完整模型验证、长上下文测量尚未完成。`file-backed-ram` 文件映射内存目前仅适用于原无 KV 路径，尚未接入常驻 KV 会话。

## 整体怎么实现

提示词 → 编码 → Prefill → 选下一个 token → 复用 K/V 逐步 Decode → 转换为文字。

| 本机 Python | QEMU 中的 RISC-V 程序 |
| --- | --- |
| 编码输入、按最高分选 token、转换文字、用 ORT 校验 | 执行模型、保存并更新 K/V、返回当前位置的全部词表分数 |

完整三例每例执行 1 次 Prefill 和 7 次 Decode，例间重置缓存，复用同一 ELF、权重和 QEMU 进程。`vocab-blocked` 启用 LM Head 分块；`kv-heads` 只用于匹配的 Decode Attention，Prefill 保留原 Attention 实现。

短 Prefill 在导出时固定输入长度：5 位置图只计算 5 行投影和 MLP、5×5 Attention，返回时将 K/V 补零至 256，再接原 Decode。最多接收 5 个提示词 token，超长输入会报错；目前不自动按输入长度选图。关闭工作区 padding 不等于取消模型输入补齐。

<details>
<summary>耗时分析、历史实验与 PPT</summary>

完整 France 的 346.226 秒中，Prefill 占 72.87%，7 次 Decode 占 9.60%，ORT 占 13.56%，启动占 3.90%。后续优先验证完整模型的短 Prefill，再分析长文本 Decode 的入口检查。

[详细实验报告](EXPERIMENTS.md)保留分阶段耗时、优化原理、此前的 Dense / LM Head / Attention 对照及原始数据链接。[三页 PPT](W6-summary.pptx)同步概括本次功能、完整模型与短 Prefill 结果及复现入口。

</details>

## 如何复现

在仓库根目录、Python 3.12 环境运行。本次使用 Zig 0.14.1、QEMU 8.2.2；系统工具和模型准备见[环境说明](../LINUX_REPRODUCTION.md)。

```bash
python -m pip install -v -e ".[llm-deploy]"

# 无需下载完整模型；输出目录必须是新的。
python probes/w6_performance/prefill.py --output-dir output/prefill-short-repeat --cores 4 --pairs 6
```

小图命令完成导出、编译、数值检查和交替测量，输出 `report.json`、ORT 数组及两组 QEMU 日志。单独导出短 Prefill 使用 `scripts/export_qwen3_kv_onnx.py --prefill-length 5`；运行器识别图中长度，旧构建仍按 256 处理。

<details>
<summary>完整模型三例、原实现对照与构建复用</summary>

取得已有模型资产后，构建并运行优化版三例：

```bash
git lfs pull --include="assets/qwen3-0.6b/onnx/**,assets/qwen3-0.6b/kv-onnx/**,assets/qwen3-0.6b/tokenizer/**" --exclude=""

python scripts/run_inference.py --kv-cache \
  --model assets/qwen3-0.6b/kv-onnx/prefill.onnx \
  --decode-model assets/qwen3-0.6b/kv-onnx/decode.onnx \
  --tokenizer-dir assets/qwen3-0.6b/tokenizer \
  --verify-onnx assets/qwen3-0.6b/onnx/model.onnx --reference-suite \
  --cores 4 --matmul-policy blocked_fma --matmul-tile 2x4 \
  --matmul-rhs-layout column-panel --mean-policy two-four-lane-last-axis \
  --nonlinear-policy fp32-reference --lm-head-policy vocab-blocked \
  --attention-decode-policy kv-heads --timeout 1800 --total-timeout 14400 \
  --output-dir output/w6-kv-repeat
```

`report.json` 应记录三例通过、24 步数值检查及正常退出。此命令使用已发布的 256 位置 Prefill 资产。

- 复用构建：将 `--model`、`--decode-model` 替换为 `--reuse-build output/w6-kv-repeat/executable.json`，使用新输出目录，其余参数保持一致。只编译时加 `--build-only`。
- KV 原实现：两项优化参数都改为 `original`，重新构建。只跑 France：将 `--reference-suite` 换为 `--prompt "The capital of France is" --max-new-tokens 8`。
- 无 KV 对照及逐阶段测量范围见[详细实验](EXPERIMENTS.md#kv-integration)。

</details>
