# 真实 Qwen3 结构的两层数值探测

此探测直接使用固定版本 `transformers==4.51.3` 的[官方 Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)，参照[固定版本 Qwen3-0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)缩小尺寸并生成固定种子的随机权重，验证 **PyTorch → ONNX Runtime → ScratchV IR 解释器** 的数值一致性。无需下载 Qwen3-0.6B 权重或访问 Hugging Face。

与 W1 手工拼接的两层图不同，这里保留真实 Qwen3 的 Q/K RMSNorm、全 head_dim RoPE、GQA、SwiGLU 和 residual 路径。随机权重仅用于检查结构和计算，不具备语言生成能力；通过该探测也不表示完整 0.6B 模型或 RISC-V 后端已经验证通过。

## 固定配置

| 项目 | 值 |
|---|---|
| Python / PyTorch / Transformers | 3.12 / 2.7.1 CPU / 4.51.3 |
| 层数 / hidden_size / intermediate_size | 2 / 32 / 96 |
| Query heads / KV heads / head_dim | 4 / 2 / 16 |
| 词表 / batch / 序列长度 | 128 / 1 / 256 |
| 数值类型 / 随机种子 | FP32 / 0 |
| Attention / KV Cache | eager / 关闭 |
| RMSNorm epsilon / RoPE theta | `1e-6` / `1e6` |

`hidden_size=32` 与 `query_heads × head_dim=64` 刻意不相等，以覆盖 Qwen3 的独立 head_dim 配置。不要假设投影宽度必须等于 hidden_size。

ONNX 输入为 `input_ids: INT64 [1,256]` 与 `attention_mask: FP32 [1,1,256,256]`。mask 是加性因果和 key-padding 掩码（允许位置为 0，屏蔽位置为 FP32 最小有限值），不是 tokenizer 常用的二维 0/1 mask；位置编号固定为 `0..255`。普通图输出为 `logits: FP32 [1,256,128]`。

## 本地运行

先确认用于建环境的 `python` 是 Python 3.12。下面是 PowerShell 命令；Linux/macOS 可将 `& $probePython` 换成 `output/qwen3-probe-venv/bin/python`。

```powershell
python -m venv output/qwen3-probe-venv
$probePython = ".\output\qwen3-probe-venv\Scripts\python.exe"
& $probePython -m pip install --upgrade pip
& $probePython -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
& $probePython -m pip install -r requirements/qwen3-small-probe.txt
& $probePython -m pip install --no-deps -e .
& $probePython -m pip check
& $probePython probes/w2_qwen3_small/run.py --output-dir output/qwen3-small
& $probePython -m pytest tests/test_qwen3_small_probe.py tests/test_qwen3_small_model.py tests/test_qwen3_small_gate.py -q
```

CPU PyTorch 必须在安装依赖文件前单独安装；随后 `torch==2.7.1` 会保留已安装的 `2.7.1+cpu`。不要只执行依赖文件安装而让 pip 从默认索引另选 CUDA 依赖。导出相关包采用已验证的固定组合，避免环境升级改变导出图或数值基线。

`--output-dir` 必须不存在或为空。复跑时指定新的目录，例如 `--output-dir output/qwen3-small-run02`；脚本拒绝覆盖已有报告和模型。默认模型权重 seed 为 0，可通过 `--model-seed` 显式更改，所用值和权重哈希会写入报告。

## 验证与产物

所有观察点必须形状、dtype 一致且数值有限，**最大绝对误差严格小于 `1e-5`，`rtol=0`**。完整 256 个位置都参与验收，包括 padding query 的输出；JSON 同时分别记录有效 token 和 padding query 的误差。任何比较或执行失败都会使最终退出码非零。

固定的 7 组输入如下。输入 token 的 seed 与模型权重 seed 独立；短输入右侧以 token 0 补齐，所有输入均保持 `[1,256]`。

| Case | 有效长度 / 输入来源 | 检查目的 |
|---|---|---|
| `full_seed_0` | 256 / seed 0 | 全长基本用例 |
| `full_seed_42` | 256 / seed 42 | 另一组固定随机输入 |
| `one_token` | 1 / seed 7 | 最短有效前缀、大量 padding |
| `short_17` | 17 / seed 7 | 短输入与右侧 padding |
| `short_255` | 255 / seed 7 | 接近最大长度的边界 |
| `changed_future` | 256 / 修改 `full_seed_0` 的第 64 个位置及以后（从 0 计数） | 三种执行路径的前 64 个位置应保持一致，检查因果性 |
| `changed_padding` | 17 / 将 `short_17` 的 padding token 改为 seed 99 的随机 token | 三种执行路径的前 17 个位置应保持一致，检查 padding 隔离 |

不变性检查同样使用严格绝对误差阈值。每层还用独立 NumPy 计算 KV head 重复、attention probabilities 和 context，并确认被 mask 屏蔽的位置没有 attention 概率泄漏。

官方模型的临时 hooks 和调用原始函数的 eager-attention 观察器收集 **29 个 checkpoint**；每组输入也直接运行一次没有观察器的官方模型，核对观测是否改变 logits。观察点按以下顺序用于定位首次偏差：

- `token_embedding`
- `layer_0.` 和 `layer_1.` 各自的 `input_norm`、`q_norm`、`k_norm`、`v_proj`、`rope_q`、`rope_k`、`attn_probs`、`attn_context`、`attn_output`、`attn_residual`、`post_attention_norm`、`mlp`、`residual`
- `final_norm`、`logits`

ScratchV 当前只返回一个 IR 输出。为在一次执行中观察所有 checkpoint，诊断图把它们展平后拼成第一输出 `trace_pack`，同时保留原始命名输出供 ORT 独立核对。`checkpoints.json` 定义每段的名字、shape、offset 和 size，解释器结果按此表拆回各张量。

普通 `model.onnx` 只公开 logits；`diagnostics.onnx` 公开 `trace_pack` 和 29 个命名观察输出。**两张图都必须经 ORT 和真实 IR 解释器运行通过**，同时核对普通图与诊断图的 logits 一致，避免诊断机制掩盖普通执行路径的问题。

默认输出目录为 `output/qwen3-small/`：

| 文件 | 用途 |
|---|---|
| `model.onnx` | 普通模型图，单个 logits 输出 |
| `checkpoints.onnx` | 官方模型导出的 29 个命名输出图 |
| `diagnostics.onnx` | 增加第一输出 `trace_pack` 的诊断图 |
| `checkpoints.json` | 打包张量各段的名称、形状和偏移 |
| `config.json` | 实际使用的完整官方模型配置 |
| `inputs_*.npz` / `logits_*_*.npy` | 各 case 输入及 PyTorch、ORT、IR 的 logits |
| `report.json` | 可机器读取的配置、逐层比较结果和失败详情 |
| `report.md` | 可直接阅读的数值报告 |
| `report.html` | 可离线打开、展开各 case 的逐层误差报告 |

定位失败时先看 `report.md` 的 case 与 First divergence，再在 `report.json` 中查看该 case 的 `pytorch_vs_ort` 或 `ir_vs_ort`：`first_divergence` 指向上述顺序里第一个未通过的观察点；对应条目包含最大误差、最差元素下标、双方数值以及形状、dtype、有限值结果。这是“最早观测到的偏差”，不直接等同于根因算子。解析或执行异常会记录 `stage`、`current_case`（若已进入 case）和错误信息；不变性或其他辅助检查的失败详情也在 JSON 中。

模型和诊断结果均为生成产物，保存在已忽略的 `output/` 下，不提交 Git，不走 Git LFS。报告还包含环境版本、模型与输入哈希和源码信息，便于复现。

## CI

`.github/workflows/llm-deploy.yml` 在 PR、定时及手动运行中执行 `numeric:ir-qwen3-small`。它创建独立 CPU 环境，执行完整探测及专项测试，将报告写入 Actions Summary，并上传 `qwen3-small-probe-output` artifact；失败运行中已产生的报告同样保留。

现有 `probe:small-transformer` 继续使用 PR #89 提交的 LFS 模型执行原来的数值验收。通用编译器 CI 不安装 PyTorch；涉及官方模型导出和执行的重型验证由本专用工作流承担。

专项测试按依赖区分：`test_qwen3_small_probe.py` 检查打包、拆包和数值诊断，仅需 NumPy/ONNX；`test_qwen3_small_model.py` 检查官方模型、配置及观察器；`test_qwen3_small_gate.py` 检查完整门禁的错误传播。后两者使用 PyTorch，在通用轻量环境可跳过，在专用 CI 的固定环境中执行。
