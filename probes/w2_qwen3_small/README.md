# 真实 Qwen3 结构的两层数值探测

本目录提供两个独立验收入口：`run.py` 验证 PyTorch/ORT/IR；`riscv.py`
继续将同一普通图和诊断图经 ScratchV 编译流程生成张量 C 内核，再交叉编译成
RV64GC ELF，在 QEMU `virt` 裸机执行。下文“不是 RISC-V 验证”的说明仅指
`run.py`；RISC-V 的复现步骤和边界见最后一节。

此探测直接使用固定版本 `transformers==4.51.3` 的[官方 Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)，参照[固定版本 Qwen3-0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)缩小尺寸并生成固定种子的随机权重，验证 **PyTorch → ONNX Runtime → ScratchV IR 解释器** 的数值一致性。无需下载 Qwen3-0.6B 权重或访问 Hugging Face。

与 W1 手工拼接的两层图不同，这里保留真实 Qwen3 的 Q/K RMSNorm、全 head_dim RoPE、GQA、SwiGLU 和 residual 路径。随机权重仅用于检查结构和计算，不具备语言生成能力；只通过 `run.py` 不表示 RISC-V 路径通过，两入口均通过也不能替代完整 0.6B 模型验收。

复现 PR #91 本次更新时，按 [W1 独立复现指南](../../docs/llm-deploy-v1.0/W1/README.md) 获取并核对 PR 的精确 head SHA。历史 `5ea22ec` 不含本次新增的环境预检和时长汇总，不能用旧提交的成功记录代表新代码结果。

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

## Linux 复现

正式交付使用 Ubuntu 24.04 x86_64 / Bash / Python 3.12。通用工具准备见 [Linux 复现约定](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md)；在仓库根目录执行：

```bash
set -euo pipefail
python3.12 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/qwen3-small-probe.txt
python -m pip install --no-deps -e .
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python probes/w2_qwen3_small/run.py --output-dir output/qwen3-small
python -m pytest tests/test_qwen3_small_probe.py tests/test_qwen3_small_model.py tests/test_qwen3_small_gate.py -q
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

## 编译优化与 RISC-V/QEMU 验收

执行链路为：

```text
官方 Qwen3 → ONNX → ONNXParser → shared Program IR
  → CompilerDriver / 优化 pass / IR verifier
  → tensor-c（显式张量循环、FP32、静态 arena 与权重）
  → Zig 0.14.1 / LLVM → RV64GC LP64D ELF
  → QEMU virt / TCG → UART 原始输出 → ORT 逐点对照
```

这里实际执行的是 RISC-V 指令。Zig/LLVM 承担 C 到机器码的交叉编译，
ScratchV 承担 ONNX/IR、IR 优化、张量循环生成、内存规划与输入输出接口。
这条新路径不经过旧的标量寄存器选择器，不表示该选择器已经实现张量计算；
标准编译入口现在会拒绝把张量图交给旧 `riscv/llvm` 后端，以免生成占位汇编。

Linux 工具准备（Python 导出依赖仍按前面的固定版本安装）：

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends qemu-system-misc
python -m pip install ziglang==0.14.1
export SCRATCHV_CC="$(python -c 'from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent / "zig")')"
```

设置 `SCRATCHV_QEMU="$(command -v qemu-system-riscv64)"`；所有工具均为 Linux 可执行文件。脚本不会隐式联网安装工具。

先验证矩阵乘，再导出模型并运行完整验收；输出目录必须为空，避免覆盖旧证据：

```bash
python probes/w1_matmul_4x4/run.py --output-dir output/qemu-matmul
python probes/w2_qwen3_small/run.py --output-dir output/qwen3-small
python probes/w2_qwen3_small/riscv.py \
  --model-dir output/qwen3-small --output-dir output/qwen3-riscv
```

第三步对普通图、诊断图均执行 `none/basic/all` 优化后的 IR 数值验收；并分别
将 `none/all` 编译为独立 ELF，运行全部 7 组输入。因此一轮有 4 次模型构建、
28 次 QEMU 执行，涵盖全部 29 个检查点、普通/诊断 logits 一致性、因果性与
padding 隔离。每个比较仍要求 FP32、形状一致、有限值、最大绝对误差严格
小于 `1e-5`，不提高容差。独立矩阵乘探测包含 4×4 小数、投影、Attention、
双侧批次广播及三种向量乘法。

进入工具链前，RISC-V 入口先核对上游 PyTorch/ORT/IR 成功报告、普通/诊断 ONNX
哈希、checkpoint schema 和 7 组输入指纹。`export_evidence` 保存对应证据；上游
失败、输入目录混用或哈希不一致应直接失败，不能只靠当前 ORT 比较通过。该绑定
用于避免意外混用产物，不是第三方认证；参考计算仍须在指定环境重新生成。

当前执行环境与完整源码归属的报告增强单独见 [W1 PR #93](https://github.com/ScratchV-Compiler/ScratchV/pull/93)。
W2 分支未包含该报告代码修复；未合并前不能把原导出环境当成本次 QEMU 对照的执行环境。

`output/qwen3-riscv/` 包含 `report.json/md/html`、生成的 C、启动汇编与链接脚本、
ELF、编译器/QEMU 命令与日志、每次执行的输入和 UART 二进制以及 QEMU/ORT 数组。
JSON 记录源码/模型/输入/ELF 哈希、工具版本、优化统计、工作区大小、最差元素与
首次发生偏差的检查点。缺工具、编译失败、trap、超时、损坏/截断的 UART、错误
退出或任一数值不匹配都会失败，不允许跳过后返回成功。

编译器或 QEMU 的日志文件保存失败时，仍分别尝试保存 stdout 和 stderr。
已有的编译错误、guest 状态、协议错误或超时保持为主要错误，日志保存错误另行附加；
执行成功但必需日志无法保存时，入口仍返回失败，避免把不完整证据当作完整验收。

### 查看执行时长

使用上面的 `--output-dir output/qwen3-riscv` 命令即可生成时长报告，无需另开计时
模式。直接打开 `output/qwen3-riscv/report.md` 或 `report.html`；自动分析读取
同目录 `report.json`。每次复跑使用新的空输出目录，保留不同运行的原始证据。

| 字段 / 展示 | 含义 |
|---|---|
| `report.timing.pipeline_seconds` | RISC-V 探测流水线墙钟，包括产物核验、IR/ORT 对照、编译和 QEMU 等阶段，不含最终报告渲染 |
| `report.timing.cross_compile_seconds_total` | 仅汇总已记录的成功交叉编译，不含 ScratchV 解析/优化/生成 C 或失败构建耗时；另看构建完成数 |
| `report.timing.qemu_process_wall_seconds_total` | 已记录的 QEMU 进程墙钟汇总，不含独立的编译时间 |
| `report.timing.groups` | 按 normal/diagnostic、none/all 区分普通/诊断输出和优化级别，不能把不同工作量混为同一前向性能 |
| 逐次 `qemu_process_wall_seconds` / `seconds` | 本次 QEMU 进程实测墙钟；`seconds` 为相同计时口径的兼容字段 |
| 逐次状态 | `success`、`numeric_failed`、`runtime_error`、`timeout`、`not_started`；失败保留已测时长，无测量为 null |
| `not_attempted_count` | 尚未尝试的计划项数；区别于 `not_started` 的已尝试但 QEMU 未启动，二者均非零秒成功 |

QEMU 进程墙钟包含启动、guest 计算、输出打包/UART 和退出；超时路径还包含子进程清理，
不含 host 输入准备和输出解码，**不是纯模型 forward 耗时**。流水线包含额外对照工作，不能把它称为 QEMU 推理时间。
首次 Zig 缓存、系统负载、工具版本和诊断图较大的输出都会影响测量。当前记录只用于
观察与后续定位，不新增性能通过阈值，也不据此宣称优化加速或目标硬件性能。
成功与失败样本分别统计，7 种输入不是同一输入的重复采样。

Markdown/HTML 在数值结果旁展示汇总、分组与逐次耗时；CI 将相同 Markdown 写入
Actions Summary，并将 JSON/HTML 和已生成的失败报告保存在
`riscv-tensor-probe-output` artifact。失败时同时核对实际执行次数、状态与日志，
不能将不完整的一轮写成 4 构建、28 次全部通过。

### 仅生成 C 与当前边界

只生成张量 C 可使用标准入口：

```bash
python -m scratchv.main model.onnx --backend tensor-c --optimize all \
  --verify-ir --tensor-workspace-mib 256 -o model.c
```

`--verify-ir` 检查编译流程的 IR 契约；执行数值验收须运行上述 QEMU 探测。
旧 `--verify` 不会执行这份生成代码，因此新后端拒绝该选项。

本次优化器修复保持可执行语义：常量折叠遵循 dtype 的逐步舍入；peephole 仅
消除安全的恒等运算，并重定向 SSA 使用；现有 MulAddFusion 因缺少合法融合
opcode 暂时不执行融合。`all` 仍运行注册的全部 pass，融合次数为零。

未使用算子的数值错误是否属于优化必须保留的可观察行为，仍待团队在接口 D8 中
决定；当前存在 DCE 删除无用除零后错误不再出现的差异，本次 Qwen 探测未触发它。
现象、候选契约和后续处理见 [收尾与审查报告](../../docs/llm-deploy-v1.0/W1/W1-本地收尾与审查报告.md)，
不能因为本探测通过就宣称所有失败行为已在不同优化级别间等价。

当前边界：静态形状、连续张量、FP32 计算和 INT32/INT64 索引；固定单函数直线
IR；静态 arena 按 SSA 最后使用复用，函数不可重入。权重嵌入只读段，中间张量
不放栈。当前默认工作区上限 256 MiB、QEMU 内存 512 MiB。本探测使用小尺寸随机
权重，不覆盖完整 0.6B 权重加载、Tokenizer、生成循环、KV Cache 或性能加速。
数学函数使用 Zig 随附的 musl libm，而非粗略近似；关闭 fast-math 和浮点乘加融合。

CI 新增强制 `probe:qemu-matmul`、张量编译器/运行时回归与
`numeric:qemu-qwen3-small`，输出上传为 `riscv-tensor-probe-output`。
旧 IR verifier benchmark 改为输出优化后 IR，继续比较验证开关的影响；
它不再生成不可执行的 CNN 占位汇编，计时口径不可与旧版直接比较。
