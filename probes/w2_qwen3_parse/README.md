# W2：完整 Qwen3 ONNX 前端解析探测

入口 `probes/w2_qwen3_parse/run.py` 面向固定发布的完整 28 层 Qwen3-0.6B 图，验证 ScratchV 前端解析与结构、节点及权重绑定审计。使用 W1 已固定的 FP32、batch=1、L=256、无 KV Cache 的 ONNX 产物和 [manifest](../w1_qwen3_export/manifest.json)。

本入口不执行完整模型 IR，不生成完整模型 RISC-V，不运行 QEMU，也不重新导出模型或比较 PyTorch/ORT 数值。完整模型的 ORT 执行继续由 [W1 导出门禁](../w1_qwen3_export/README.md) 负责；两项门禁分别报告结果。

## 准备环境

正式复现使用 Ubuntu 24.04 x86_64 / Bash / Python 3.12，固定 NumPy 2.2.6、ONNX 1.18.0、ONNX Runtime 1.22.1 和 protobuf 5.29.5。可直接复用 W1 的固定环境；单独准备时无需安装 torch、Zig 或 QEMU。

Linux Bash：

```bash
set -euo pipefail
python3.12 -m venv output/qwen3-parse-venv
. output/qwen3-parse-venv/bin/activate
python -m pip install "numpy==2.2.6" "onnx==1.18.0" "onnxruntime==1.22.1" "protobuf==5.29.5"
python -m pip check
python -X utf8 -B probes/w2_qwen3_parse/run.py --mode verify \
  --model-dir output/qwen3-full-model --output-dir output/qwen3-parse --timeout 300
```

所有命令在仓库根目录执行。`--model-dir` 必填，应指向同目录保存 `model.onnx` 和三个真实 external-data 分片的完整产物，不能只有 ONNX 文件或 LFS pointer。`verify` 是默认模式，不访问网络；不存在完整模型时，将命令中的 `--mode verify` 改成 `--mode download`，由入口获取并校验 W1 固定发布产物。不要用新导出或其他 revision 的文件替换固定产物。

`--output-dir` 默认是 `output/qwen3-parse`，必须不存在或为空，每次使用新的目录以保留失败日志与首次报告。`--timeout` 默认 300 秒，限制整个验证子进程，包含文件检查、解析、审计，以及 download 模式的下载；慢网络应据实调整此上限。它是超时设置，不是执行耗时或纯解析耗时。

固定 ZIP 约 1.24 GB，解包后需要数 GiB 磁盘空间，完整权重解析也需要数 GiB 内存。解析通过不表示当前 tensor-c 的常量上限或 QEMU 内存布局能够装载完整模型；不要从小型 fixture 的内存消耗推断完整图容量。

## 验收范围与证据

检查顺序及具体失败阶段以 `report.json` 为准。完整门禁覆盖固定产物身份、完整图结构与 28 层覆盖、实际 ONNX→共享 IR 解析、节点转换与 initializer 绑定审计，以及共享 IR 验证器检查。节点数量、算子统计、绑定数量和实际耗时应读取本次报告，不使用旧报告代替执行。

结构检查沿实际输入依赖追踪模型，而不是只统计节点名中的层号：

- 核对全部 initializer 的名字、shape、dtype，以及两个静态输入和 logits 输出。
- 核对每层投影和 RMSNorm、Q/K 归一化、attention 的缩放/掩码/Softmax/V/O 路径、RoPE 位置依赖和 GQA 扩展。
- 核对 SwiGLU、两个残差连接和完整层顺序，最后归一化连接共享 LM Head。
- 要求全部节点和 initializer 都参与返回 logits 的依赖，避免断开的装饰节点被当作完整模型。

这些是固定图的结构规则，不是对算子数值实现或所有可能 ONNX 导出形式的证明。

转换审计记录实际 parser 的每个节点，核对 IR 指令区间、opcode、输入 Value、输出映射和静态属性，允许合法的 Constant 绑定与 Identity 别名而不把它们误报为漏转换。形状、切片和轴等控制参数从原 ONNX 小型控制子图独立求值，不复用被检查解析器的常量缓存。所有 ONNX value 的静态 shape/dtype、全局张量集合与唯一 RETURN 也会核对。权重和 Constant 绑定逐张量比较 shape、dtype 与内容 SHA256；外部分片按块读取校验，不为审计再保存一套权重。该审计采用明确支持的基础算子转换规则，未知转换方式应报错，由新增规则和测试后再接受。

| 文件 | 用途 |
|---|---|
| `report.json` | 机器可读的本次结果、阶段、来源与审计摘要 |
| `report.md`、`report.html` | 便于 review 的摘要和可浏览报告 |
| `nodes.json` | ONNX 节点与转换审计证据 |
| `bindings.json` | initializer / IR 绑定审计证据 |
| `ir.txt` | 本次解析得到的共享 IR 文本 |
| `worker.stdout`、`worker.stderr`、`worker-progress.json` | 子进程日志和最后进度，用于定位超时或异常退出 |

报告目录不包含权重分片。CI 上传该目录，不上传模型目录。失败可能发生在报告明细生成之前，应结合报告与命令日志判断缺失文件，不能把缺失当作通过。

`attempt` 保存 supervisor 启动前的命令、输入路径、解释器与源码身份；它用于定位早期失败，不代表 worker 已初始化或验证通过。`log_errors` 和 `evidence_errors` 分别保存附加日志、失败明细的写入问题，不覆盖原始解析/审计错误。成功路径的必需证据或日志无法保存时仍失败。

`seconds` 记录 supervisor 溯源开始至验证与日志保存结束的耗时，不含随后读取最后进度与生成最终三种报告的时间；各阶段和纯解析耗时另行记录。峰值 RSS 覆盖验证子进程，不包含 supervisor 父进程。平台无法提供的测量值保持未测得。超时后只报告已经保存的进度，不能将 300 秒上限当作实际解析耗时或补全缺失的峰值内存。

成功需要命令退出 0 且本次 `report.json` 的 `passed` 为 true。失败时保留退出码、失败阶段、错误、输入模型身份和已经产生的明细。记录实际 checkout SHA；有未提交修改时还应保存完整 diff 和新增文件，单独记录 HEAD 不能确定所执行代码。

`passed=true` 只证明这份固定产物完成本门禁的前端解析与审计。它不证明完整 IR 数值正确、优化后语义等价、完整权重部署、文本生成或目标硬件性能。下一阶段仍需逐步开展完整模型 IR/ORT 数值对照和容量验证。

## 轻量测试与 CI

```bash
python -m pip install "pytest==9.1.1"
python -m pytest tests/test_qwen3_full_structure.py tests/test_qwen3_full_audit.py tests/test_qwen3_full_parse.py -q
```

命令中的 `python` 指已激活的 `output/qwen3-parse-venv/bin/python`。这些测试使用小型 fixture 和故障注入，不下载或加载完整权重，不能代替完整产物实跑。

[LLM 工作流](../../.github/workflows/llm-deploy.yml) 的 PR 路径接入上述三组测试，并在独立 `full-qwen3-frontend` job 以 `--mode download` 实跑完整固定图。手动/定时同样运行该解析任务。解析有独立 Summary，并以 `always()` 上传 `output/qwen3-parse/`，artifact 名为 `qwen3-full-frontend-parse-output`。下载约 1.24 GB，解析 worker 峰值 RSS 约 5 GB；runner 的实际容量与耗时仍须远端验证。

`w2-acceptance` 汇总任务要求数值任务和完整解析任务均实际成功，失败、取消、跳过或缺失都不能通过。手动/定时的 `full-qwen3-onnx` job 独立保留 W1 完整 ONNX ORT 验证；解析不依赖它成功。并行的两个完整模型任务会各自下载资产。

PR #91 已合并为 W1 基线；这些 W2 代码、测试、文档与 CI 配置独立发布，W1 两层报告修复见 [PR #93](https://github.com/ScratchV-Compiler/ScratchV/pull/93)。[W2 交付记录](../../docs/llm-deploy-v1.0/W2/README.md) 保存 Linux 复现入口与已记录的 Linux CI 证据，CI 配置存在不等于远端已通过。W1 已有旧提交的两层第二人复现；完整模型第二人复现和团队接口确认仍待完成，见 [W1 记录](../../docs/llm-deploy-v1.0/W1/README.md)。
