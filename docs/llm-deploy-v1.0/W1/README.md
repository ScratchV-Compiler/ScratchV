# W1 独立复现与验收记录

本指南用于另一名成员从目标提交重新生成参考结果、编译产物并运行探测。当前已知本地结果见 [执行计划](W1-执行计划.md)，接口候选见 [interfaces.md](interfaces.md)，风险见 [risks.md](risks.md)。**第二人复现和团队确认尚未完成**；请填写末尾模板，不要把这些文字当作签字记录。

所有命令在仓库根目录执行。先记录 `git rev-parse HEAD` 和 `git status --short`；干净checkout便于关联提交，存在本地改动则保留diff并明确标记。不要使用第一人的预生成参考数组代替自己的运行。下述输出目录用完后保留，下一轮使用不同名称，不删除旧证据。

## 1. 建立固定CPU环境

要求Python3.12，固定依赖在 `requirements/qwen3-small-probe.txt`。先安装CPU PyTorch，再装其余依赖，避免默认索引解析成CUDA包。

Windows PowerShell：

```powershell
python -m venv output/w1-repro-venv
$probePython = ".\output\w1-repro-venv\Scripts\python.exe"
& $probePython -m pip install --upgrade pip
& $probePython -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
& $probePython -m pip install -r requirements/qwen3-small-probe.txt
& $probePython -m pip install --no-deps -e .
& $probePython -m pip install "ziglang==0.14.1"
& $probePython -m pip check
$env:SCRATCHV_CC = (& $probePython -c "from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent / 'zig.exe')")
```

准备 `qemu-system-riscv64.exe` 便携工具，可放 `output/tools/` 由运行器自动查找；否则将 `SCRATCHV_QEMU` 设为已安装可执行文件的实际路径。QEMU Windows分发入口见 [QEMU下载页](https://www.qemu.org/download/#windows)。不需要WSL或Linux镜像。

Linux shell：

```bash
python3.12 -m venv output/w1-repro-venv
. output/w1-repro-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/qwen3-small-probe.txt
python -m pip install --no-deps -e .
python -m pip install "ziglang==0.14.1"
python -m pip check
sudo apt-get install --no-install-recommends qemu-system-misc
export SCRATCHV_CC="$(python -c 'from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent / "zig")')"
qemu-system-riscv64 --version
```

后续代码块中的 `python` 均指上述环境。PowerShell请将行首 `python` 换成 `& $probePython`。CPU小模型无需官方权重下载；首次编译Zig会建立缓存，应允许足够编译时间，不要将首次构建耗时解释为推理性能。

## 2. 文档、回归与原两层合成图

```bash
python -X utf8 -B scripts/check_llm_deploy_docs.py --week W1
python -X utf8 -B -m pytest tests/test_ir_interpreter.py tests/test_ir_interpreter_control_flow.py -q
python -X utf8 -B -m pytest tests/test_tensor_c_codegen.py tests/test_tensor_compiler.py tests/test_riscv_tensor_runtime.py tests/test_optimizer_numeric_semantics.py tests/test_w1_qwen3_export.py -q
python -X utf8 -B -m pytest tests/test_qwen3_small_probe.py tests/test_qwen3_small_model.py tests/test_qwen3_small_gate.py -q
```

保留pytest实际输出中的passed/skipped/failed和依赖版本；没有torch导致skip不能当成官方模型测试通过。本轮历史“237项”是当时的专项集合计数；未来新增测试后以新输出为准。

原PR89模型使用Git LFS。若checkout中只有pointer，先取得真实对象：

```bash
git lfs pull --include="probes/w1_tiny_transformer/out/tiny_transformer_2l.onnx"
python -X utf8 -B probes/w1_tiny_transformer/run.py --model probes/w1_tiny_transformer/out/tiny_transformer_2l.onnx --output-dir output/w1-repro-synthetic
```

确认 `report.json` 中 `model_ok`、`ir_ok`、`passed` 为true，`ir_max_abs < 1e-5`。该合成图没有官方Q/K RMSNorm且采用partial RoPE，因此还需要下一节。

## 3. 真实Qwen3小结构与真实RV64执行

按顺序执行，各输出目录应不存在或为空：

```bash
python -X utf8 -B probes/w1_matmul_4x4/run.py --output-dir output/w1-repro-matmul
python -X utf8 -B probes/w2_qwen3_small/run.py --output-dir output/w1-repro-qwen3-small
python -X utf8 -B probes/w2_qwen3_small/riscv.py --model-dir output/w1-repro-qwen3-small --output-dir output/w1-repro-qwen3-riscv
```

所有命令必须退出0；核对报告，不能只看终端最后一行：

| 报告 | 应核对内容 |
|---|---|
| MatMul `report.json` | passed=true，7用例；其中4×4小数FP32真实QEMU，max_abs严格小于1e-5 |
| 小模型 `report.json/md/html` | 7输入、29检查点、普通/诊断图的PyTorch/ORT/IR比较；因果/padding不变性与独立GQA检查 |
| RV64 `report.json/md/html` | 4构建、28次QEMU执行；none/basic/all优化IR比较；none/all普通/诊断QEMU结果、全部检查点与不变性 |
| 构建/运行目录 | 生成C、启动汇编、链接脚本、ELF、命令/日志、UART原始二进制及参考/实际数组 |
| 指纹 | 报告中的source/model/input/ELF哈希、工具版本与实际目标 |

阈值固定 `max_abs < 1e-5`、`rtol=0`，shape/dtype/有限值均须一致，包括padding query。失败时先看stage/current_case/first_divergence；保留报告，不提高容差或只重新运行成功的case。

这是真实RISC-V指令执行，但使用小维随机权重；不会产生有意义文本，不代表完整0.6B加载、预训练两层切片或目标硬件性能通过。旧标量selector不在这条路径内。

## 4. 完整Qwen3 ONNX门禁

入口 `probes/w1_qwen3_export/run.py` 提供三种模式。所有模式最终都执行真实ORT两组输入（短padding/满256），检查FP32静态I/O、完整logits形状和有限值；不是只读历史JSON便判PASS。

### 4.1 下载固定发布产物并验证

```bash
python -X utf8 -B probes/w1_qwen3_export/run.py --mode download --model-dir output/w1-repro-full-model --output-dir output/w1-repro-full-download --threads 2
```

这是数GiB下载，需预留下载、解包、运行内存与磁盘空间。入口使用固定release与ZIP SHA256，并核对模型及三个权重分片哈希。已有固定ZIP可加 `--archive <实际ZIP路径>` 避免重复联网；下载目录按入口要求准备。产物manifest见 `probes/w1_qwen3_export/manifest.json`。

`download` 模式验证既有导出并重新跑ORT，**不重新导出、不重新比较PyTorch数值**。它可重验W1“完整导出可载入、固定形状、ORT可执行”的交付物，不能写成完成了一次新导出。

### 4.2 验证已有完整模型

将从固定发布产物获得的 `model.onnx` 与真实三个external-data文件放同一模型目录，使用：

```bash
python -X utf8 -B probes/w1_qwen3_export/run.py --mode verify --model-dir output/w1-repro-full-model --output-dir output/w1-repro-full-verify --threads 2
```

不能仅复制ONNX protobuf，不能使用Git LFS pointer冒充权重。验证模式同样校验固定产物哈希并实际运行ORT；与下载模式的执行/数值范围相同。

### 4.3 从固定官方snapshot重新导出

源目录应含revision `c1899de289a04d12100db370d81485cdf75e47ca` 对应的完整官方snapshot。源checkpoint SHA256在manifest中固定。使用前述CPU小模型环境即可，也可单独按 `requirements/qwen3-export.txt` 准备同版本CPU环境。

```bash
python -X utf8 -B probes/w1_qwen3_export/run.py --mode export --source-dir output/qwen3-official-snapshot --model-dir output/w1-repro-reexported-model --output-dir output/w1-repro-full-export --threads 2
```

`--source-dir` 必须替换为已经准备好的固定snapshot目录，入口不把任意目录/版本视为官方基准；模型目标目录必须为空，报告在模型目录之外。此模式调用仓库 `export_qwen3_onnx.py`，检查来源与新产物，并运行PyTorch/ORT有效位置比较。该既有导出验证使用 `atol=rtol=1e-4`，与小模型IR/QEMU严格1e-5门槛分开；padding全张量比较结果单列，不能隐藏失败。

### 4.4 阅读完整模型报告

输出 `report.json`、`report.md`；核对mode、passed、各阶段hash/structure/ort、输入有效长度、shape、dtype、有限值、provider/线程数。导出模式另有export日志与export_validation；下载/验证模式不具备新PyTorch对照。

peak_memory是验证器进程峰值RSS，**不包含导出子进程**，不能写成整次导出峰值内存。新报告的 `source_fingerprints` 保存入口wrapper、导出脚本、manifest的SHA256，成功/失败均记录；源码不可读取则为null并附错误。既有本地 `output/qwen3-full-local/report.json` 生成于此字段加入前，不能假称其已有源码指纹；按实际字段与对应checkout核对。历史verification只作来源说明。缺文件、哈希不符、版本不符、ORT失败都应非零退出。

## 5. CI状态记录

工作流为 `.github/workflows/llm-deploy.yml`：

- PR执行原合成图、小模型IR、小模型QEMU和 `tests/test_w1_qwen3_export.py` 的轻量回归，不下载完整模型。
- 手动或定时运行另启 `full-qwen3-onnx` job，使用 `--mode download --threads 2` 校验固定发布产物并真实运行ORT两case，上传 `output/qwen3-full-probe/` 报告。
- 完整模型job只需要固定NumPy2.2.6、ONNX1.18.0、ORT1.22.1、protobuf5.29.5，不需要PyTorch；这也意味着它不重新导出或比较PyTorch。
- PR中完整模型job按条件未执行，应明确记录“ORT重型门禁未执行”，不能用轻量测试通过代替。

已确认本地28次两层QEMU执行，以及完整模型新入口verify的hash/checker/ORT两case通过（验证器峰值RSS约6.40 GB，报告 `output/qwen3-full-local/report.json`）。提交 `545e696` 已有两项Linux成功证据：

- [完整ONNX重型任务](https://github.com/yuki-328/ScratchV/actions/runs/36983119833/job/110761955767)：真实固定release下载、校验、ORT两case及artifact上传。
- [小模型部署功能任务](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/36983000988/job/110761571841)：所有部署步骤通过，含MatMul、合成图IR、官方Qwen3小模型IR和28次真实QEMU执行，artifacts上传成功。

545e696首次通用主CI出现过读取 `/proc` 时进程退出的测试竞争失败，已仅修改测试修复，追加确定性用例后本地26项运行时测试通过；运行时源码未因此改变。不要把这个历史提交写成“整个CI全绿”，修复提交及后续主CI状态以PR Checks为准。上述自动运行也不等于第二名成员独立复现或团队确认。

后续提交以对应Checks为准。独立复现报告须附run URL及head SHA，逐项确认步骤实际执行；源码或编译器变化时重跑受影响验证。“先前PR绿色”不足以证明新提交的tensor-c路线。

## 6. 独立复现模板

复制以下表格到复现记录或PR评论，填写实际证据。未经执行保持“未执行”，未经本人确认保持“待确认”。

| 字段 | 记录 |
|---|---|
| 复现人 / 日期 | 待填写；应为另一名成员 |
| checkout commit / 是否有本地diff | 待填写 |
| OS / Python / torch / transformers / onnx / ORT | 待填写 |
| Zig / QEMU版本、CPU/线程设置 | 待填写 |
| 执行命令、各退出码、测试passed/skipped/failed | 待填写 |
| 原合成图结果 | 待填写：最大误差与报告 |
| 官方结构IR结果 | 待填写：7case/29checkpoint/不变性 |
| MatMul / RV64结果 | 待填写：7MatMul、4构建/28执行、最大误差/首次偏差 |
| 完整ONNX模式与执行 | 待填写：download/verify/export；revision、各文件hash、ORTcase结果 |
| 完整ONNX数值范围 | 待填写：是否重跑PyTorch、有效token/全padding结果 |
| 源码/模型/输入/ELF指纹与报告artifact | 待填写 |
| 对应LinuxCI run URL / commit /实际执行步骤 | 待填写 |
| 未完成项、失败原因、后续责任人 | 待填写 |
| E4运行时复现确认 / E5数值复现确认 | 待本人填写 |

## 7. W1出口评审模板

| 项 | 结论 / 确认人 / 日期 |
|---|---|
| 四探测及新QEMU路线实际结果齐全 | 待确认 |
| 接受tensor-c/RV64GC裸机路线，保留旧selector范围 | 待确认 |
| 接受基础算子IR、INT64 IDs、指针数组ABI及不可重入边界 | 待确认 |
| 高风险Plan B与完整模型后续任务责任人 | 待确认 |
| interfaces D1–D7逐项确认 | 待确认 |
| 第二人复现与CI证据关联到当前提交 | 待确认 |
| E2汇总最终冻结版本和仍未关闭事项 | 待确认 |

文档结构检查无法验证会议是否举行、人员是否同意、第二人是否真的执行。只有明确记录这些事实后，才能关闭相应人工验收项。
