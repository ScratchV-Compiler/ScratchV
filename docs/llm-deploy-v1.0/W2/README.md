# W2 Linux 独立复现与验收

普通 PR 的 W2 验收现在归入统一 **CI** 运行页，关键数据仍由部署 job 写入同页 Summary；详见 [Actions 入口说明](../../CI.md)。下文历史运行链接保留原身份。

正式交付、CI 和第二人复现统一使用 **Ubuntu 24.04 x86_64 / Bash / Python 3.12**。完整环境准备见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。Linux 为宿主环境；QEMU 内的 RV64GC/LP64D 程序仍使用当前裸机 ABI。

W2 的范围是完整前端解析、小模型 IR 数值、基础后端和 Host 单步接口。完整 28 层模型的 IR 数值前向属于 W3；完整模型 QEMU 与生成循环属于后续阶段。W1 的人工接口决议继续见 [W1 记录](../W1/README.md) 和 [Issue #92](https://github.com/ScratchV-Compiler/ScratchV/issues/92)。

## 独立复现版本

[PR #94](https://github.com/ScratchV-Compiler/ScratchV/pull/94) 的既有 Linux 记录对应 `2d07ccc42936dcfde19d552bca02a324cd33821c`。后续文档或源码更新不自动继承该提交的执行结果。开始前记录约定 SHA、`git rev-parse HEAD` 和 `git status --short`；有未提交增量则保留完整 diff 和新增文件，不能只报告 HEAD。

发布前个人工作树的详细测量按原始身份保留于 [固定提交归档](https://github.com/ScratchV-Compiler/ScratchV/blob/2d07ccc42936dcfde19d552bca02a324cd33821c/docs/llm-deploy-v1.0/W2/README.md)。当前页面只提供 Linux 交付入口与可关联到具体提交的 Linux CI 记录，不把历史个人测量改名为 Linux 结果。

## 七项统一验收

安装与系统工具要求见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。以下命令在仓库根目录使用已安装 `llm-deploy` 的 Python 环境执行。

```bash
python scripts/run_w2_acceptance.py --output-dir output/w2-acceptance-linux-01
```

默认读取 Git LFS 管理的 `assets/qwen3-0.6b/tokenizer` 和 `assets/qwen3-0.6b/onnx`，资产获取见环境指南。使用其他本地目录时可指定 `--tokenizer-dir` 和 `--model-dir`。所有文件必须匹配固定 manifest，只有 LFS pointer 不可运行。测试全程离线；解析需要数 GiB 空闲内存。`--output-dir` 必须是本次全新的路径，连已存在的空目录也会被拒绝。

| 门禁 | 实际执行与判断 |
|---|---|
| `frontend` | RMSNorm、RoPE、SwiGLU、GQA 四种基础算子分解图；独立公式与反例 |
| `backend` | 七类基础算子的 16 个小图 × none/all，共 32 次 QEMU；误差 `<1e-5` |
| `runtime` | 官方 Tokenizer 的 20 组语料、480 次对照，以及输入、掩码、greedy、停止边界 |
| `runtime-model` | 完整词表两层随机小配置；2 文本、8 次 Torch/ORT/IR 调用及单步 greedy/解码 |
| `small-ir` | 官方 Qwen3 类两层小配置，7 输入、29 检查点、none/basic/all，误差 `<1e-5` |
| `small-qemu` | 同一本轮普通/诊断图 × none/all × 7 输入，共 28 次真实 QEMU，误差 `<1e-5` |
| `full-parse` | 固定完整 28 层 ONNX 的结构、逐节点转换、真实权重绑定与 IR verifier 审计 |

所有数值门禁同时检查 shape、dtype、有限值；上述阈值使用最大绝对误差、`rtol=0`，包括 padding 输出。Attention 使用基础算子组合，不要求新增融合 opcode。Tokenizer 的“往返 100%”指与官方 NFC/特殊 Token 规则一致，不是任意字节可逆。

仅七项全部执行并通过才会退出 **0 / PASS**；`--gates` 选择部分门禁即使成功，也只会退出 **2 / PARTIAL**。失败退出 1，中断按实际退出状态记录。`runtime-model` 自动带上 `runtime`，`small-qemu` 自动带上 `small-ir`。缺失、跳过、非有限误差、复用旧输出或执行中源码变化均不能冒充完整通过。

## 报告与独立复现记录

总报告位于本次 `--output-dir` 下的 `report.json`、`report.md`、`report.html`；子目录保存各门禁原始报告、命令、日志、数组和模型/ELF 身份。源码指纹核对的是实际执行文件，不能把别处安装的模块误当当前 checkout。

数值报告首页按链路展示普通图的最终 logits（模型输出）：ORT 对 PyTorch、原生 NumPy IR 对 ORT，以及生成 C 后经 RISC-V/QEMU 执行的输出对 ORT。最后一项同时验证 C 生成和目标端执行，并不是独立的宿主 C Transformer 测试，也不验证旧标量汇编后端。中间检查点用于定位误差，单独折叠，不混入最终输出的精度。

每次实际数组比较新增余弦相似度与相对 L2 误差：前者为两输出向量夹角的余弦，越接近 1 越一致；后者为 `||实际输出−参考输出||₂ / ||参考输出||₂`，越接近 0 越一致。汇总取各次比较中最小余弦、最大相对 L2 和最大绝对误差，它们可能来自不同用例；原有最大绝对误差门槛保持不变。零范数、非有限值等无法定义的指标明确标注原因，不填成 0；旧报告缺少新增指标时显示未记录。

逐用例、检查点、计时和完整 JSON 默认折叠。对应表下解释普通图/诊断图及 IR 优化等级；QEMU 秒数仅为进程墙钟时长。报告精简不减少实际测试：七组输入包含两组满长随机输入与五组边界/扰动用例，不是七个随机种子。

Actions Summary 直接展示上述精简报告。步骤被跳过时明确显示“未运行”，不使用残留报告；步骤已运行但报告缺失时显示“未生成报告”，不能据此认定数值验收通过。

**PPL / zero-shot 暂不纳入当前验收。** PPL（困惑度）衡量模型预测真实文本下一个 token 的能力；zero-shot 衡量不提供示例时完成指定任务的能力。W2 使用随机权重小模型，这两项不能说明其语言质量。后续应在真实预训练权重上固定语料、Tokenizer、上下文窗口和任务评分方式，再比较编译前后质量；本轮明确标记未评测。

- 保存 checkout SHA、dirty/diff、Linux 发行版、CPU、Python/NumPy/ORT/Torch、Zig/QEMU 版本。
- 保存每项实际退出码、PASS/FAIL/PARTIAL/未执行、原始误差、首次失败阶段及报告位置。
- QEMU 时长是进程墙钟，包含启动、guest 执行、UART 和退出；不是目标硬件性能或纯 forward 时间。
- 重新生成自己的模型、参考与输出，不能下载第一人的数组后称为独立执行。
- 独立复现和团队接口/出口确认分开记录，作者再次执行或 CI 自动运行不替代第二人。

## 已有 Linux CI 证据

以下全部归属历史 SHA `2d07ccc42936dcfde19d552bca02a324cd33821c`，不作为这次文档更新的新执行结果：

| 工作流 | 已记录结果 |
|---|---|
| [通用 CI](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37277915411) | 成功；2927 passed / 7 skipped。Linux `test_w2_acceptance.py` 55/55、无跳过 |
| [LLM Deploy](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37277915420) | 成功；32/32 基础 QEMU、28/28 两层 QEMU、3/3 显式产物、20/20 超时清理；两层诊断最大误差 `1.7285346984863281e-6` |
| [Topic 06](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37277915428) | 成功；范围以该 run 的实际步骤与 artifact 为准 |

PR 中按条件未运行的完整 ONNX ORT 重型任务仍是未执行，不因上述绿色记录而算作完整 ORT 重验。新的提交需要新的 Checks；已有 Linux CI 不替代本次七项统一入口的第二人独立复现。

## 分项入口与后续边界

- [完整前端解析](../../../probes/w2_qwen3_parse/README.md)：验证完整图的解析与绑定。报告含 `nodes.json`、`bindings.json`、`ir.txt`。
- [前端模式与基础后端](../../../probes/w2_backend_ops/README.md)：NumPy 独立公式、ORT、IR 和 QEMU 四路检查。
- [Tokenizer 与输入](../../../probes/w2_runtime/README.md) 和 [单步模型接口](../../../probes/w2_runtime_model/README.md)：模型词表 151936 与实际可解码 ID 集合分开处理，不取模重映射。
- [两层数值探测](../../../probes/w2_qwen3_small/README.md)：普通图、诊断图、产物绑定、仿真计时与精度边界。
- [发布范围与修复](W2-分支验收与发布说明.md)：共享后端、日志和 Linux 取消清理的交付归属。

完整解析通过不能替代完整 IR 数值、完整权重独立装载、完整模型 QEMU 前向或真实文本生成。阶段安排见 [开发计划](../开发计划.md)。
