# W3 完整资产与静态尺寸预检

复现统一采用 **Ubuntu 24.04 x86_64、Bash、Python 3.12**。先完成 [Linux 环境与资产准备](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md)，再在同一个 Bash 会话、仓库根目录执行命令；该指南设置 `SCRATCHV_PYTHON`、`SCRATCHV_CC` 和 `SCRATCHV_QEMU`。

~~~bash
set -euo pipefail
W3_MODEL=/absolute/path/to/qwen3-0.6b-onnx
"$SCRATCHV_PYTHON" -B -X utf8 probes/w3_full_preflight/run.py --model-dir "$W3_MODEL" --output-dir output/w3-full-preflight-new
~~~

离线流式校验 W1 固定 manifest 的模型和三个 external-data 文件，复用 ONNX 静态契约/范围检查，统计 initializer、嵌入/外部 tensor、具名 value 与输出的逻辑字节。不加载外部数组进行 IR 运算，也不执行完整 ORT。

报告始终标记 full_ir_executed=false、w3_exit_accepted=false。PASS 只代表准备检查通过；sum_named_value_logical_bytes 包含别名、可能重复，不能作为所需物理内存上限或峰值承诺。process_peak_rss_bytes 只属于这个 metadata-only 预检进程。

完整数值验收仍需 W1 人工前置确认、中等与真实子图基线、完整 IR/ORT <1e-4、三人复现和 Nightly。参见 [W3 准备说明](../../docs/llm-deploy-v1.0/W3/README.md)。
