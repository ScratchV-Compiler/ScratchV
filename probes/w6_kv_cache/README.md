# Qwen3 KV cache probe

用项目两层随机权重 Qwen3 验证固定容量 256 的 Prefill/Decode。默认执行 PyTorch、ONNX Runtime、IR 解释器和编译后的 Host C；`--qemu` 额外构建真实 RISC-V ELF 并执行常驻会话。没有模拟目标数值或用 Host 结果替代 QEMU。

```bash
.venv-llm/bin/python probes/w6_kv_cache/run.py \
  --output-dir output/kv-check --qemu --cores 1
```

`--cores` 支持 1/2/4；`--benchmark` 额外比较相同权重、固定 17 token 提示词和 4 token 预算的无缓存/缓存生成。请单独执行性能测量，避免其他验证争用 CPU。所有输出必须放在新目录。

`export.json` 包含接口、opset、算子及形状清单；`validation.json` 包含逐步误差与内存；`qemu-N/session.json` 包含请求编号、提交长度、阶段耗时及回收记录。`--benchmark` 还产生 `benchmark.json`。失败时检查日志和未删除的构建/会话目录；大型文件写在 `output/`，不发布为模型验收资产。

完整操作说明和已完成的验证范围见 [W6](../../docs/llm-deploy-v1.0/W6/README.md)。

## 完整预训练 Qwen3-0.6B

`full.py` 使用原始 checkpoint，分进程执行参考计算、编译和真实目标验证，避免多个模型实例同时占用内存。需要固定 revision 的 `model.safetensors`、`config.json`、`generation_config.json`；分词器和原有无缓存 ONNX 使用仓库现有资产。不能用两层随机模型的结果代替本流程。

以下命令从仓库根目录执行，使用固定 `.venv-llm` 环境。`output/kv-full` 开始时必须不存在；两个图分开导出到同一目录。完整模型默认使用 Dynamo 导出，不升级依赖。

```bash
.venv-llm/bin/python scripts/export_qwen3_kv_onnx.py \
  --model-dir output/qwen3-0.6b-source --output-dir output/kv-full \
  --mode prefill --threads 4
.venv-llm/bin/python scripts/export_qwen3_kv_onnx.py \
  --model-dir output/qwen3-0.6b-source --output-dir output/kv-full \
  --mode decode --threads 4

# 按顺序执行，每个阶段单独启动一个进程。
for stage in torch ort-prefill ort-decode baseline build qemu host; do
  .venv-llm/bin/python -u probes/w6_kv_cache/full.py \
    --model-dir output/qwen3-0.6b-source --output-dir output/kv-full \
    --stage "$stage" --cores 4 --threads 4 --timeout 3600 || break
done
```

`qemu` 需要允许本地 UART/QMP socket。阶段报告是同名 JSON；成功阶段拒绝覆盖，可以从下一个阶段继续。QEMU 会话目录也必须是新目录。中断后保留失败报告及日志，重试时先归档该阶段的会话目录。

参考用例包括有效长度 1、5、17、255，以及 Decode 写入物理位置 255；检查所有 57 个输出、每个 head 的有效 K/V、缓存输入不变、padding 隔离和历史缓存敏感性。Host C 检查文本四步和容量末端两步。真实四核 QEMU 使用生产生成循环和官方分词器，执行一次 Prefill、三次 Decode，逐步读取全部缓存与 ORT 对照，并验证 RESET 和进程回收。全模型标准沿用 W4：所有有效元素严格 `max_abs < 1e-3`、`rtol=0`，greedy token 完全一致；无效尾部要求有限，QEMU 提交后要求清零。

全模型采用 `blocked_fma / 2x4 / column-panel`、`two-four-lane-last-axis`、`fp32-reference`，与现有完整模型验证策略一致。Host C 为单核本机参考；`--cores 4` 指目标 QEMU 核数。逐请求目标前向时间在 `qemu/session.json` 中；`qemu.json` 的阶段时间还包括缓存读取与逐元素对照，不应当作纯生成性能。
