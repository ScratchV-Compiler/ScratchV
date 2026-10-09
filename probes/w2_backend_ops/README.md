# W2 前端模式与基础后端算子验收

正式复现使用 Ubuntu 24.04 x86_64 / Bash / Python 3.12，先按 [Linux 复现约定](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md) 准备并激活环境。下文 `python` 为 `.venv-linux/bin/python`。

`unit:frontend-ops` 明确检查 RMSNorm、RoPE、SwiGLU、GQA 四种分解图模式。
这些模式由现有 ONNX 基础算子表达，不声明已有独立融合 Attention/RMSNorm IR。

```bash
python -m pytest tests/test_qwen3_frontend_patterns.py -q
```

`unit:backend-ops` 从真实 ONNX 图走 production parser、优化器、tensor-c 后端，
交叉编译为 RV64GC ELF，然后在 QEMU `virt` 裸机执行。16 个用例分别运行
`none`、`all` 两个优化级别，共 32 次目标执行；不下载模型权重。

```bash
python probes/w2_backend_ops/run.py --output-dir output/w2-backend-ops --cc "$SCRATCHV_CC" --qemu "$SCRATCHV_QEMU"
python -m pytest tests/test_w2_backend_ops_gate.py -q
```

也可省略工具参数，使用 PATH 或 `SCRATCHV_CC`、`SCRATCHV_QEMU`。
使用现有 `requirements/qwen3-small-probe.txt` 中的 NumPy/ONNX/ORT 环境即可，
本探测不需要 PyTorch。输出目录须不存在或为空；依赖、工具、编译、执行或
比较失败均非零退出，不将跳过当通过。

| 类别 | 用例与针对的问题 |
|---|---|
| MatMul | 4×4、四维 batch 广播、向量 dot 标量返回 |
| 逐元素 | Add/Sub/Mul/Div 串联，标量与尾维广播 |
| Softmax | axis=-1/0/1，大 logits 稳定性及非末轴 |
| RMSNorm | hidden/Q/K 的不同 head 数，全零/极小输入、权重、1e-6/1e-3 epsilon |
| RoPE | 全 head_dim rotate_half、Q/K head 广播、position=0/1/255 及非零起点 |
| SwiGLU | 两组不同 gate/up 权重，SiLU 仅作用于 gate，再经 down 投影 |
| GQA | KV 连续重复顺序、FP32 最小有限值因果 mask、attention context，重复比 2 与 MHA 比 1 |

NumPy 参考使用独立数学公式及双精度中间值，最后转 FP32；ORT 单独执行 ONNX。
IR 与 QEMU 都同时对照这两种参考，要求 shape、dtype 一致、全部有限，且
`max_abs < 1e-5`、`rtol=0`。前端单测还注入错误 epsilon、旋转符号、门控分支、
KV head 顺序，确认用例可以发现这些语义错误。

`report.json`、`report.md`、`report.html` 含逐例误差、失败阶段、环境及源码哈希，
还有编译命令、ELF 哈希和每次 QEMU 进程实测墙钟时长。
`qemu_process_wall_seconds` 包括启动、guest 计算、UART 输出与退出，不是 guest
纯计算时间或周期数。每例同时保存模型、输入、NumPy/ORT/IR/QEMU 张量和工具日志，
可定位第一次不一致发生于参考、IR 还是后端。

门禁验证静态 FP32 小图，不代表完整 0.6B 的内存容量、性能或完整模型 QEMU 前向已通过。
