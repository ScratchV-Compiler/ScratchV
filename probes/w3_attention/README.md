# W3 组合 Attention 后端

入口 run.py 将 Q/K RMSNorm、全 head_dim RoPE、GQA 连续 KV 重复、因果/key-padding mask、Softmax 和 context MatMul 连成一个 ONNX 图。4 个 Q head、2 个 KV head、head_dim8，包含 L17 和 L256 边界；它是小尺寸组合验证，不是完整 Qwen3 前向。

~~~text
python -B -X utf8 probes/w3_attention/run.py --output-dir output/w3-attention-new --cc /path/to/zig --qemu /path/to/qemu-system-riscv64
~~~

6 图分别执行 none/all，要求 12 次真实 RV64 QEMU、每次 IR/QEMU 同时对照 ORT 和独立 FP64 NumPy 公式，全部 max_abs <1e-4、rtol=0。未来 token 和 padding key/value 扰动产生 10 项跨 ORT/IR/QEMU 的隔离检查。NumPy 按 query 的可见 key 范围独立计算，不复用导出 mask；单测注入“额外屏蔽合法 key”确认参考能捕获错误。

所有输入输出、ONNX、ELF、编译/QEMU 命令及工具日志保留在新输出目录。失败/超时的 QEMU 命令与实际耗时也记入 report.json，必需报告写入失败时不保留 PASS 视图。该门槛不修改 W2 的 <1e-5 测试。

参见 [统一准备说明](../../docs/llm-deploy-v1.0/W3/README.md)。
