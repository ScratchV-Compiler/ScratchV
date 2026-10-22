# W1 测试

在仓库根目录运行全部 W1 测试：

```bash
python -m pytest tests/llm-deploy/w1 -q
```

| 测试项 | 测试内容 | 单独运行 |
|---|---|---|
| 矩阵乘法 | 7 组输入：4×4 小数矩阵、线性投影、注意力矩阵、批次广播、向量点积、矩阵乘向量、向量乘矩阵。将 ScratchV 编译后的 RISC-V 程序放到 QEMU 中运行，与 ONNX Runtime 比较。 | `python -m pytest tests/llm-deploy/w1/test_matmul.py -q` |
| 两层 Transformer | 使用固定种子的随机权重，比较独立 NumPy 计算、ONNX Runtime 和 ScratchV 中间表示解释器的输出。模型由 fixture 自动生成。 | `python -m pytest tests/llm-deploy/w1/test_tiny_transformer.py -q` |
| 完整 Qwen3-0.6B 资产 | 检查本地 Git LFS 模型及权重文件的大小、哈希、数据类型、外部权重引用和固定输入输出形状。 | `python -m pytest tests/llm-deploy/w1/test_full_qwen3.py -k assets -q` |
| 完整 Qwen3-0.6B 运行 | 使用本地完整权重，在 ONNX Runtime 中分别运行有效长度为 5 和 256 的输入；检查输出形状、FP32 类型，且所有数值有限。 | `python -m pytest tests/llm-deploy/w1/test_full_qwen3.py -q` |

数值对比要求最大绝对误差小于 `1e-5`。完整模型位于 `assets/qwen3-0.6b/onnx/`，测试读取已有资产。
