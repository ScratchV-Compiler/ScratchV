# W2 测试

在仓库根目录运行全部 W2 测试：

```bash
python -m pytest tests/llm-deploy/w2 -q
```

| 测试项 | 测试内容 | 单独运行 |
|---|---|---|
| 前端算子组合 | 检查 RMSNorm（归一化）、RoPE（位置旋转）、SwiGLU（前馈网络）、GQA（共享键和值的注意力）四类计算。对比独立 NumPy 公式、ONNX Runtime 和 ScratchV 解释器；故意改错 epsilon、旋转符号、门控分支和注意力头顺序，确认测试能发现错误。共 22 项。 | `python -m pytest tests/llm-deploy/w2/test_frontend_ops.py -q` |
| 后端算子 | 16 组矩阵乘法、逐元素运算、Softmax、RMSNorm、RoPE、SwiGLU、GQA 输入，分别使用 `none/all` 优化。在解释器和真实 QEMU 中运行，均对比 NumPy 与 ONNX Runtime，共 32 次 QEMU 执行。 | `python -m pytest tests/llm-deploy/w2/test_backend_ops.py -q` |
| Tokenizer 与输入准备 | 20 组文本，对比官方 fast/slow Tokenizer 和固定预期结果，保留 480 次编码、解码检查；验证默认选项、特殊 token、空白处理及词表信息。检查长度 1、17、255、256 的输入、遮罩、最后有效位置的 token 选择、并列分数处理，以及结束符、长度和生成数量限制。 | `python -m pytest tests/llm-deploy/w2/test_runtime.py -q` |
| 文本到下一个 token | 中文和含特殊 token 的两组文本，使用保留 151936 词表的两层随机权重 Qwen3；比较 PyTorch、ONNX Runtime、ScratchV 解释器的 `none/all` 输出，以及所选 token 的编号和解码结果。共 8 组运行对比。 | `python -m pytest tests/llm-deploy/w2/test_runtime_model.py -q` |
| 两层 Qwen3 数值 | 7 组输入，检查嵌入、两层注意力和前馈网络、最终归一化与输出，共 29 个中间结果。比较 PyTorch、ONNX Runtime 和解释器；验证收集中间结果不改变输出，并用独立 NumPy 计算核对注意力。 | `python -m pytest tests/llm-deploy/w2/test_qwen3_small.py -q` |
| 两层 Qwen3 优化与 RISC-V | 只输出最终结果的模型，以及额外输出中间结果的模型，各运行 7 组输入。解释器使用 `none/basic/all`，共 42 组；QEMU 使用 `none/all`，共 28 组。比较最终及中间结果，并验证修改后续输入或填充位置不影响前面的有效输出。 | `python -m pytest tests/llm-deploy/w2/test_qwen3_small.py -k 'small_ir or small_riscv or causality' -q` |
| 完整 Qwen3-0.6B 解析 | 读取本地 Git LFS 模型和权重，检查全部 28 层的连接关系，实际解析成 ScratchV 中间表示；逐项核对节点、类型、形状、权重内容和最终输出的对应关系。 | `python -m pytest tests/llm-deploy/w2/test_qwen3_parse.py -q` |

两层 Qwen3 的 7 组输入为：两个不同种子的长度 256 输入、长度 1/17/255 输入、修改第 64 个位置及以后内容的输入，以及修改填充位置的输入。两层模型由 fixture 自动生成固定随机权重；完整权重和 Tokenizer 分别读取 `assets/qwen3-0.6b/onnx/`、`assets/qwen3-0.6b/tokenizer/`。

数值对比检查形状、数据类型和有限值，要求最大绝对误差小于 `1e-5`，包含有效位置和填充位置。完整模型解析测试检查结构与权重对应关系。
