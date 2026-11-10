# W6 · Qwen3 完整前向的耗时和内存优化

W5 让 QEMU 能根据提示词逐个生成 token。W6 给完整模型的单次前向增加了两项可选优化。在同一组输入上，两项合用后，28 层模型从 **444.0 秒降至 337.6 秒**，本机 Python 与 QEMU 的合计内存峰值从 **6.54 GiB 降至 3.80 GiB**。

[三页演示稿（PPTX）](W6-summary.pptx) · [详细实验报告](EXPERIMENTS.md)

## 改了什么

1. **分块计算模型最后一层。** 最后一层（LM Head）要为词表中的每个候选 token 算分。以前需要把与输入嵌入层共用的大权重矩阵完整转置。W6 每次只取一小块词表权重，供多个输入位置计算，省下约 **593.5 MiB** 的临时空间，仍输出全部候选分数。
2. **让 QEMU 通过文件映射装载权重。** Linux 运行器把权重写到临时文件的指定位置，QEMU 把这份文件映射为模拟内存。相同的分块程序切换到这种装载方式后，合计内存峰值从 **6,152.1 降至 3,886.7 MiB**。

两个选项均可在完整前向入口启用。W5 的文本生成命令继续使用原末层。本次测量覆盖固定输入的一次前向，整条提示词生成 8 个 token 的耗时仍以 W5 的生成报告为准。[W5 生成报告](../W5/README.md)

## 实测结果

模型为 Qwen3-0.6B，28 层、256 个输入位置。使用同一组 `full_seed_0` 输入，固定 QEMU 4 核、FP32、工作区 padding 关闭。每种配置运行一次。

| 完整模型配置 | 单次前向（秒） | Python 与 QEMU 合计内存峰值（MiB） |
| --- | ---: | ---: |
| 原末层，原装载方式 | 443.997 | 6,693.9 |
| 分块末层，原装载方式 | 338.393 | 6,152.1 |
| 分块末层，文件映射装载 | 337.647 | 3,886.7 |

**速度提升主要来自分块末层，内存节省主要来自文件映射。** 两项合用的单次前向加速为 **1.315×**，耗时减少约 **106 秒（24.0%）**，内存峰值降低 **41.94%**。前向从 QEMU 启动计至程序发出完成通知。内存数值取两个进程同一时刻的采样和。

每种配置都与 ONNX Runtime（ORT）比较了全部 **38,895,616** 个输出分数，最大绝对误差为 **0.00002480**，低于要求的 **0.001**。三组输出彼此逐位一致，256 个位置选出的最高分 token 都与参考相同。[完整前向数据](reports/full-memory-4core.json)

## 算子实验与验收进度

| 测量项 | 做了什么 | 实测与进度 |
| --- | --- | --- |
| Dense 矩阵乘法 | 用固定小图复测已有的 2×4 分块与列布局 | **3.754×**，达到 1.5× 目标 |
| 单独测量 LM Head | 在 151,936 词的真实词表上比较两种算法 | **184.33 → 74.06 秒，2.489×**，达到 2× 目标 |
| 独立 Attention 分数矩阵乘法 | 比较不同分块、四核分工、数据读写及向量指令 | 专项选定的 4×4 标量实现为 **1.285×**，低于 1.5× 目标 |
| 转置、矩阵乘法与 Softmax 组合图 | 算完一组分数就做 Softmax，复用临时空间 | 两批分别 **1.081×、1.068×**，工作区 **10 → 4 MiB** |
| W6 专项验收 | 增加小图对照、完整前向数值与内存检查，以及定时工作流 | 本地小图验收因独立 Attention 未达标而失败。完整 Nightly 成功记录待取得 |

<details>
<summary>Attention 还试过哪些做法</summary>

- **工作分配与输入检查。** 每次领取 4 块的收益在复测中反转。四核检查输入虽省约 1.34 毫秒，整张小图用时反而增加约 3.1%。专项选用每次领取 8 块、串行检查输入。
- **分块和数据搬运。** 比较了多种标量分块、K 的打包方式、输出检查与复制、直接写回及成对读取 Q。成对读取 Q 相对原始 1×4 的两批加速为 1.316× 和 1.276×，相对逐个加载 Q 的收益方向相反。
- **浮点计算与调度。** 按核计时、采样和反汇编都指向乘加循环。4×8 短循环两次实测，RVV 比同宽标量用时更长。将四个虚拟核分别固定到本机逻辑 CPU 后，两批从 QEMU 启动计时的前向仍略长，继续由 Linux 分配 CPU。

[独立 Attention 专项](reports/matmul-gate-4core.json) · [成对加载与调度数据](reports/attention-paired-q-4core.json) · [组合图数据](reports/attention-chain-4core.json) · [全部实验与逐项报告](EXPERIMENTS.md)

</details>

## 如何复现

从仓库根目录运行。小图使用生成的输入；完整前向需按 Linux 环境说明用 Git LFS 下载真实的 `model.onnx` 和权重分片。每次指定新的输出目录。[Linux 环境与模型准备](../LINUX_REPRODUCTION.md)

<details>
<summary>小图和完整前向命令</summary>

```bash
python -B -m probes.w6_performance.run --compare-candidates --output-dir output/w6-small-repeat/matmul

python -B -m probes.w6_performance.full_forward --model-dir assets/qwen3-0.6b/onnx \
  --candidate-weight-transport file-backed-ram --output-dir output/w6-full-repeat
```

小图数据写入 `matmul/result.json`，三组完整前向对照写入 `w6-full-repeat/comparison.json`。完整前向入口的两个选项是 `--lm-head-policy vocab-blocked` 和 `--qemu-weight-transport file-backed-ram`；上述对照命令自动为三组选择相应配置。[完整前向入口](../../../probes/w6_performance/full_forward.py) · [LM Head 数据](reports/lm-head-4core.json)

</details>
