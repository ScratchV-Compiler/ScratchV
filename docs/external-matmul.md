# 外部 MatMul 调用

对应 [Issue #109](https://github.com/ScratchV-Compiler/ScratchV/issues/109)。

**首期七项已有本地实现和定向测试证据；发布状态与远端 CI 结果以 Issue #109 关联 PR 的 Checks 为准。**
下文各轮实验的“未提交、未推送”等说明记录当时状态，不代表当前发布状态。
验收摘要的交付状态明确保留为 2026-10-24 发布前快照；Issue 中原有的验收勾选不由本文自动更新。首期要求是接口、真实 C 调用、
错误处理、Host 成熟库、数值约定、RISC-V 小图和文档/现有 CI 复用；不要求七个模型用例
或真实硬件验收。完整模型、并行调用和按实测选择属于随后授权的增强。

当前工作分支已同步主干到 `735dd00`，并恢复此前本地改动。
下文 Issue #109 的历史完整模型结果基于 `61bded3` 加当时未提交的实现；
它们**不是同步到 `735dd00` 后重新跑过的完整模型结果**。
引用的 2026-10-22 原 W4 基准保留其自身的历史源码和构建记录。
同步后已完成共享库路径修复的 14 项定向检查（28 项按范围未选），覆盖带空格路径、
版本化 `.so` 文件、不同调用/构建目录以及真实小 C 调用；本地日志为
`output/issue109-pr-prep/link-tests/pytest.log`，机器可读结果见验收摘要。
随后选择的 79 项集成检查首次得到 78 通过、1 失败、0 跳过：失败项是本地展开的
OpenBLAS 在运行时找不到 `libgfortran.so.5`。恢复既有 `LD_LIBRARY_PATH` 配置后，
只复验该失败项，得到 1 通过、0 跳过；没有改生产逻辑，也没有重跑已通过的 78 项。
因此选定的 79 项现在均有通过证据，不能表述为一次运行的“79 项全过”。
其中包含 `CompilerConfig` 到真实 NMSIS/QEMU 小图的调用、CI 路由和 UART 完成通知检查。
首次结果和复验分别保存在本地 `output/issue109-pr-prep/integration-tests/` 与
`output/issue109-pr-prep/openblas-runtime-recheck/`；[验收摘要](external-matmul-evidence.json)
分别保留两次结果。这些同步后检查均未运行完整模型。
随后完成 QK、AV、Q 投影的 12 次耗时拆分、36 次未插桩三方对照和 10 次边界数值检查。
循环展开候选在本次 QK/AV 样本中缩短了 NMSIS 耗时，但三类均未达到相对原实现的选用门槛；
当前生产实现、poll 和既有实测选择配置继续保留。详细数值与不同误差门槛见下文新增章节。
随后又做按四列整理输入与本地分块候选的 28 次边界检查、48 次性能对照。
Q 投影在继续调用官方 NMSIS 算子的情况下，中位耗时由 11.8866 秒降到 1.4979 秒；
本地派生的分块候选为 1.4466 秒，两者仍慢于原优化的 1.0109 秒。
三类矩阵均未达到选用门槛，生产实现和旧测量配置保持原样；本轮没有运行完整模型。
再进一步仅针对 Q 投影复用静态权重，完成 12 次边界运行、8 项生成拒绝检查与 16 次性能对照。
新本地候选比本轮重测的旧候选少 17.57% 前向时间，但仍比原优化多 12.07%；
计入准备成本后仍未准入，现有生产策略继续保留。
随后整理本地核心热循环并对照内联，另完成 12 次边界运行、8 项生成拒绝与 16 次性能运行。
内联候选中位比原优化少 4.39%，但样本范围重叠，含准备成本的 10% 门槛仍未通过；生产策略不变。

可分享的 [验收证据摘要](external-matmul-evidence.json) 保存关键数值、配置、计时限制、
本地原始记录位置及归档回执信息。摘要是已保存结果的转录，未包含完整输入、输出或权重。
本文标为 `output/…` 的位置是被 Git 忽略的**本地材料**，新检出不能直接下载这些数据；
复查原始材料需使用保留的本地目录或归档。各阶段测试有重叠，不能相加作为独立用例总数。

默认仍使用原有计算代码，外部实现需要显式选择。目前按三个阶段接入：

| 阶段 | 实现 | 验证范围 |
| --- | --- | --- |
| 第一批 | 受控 C 函数 | 统一接口、构建依赖和错误处理 |
| 第二批 | OpenBLAS | Linux x86-64 开发电脑、LP64、单线程 |
| 第三批 | NMSIS-DSP | RV64GC / LP64D、QEMU virt、无操作系统；单 hart 与 4 hart 完整模型，以及 1／2／4 hart 小图 |

完整 Host/OpenBLAS 和 RISC-V/QEMU 混合调用均已通过现有数值门槛。
此前 RISC-V 单 hart 与 4 hart 完整模型均保留 253 个主 hart 串行 NMSIS MatMul，
最后一个投影使用原有分块乘加；4 hart 时末层和 28 个 Softmax 使用已有并行路径。
两种配置的全部 38,895,616 个输出逐位一致，对参考最大绝对误差为 0.0001220703125，小于 0.001。
此前串行 NMSIS 的 4 hart 前向耗时 9249.00 秒，约 2 小时 34 分 9 秒，比单 hart 旧估算少约 1.23%。
旧基准曾发生计时接管，且每种配置只测一次，这一差异不足以确认稳定提速。
随后完成 NMSIS 按输出行并行调用：显式设置 `parallel_rows=True`，复用已有工作 hart。
1／2／4 hart 代表矩阵和混合小图全部通过，输出逐位一致；该新配置尚未运行完整模型。
上述完整模型结果仍属于 NMSIS 串行配置。
随后新增按实测选择的 `measured` 模式：保留原有分块和权重排布，只在匹配测量且收益足够时
调用外部库。本轮 RISC-V 代表矩阵保留原优化实现，Host 则按尺寸选择 OpenBLAS 或原实现。
最新补测已覆盖真实完整维度、转置常量和动态右侧：Host 的 9 类记录覆盖 254 个节点，
RISC-V 的 8 类记录覆盖 253 个节点，末层因 NMSIS 接口上限回退。
最终完整图生成选择为 RISC-V 全部 254 个原实现，Host 为 253 个 OpenBLAS、1 个原实现。
新配置随后已完成完整 28 层验证：Host 前向 7.61 秒，RISC-V 4 hart 前向 435.05 秒；
两端全部输出均通过严格误差门槛，RISC-V 与旧原优化结果逐位一致。
本次没有计时接管，完整数据和历史比较见文末。
随后对约 6% 的差值进行了定向排查，并完成同一 ELF 的 select / poll 完整模型重复对照。
两次预热和三组正式对照共 8 次全部通过；正式配对差异为 +0.24%、−1.09%、+1.04%，
没有复现 poll 稳定慢约 6%，原计算优化保持完整。CPU、换页和负载记录已保存；
具体历史宿主原因仍未唯一定位，三组测试也不等于性能等价证明，详见文末。
详见文末分阶段结果与计时说明。

## 选用 OpenBLAS 的原因

| 候选 | 本次判断 |
| --- | --- |
| OpenBLAS | 提供标准 C 矩阵乘法接口，Ubuntu 有现成开发包，适合用少量转换代码接入 |
| BLIS | 成熟备选，但使用其 CBLAS 兼容接口需要对应的构建配置 |
| Netlib BLAS | 提供参考实现，适合学习接口；本次优先接入经过机器优化的实现 |

参考官方说明：[OpenBLAS 安装](https://www.openmathlib.org/OpenBLAS/docs/install/)、
[OpenBLAS 使用](https://www.openmathlib.org/OpenBLAS/docs/user_manual/)、
[BLIS](https://github.com/flame/blis)、[Netlib BLAS](https://netlib.org/blas/)。
OpenBLAS 使用 [BSD 3-Clause 许可](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE)。
本项目只提供适配代码，不随 Python 包分发 OpenBLAS 二进制。

## RISC-V 目标为什么选择 NMSIS-DSP

当前 QEMU 程序没有操作系统，使用 RV64GC 指令集和 LP64D 调用约定；
现有目标未启用向量扩展或 Nuclei 专用 DSP 指令。

| 候选 | 本次判断 |
| --- | --- |
| NMSIS-DSP 1.6.0 | 有适用于嵌入式 RISC-V 的 FP32 矩阵乘法；本轮只编译其标量源码，无线程、堆内存或操作系统调用 |
| OpenBLAS / BLIS | 继续用于开发电脑；其运行时和构建依赖还需额外移植，不能把 RISC-V Linux 支持等同于当前目标支持 |
| [PULP DSP](https://github.com/pulp-platform/pulp-dsp) | 主要针对 RV32IM、XPULPV2 和 PULP 集群，与当前通用 RV64GC 目标不匹配 |

选用官方 [NMSIS 1.6.0](https://github.com/Nuclei-Software/NMSIS/releases/tag/1.6.0)
的 [riscv_mat_mult_f32.c](https://github.com/Nuclei-Software/NMSIS/blob/1.6.0/NMSIS/DSP/Source/MatrixFunctions/riscv_mat_mult_f32.c)，
现有生产适配直接调用该官方源码，不改写其矩阵乘法。文末另列的本地派生候选属于实验。
该源码和上游仓库使用 Apache-2.0 许可。
依赖仍由调用方提供，不随 ScratchV 安装包分发。

本次直接编译单个算子及必要头文件，不运行完整 NMSIS SDK 构建；
官方 SDK 的工具链要求见 [NMSIS-DSP 使用说明](https://doc.nucleisys.com/nmsis/dsp/get_started.html)。
适配代码拒绝当前未验收的向量或专用 DSP 配置。

## 两个独立的选择

“用哪个实现”和“允许哪种计算顺序”分开配置。

| 实现选择 | 行为 |
| --- | --- |
| `builtin`（默认） | 使用原有计算代码，不增加外部依赖 |
| `prefer_external` | 未配置或不支持时使用原实现，并记录原因 |
| `require_external` | 非空、非零 K 的 MatMul 无法使用外部实现时，生成失败 |
| `measured` | 显式提供同条件测量；仅在记录匹配且收益达到门槛时调用外部库，否则保留原优化 |

| 数值选择 | 行为 |
| --- | --- |
| `strict`（默认） | 外部实现必须满足既定计算顺序；第一批受控 C 实现支持顺序累加 |
| `allow_provider_fp32` | 明确允许声明为 `provider_fp32` 的库使用自己的 FP32 计算顺序 |

OpenBLAS 和 NMSIS-DSP 都被保守标记为 `provider_fp32`，需要明确许可。
OpenBLAS 即使设置为单线程，也不保证和现有顺序累加逐位相同。
本轮 NMSIS 标量源码按顺序累加，但不把第三方库所有版本和编译配置都承诺为逐位相同。
严格模式下，“优先外部库”会使用原实现，“必须外部库”会报错。

已有 `matmul_policy` 继续决定原实现和回退路径的计算方式。
允许第三方库后，它不再代表库内部的计算顺序；实际数值约定可查看
`artifact.external_matmul.numerics`，Host 构建记录也会单独标明。

## OpenBLAS 接入示例

三个适配路径都通过 Python API 显式配置，未新增命令行参数或自动下载功能：

```python
from scratchv.backend.openblas import openblas_matmul
from scratchv.backend.tensor_c_codegen import TensorCCodegen

provider = openblas_matmul(
    include_dirs=("/path/to/openblas/include",),
    library_dirs=("/path/to/openblas/lib",),
)

artifact = TensorCCodegen(
    program, initializers,
    constant_storage="external",
    matmul_implementation="require_external",
    external_matmul=provider,
    external_matmul_numerics="allow_provider_fp32",
    matmul_target="host",
).generate()
```

`program` 和 `initializers` 来自现有模型解析流程。
系统默认搜索目录中已经安装好开发包时，可以使用 `openblas_matmul()`。
显式库目录会用于链接和运行时查找；库自己的动态依赖也必须能被系统找到。

通用 `CompilerConfig` 对应设置为：
`tensor_matmul_implementation`、`tensor_external_matmul`、
`tensor_external_matmul_numerics`、`tensor_matmul_target`。
生成结果仍在 `CompilerDriver.tensor_artifact` 中。

每个 MatMul 的选择与原因位于 `artifact.matmul_selections`。
只有真正选择外部实现，`artifact.external_matmul` 才携带构建依赖。

## NMSIS-DSP 接入示例

先在本地准备固定版本的官方源码。以下命令在项目目录执行，
`output/` 是已忽略的本地产物目录；已有检出时复用它，不覆盖：

```bash
git clone --depth 1 --filter=blob:none --sparse --branch 1.6.0 \
  https://github.com/Nuclei-Software/NMSIS.git output/nmsis/NMSIS
git -C output/nmsis/NMSIS sparse-checkout set \
  NMSIS/DSP/Include NMSIS/Core/Include NMSIS/DSP/Source/MatrixFunctions
export SCRATCHV_NMSIS_ROOT="$PWD/output/nmsis/NMSIS"
```

Python 接入与第二批使用同一套接口：

```python
from scratchv.backend.nmsis import nmsis_matmul
from scratchv.backend.tensor_c_codegen import TensorCCodegen

provider = nmsis_matmul("/path/to/NMSIS")  # 官方仓库根目录
artifact = TensorCCodegen(
    program, initializers,
    constant_storage="external",
    kernel_calls=True,
    matmul_implementation="prefer_external",
    matmul_policy="blocked_fma",  # 回退时沿用原有分块优化
    external_matmul=provider,
    external_matmul_numerics="allow_provider_fp32",
    matmul_target="riscv64",
).generate()
```

这个组合会优先调用 NMSIS；不支持的矩阵使用原有分块乘加，
默认沿用 `1x4` 分块和原始权重布局。它复用已有配置，不需要新增接口或运行时。
`matmul_policy` 只控制原实现，不改变 NMSIS 内部算法。
如果要求每个非空 MatMul 都调用外部库，仍可使用 `require_external`，
但完整 Qwen3 的最后一层会因维度超限而生成失败。

工厂只检查和记录本地源码路径，不自动下载。
生成后照常使用 `write_weight_bundle`、`build_external` 和 `run_external`。
如果权重内嵌在代码中，改为 `constant_storage="inline"`，使用
`build_riscv_tensor` 和 `run_riscv_tensor`。两条构建路径无需新增运行时入口。

## 接口与支持范围

统一函数接口仍为：

```c
int function(const float *a, const float *b, float *c,
             uint64_t m, uint64_t n, uint64_t k);
```

A 为 m×k，B 为 k×n，C 为 m×n；使用固定大小、按行连续存放的 FP32 数据。
外部函数覆盖写 C，不修改输入，不保留调用方指针，返回时计算已完成。
向量、多组矩阵和广播继续由生成器处理，默认每组调用一次二维矩阵乘法。
明确启用按行并行且满足条件时，同一组的输出行交给多个 hart，各自调用同一个库函数；
多组矩阵之间仍按顺序执行。

[OpenBLAS 适配函数](../scratchv/backend/openblas_matmul.c) 调用
`cblas_sgemm`，使用不转置输入、乘数 1 和输出系数 0。
它要求标准 **LP64** 接口，即维度参数为 32 位整数；转换前检查大小范围，
编译期拒绝 64 位整数接口的头文件。头文件和库文件必须来自同一接口配置。
不要把 NumPy 私有的 ILP64 库当作标准 OpenBLAS 开发包直接链接。

[NMSIS 适配函数](../scratchv/backend/nmsis_matmul.c) 则构造三个局部矩阵描述，
调用官方 `riscv_mat_mult_f32`。它的维度字段只有 16 位，因此 M、N、K
各自不能超过 **65535**；这是接口上限，实际形状还受现有工作区和 QEMU 内存限制。
转换前检查范围，空矩阵提前返回，避免上游标量循环在零维度时仍执行一次。
不新增矩阵复制、内存申请或额外调用方缓冲区。

返回 0 表示成功；适配函数的非零返回映射为图状态 **4**，停止后续算子，
不重新执行原实现，也不复制未完成结果。按行并行时先等待本次所有调用结束，再返回错误，
避免错误返回后仍有工作 hart 写数据。非有限输入或结果仍返回状态 **2**。
CBLAS 矩阵乘法本身返回 void，因此适配器不承诺捕获库内部终止进程等所有失败。

零行、零列和零批次不调用库；K=0 使用原有代码输出正零。
即使“必须外部库”，这些特殊情况也本地处理。
外部接口仍只接受原始行布局，不传入分块排布的权重；超过当前对齐保证的实现仍不选择。
`measured` 模式允许不同节点分别保留原始和分块布局，复用已有权重准备。
外置权重的 RISC-V 运行器可使用 1／2／4 hart。NMSIS 默认仍在主 hart 同步串行调用，
原有分块乘法和满足并行条件的 Softmax 复用已有工作 hart；这个默认方式不要求库支持并发。
若设置 `nmsis_matmul(..., parallel_rows=True)`，满足条件的 NMSIS MatMul 也按输出行并行。
这一选项要求库允许同时调用并保持每个输出的计算顺序，目前只对本次 NMSIS 标量路径做了验证。
每个并行步骤结束后会等待全部工作 hart 完成，再执行后续算子。
通过既有 `build_external(..., harts=2)` 或 `harts=4` 选择数量，无需更换外部函数接口。

OpenBLAS 当前只声明支持开发电脑，不能用于项目无完整操作系统的 RISC-V 目标。
它可能使用内部临时缓冲区和线程状态，这些资源不包含在 ScratchV 的工作区大小里。
适配器不修改库的全局线程设置。本轮验证固定为单线程，不承诺其他线程配置的结果逐位相同。
这也是相对最初设计的明确取舍：OpenBLAS 的内部运行状态不满足“完全无可变全局状态”的保证；
需要严格无此类运行时依赖的目标，使用本次 NMSIS 标量路径或受控 C 实现。

## 编译和依赖

继续复用三个构建入口：

- [Host `build_library`](../probes/w4_qwen3_full/host_diagnostic.py)：接受外置权重运行接口。
- [`build_riscv_tensor`](../scratchv/runtime/riscv_tensor.py)：RISC-V 内嵌权重入口。
- [`build_external`](../scratchv/runtime/riscv_external.py)：RISC-V 外置权重入口。

OpenBLAS 使用第一个入口；NMSIS-DSP 使用后两个入口。
第一批受控 C 实现继续用于独立检查错误传播和构建依赖。

NMSIS 的构建输入是项目适配源码、官方单个算子源码，以及
`NMSIS/DSP/Include` 和 `NMSIS/Core/Include`；使用现有交叉编译器，
和模型一起编译为静态 RV64 程序。不链接开发电脑上的库。

源文件、头文件目录和库文件通过第一批的 `ExternalMatMul` 配置传入；
也保留 `extra_sources`、`extra_cflags`、`extra_ldflags`。
相对源文件路径从调用构建函数时的目录解析，库排在引用它的对象之后。
直接生成 C 后自行构建时，也必须连接对应库并保留 `artifact.compile_flags`。

模型和适配源码继续使用严格浮点编译参数，避免无关操作改变。
这些参数不能改变已经编译好的 OpenBLAS 内部计算方式。

Ubuntu 可使用发行版的 `libopenblas-pthread-dev` 开发包。
也可以只在本地展开开发包及运行时依赖，再显式指定目录。
使用标准开发包，不依赖 NumPy 的私有符号；参考
[OpenBLAS 构建选项](https://www.openmathlib.org/OpenBLAS/docs/build_system/)。

## 定向验证与已有证据

在项目既有 Python 和 C 编译器环境中，先设置库目录。以下是 Ubuntu x86-64
系统安装 pthread 开发包时的常见路径；本地展开包时换成对应目录：

```bash
export SCRATCHV_OPENBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/openblas-pthread
export SCRATCHV_OPENBLAS_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu/openblas-pthread
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest tests/test_tensor_c_openblas.py -q -s
```

如果只是把 Ubuntu 的包展开到本地目录，还需让动态加载器找到同时展开的运行时依赖，
例如 `libgfortran.so.5`。仅设置头文件和 OpenBLAS 链接目录不够；可按自己的解包目录设置：

```bash
scratchv_openblas_root="$PWD/output/openblas/root"  # 换成自己的解包目录
export SCRATCHV_OPENBLAS_INCLUDE_DIR="$scratchv_openblas_root/usr/include/x86_64-linux-gnu/openblas-pthread"
export SCRATCHV_OPENBLAS_LIBRARY_DIR="$scratchv_openblas_root/usr/lib/x86_64-linux-gnu/openblas-pthread"
export LD_LIBRARY_PATH="$scratchv_openblas_root/usr/lib/x86_64-linux-gnu:$SCRATCHV_OPENBLAS_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

这适用于本文使用的 Ubuntu x86-64 包目录，其他安装位置应对应调整。
通过系统包管理器正常安装 OpenBLAS 及其依赖时，无需添加这组本地解包路径。
同步后的首次集成检查曾遗漏此环境设置，失败及单项复验均已保留在验收摘要中。

线程变量和库搜索路径必须在启动 Python、加载库之前设置。参考
[OpenBLAS 运行时变量](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)。
未配置可选库时，真正依赖库的用例会明确跳过，选择逻辑用例仍执行；
显式配置了错误目录或无法链接时，测试失败。

数值参考使用 Python `math.fsum` 逐项计算，不调用 NumPy 矩阵乘法，
避免用同一个库给自己验证。运行前固定标准：

- 普通样本：绝对误差 ≤ `1e-4 + 2e-5 × |参考值|`。
- 强抵消样本：绝对误差 ≤ `gamma_K × sum(|各乘积|)`，
  其中 `gamma_K = K × u / (1 - K × u)`，`u = 2^-24`。

本轮 Ubuntu 24.04 / OpenBLAS 0.3.26（发行版包 0.3.26+ds-1ubuntu0.1）、
LP64、Cooperlake、实际单线程的小规模结果：

| 样本 | 最大绝对误差 | 占预定误差界的最大比例 |
| --- | --- | --- |
| 普通矩阵、向量及广播，K 覆盖 127/128/129/1024 | 3.4149209e-5 | 11.81% |
| 强抵消样本 | 1 | 0.39% |

强抵消的尺度误差界不保证很小的相对误差，也不能作为完整模型的验收标准。
这里的标准只用于新增外部库测试；原有顺序累加和分块乘加的验证标准未放宽。

第一批回归入口仍为：

```bash
python -m pytest tests/test_tensor_c_external_matmul.py tests/test_external_matmul_build.py -q
```

[受控 C 测试实现](../tests/fixtures/external_matmul.c) 的计数和错误注入仅用于测试。

## RISC-V 定向验证

启用既有 RISC-V 编译器与 QEMU 环境后运行：

```bash
export SCRATCHV_NMSIS_ROOT="$PWD/output/nmsis/NMSIS"
python -m pytest tests/test_tensor_c_nmsis.py -q -s
```

未配置可选依赖时，真正依赖库的用例明确跳过；显式路径或工具配置有误时失败。
数值参考仍复用上面的独立 `math.fsum` 计算与预定误差标准。
验证使用真正的 RISC-V ELF 和 QEMU 执行，包括内嵌及外置权重两种入口，
不是在开发电脑上调用同名替代函数。

本轮使用 NMSIS-DSP 1.6.0、Zig 0.14.1，在 RV64GC / LP64D / QEMU virt
单核环境完成 19 项 NMSIS 测试。与第一批接口、构建测试及第二批真实 OpenBLAS
测试一起运行，共 **94 项通过**。未配置可选库与 QEMU 时，
纯配置检查继续执行，依赖真实工具的部分明确跳过；CI 则要求代表性真实用例必须执行。

| 样本 | 最大绝对误差 | 占预定误差界的最大比例 |
| --- | --- | --- |
| 普通矩阵、向量及广播，K 覆盖 127/128/129/1024 | 4.5593301e-5 | 18.39% |
| 强抵消及混合尺度样本 | 1 | 0.39% |

强抵消结果符合尺度相关误差界，并不表示小相对误差或完整模型精度已经验收。

测试还检查了真实 ELF 中的 NMSIS 函数符号、同一 ELF 更换输入、
零维度不链接库、非有限输入与溢出失败、65536 维度拒绝或回退、
适配函数空指针检查、正零输出、输入只读和输出覆盖。
安装包已确认同时包含 OpenBLAS 与 NMSIS 适配源码，不夹带第三方库实现。

以下是相关回归入口；真实 OpenBLAS 用例还需按前文配置其开发包目录：

```bash
python -m pytest tests/test_tensor_c_nmsis.py \
  tests/test_tensor_c_external_matmul.py tests/test_external_matmul_build.py \
  tests/test_tensor_c_openblas.py -q
```

## Issue 整体验证（2026-10-23）

逐条核对 Issue #109 后，首期调用、选择、数值许可和构建路径均已实现。
本次补齐三项小规模运行证据：通过通用 `CompilerConfig` 从 ONNX 生成并执行 NMSIS，
以及将外部 C 算子预编译为对象文件和静态库后，分别在两种 RISC-V 权重入口中运行。
不依赖把源码重新一起编译来冒充预编译库链接。

| 验收内容 | 当前证据 |
| --- | --- |
| 默认行为、三种选择、逐节点混合使用、失败不静默回退 | 原有真实调用测试及本次回归通过 |
| Host 成熟库与 RISC-V 实际第三方库 | OpenBLAS、NMSIS 小规模数值与边界检查通过 |
| 通用编译入口、源码/对象/静态库、两种权重方式 | 实际构建和执行通过 |
| 全功能相关回归 | 外部算子 94 项 + 核心生成器及运行时 396 项，共 490 项通过 |
| 现有 CI | 已在 W4 小图任务接入库依赖、路径路由和必须执行的检查；本地配置测试通过，远端尚未运行 |

CI 使用发行版 LP64 OpenBLAS 和固定 NMSIS 1.6.0 源码，沿用现有 JUnit 检查：
代表性用例缺失、失败或跳过都不能通过。没有增加新的工作流，也没有自动启用完整模型运行。
本次仍只做本地修改，没有推送或创建 PR。

### 完整 Host 模型：已实际运行并通过

使用项目固定的 28 层 Qwen3、长度 256 的 `full_seed_0` 输入。
明确设置 `require_external` 和 `allow_provider_fp32`，
**254 个 MatMul 全部选择 OpenBLAS**。使用当前分支源码、OpenBLAS 0.3.26、单线程，
并重新计算 ONNX Runtime 参考输出。

对全部 **38,895,616** 个输出值执行原有严格门槛：
最大绝对误差必须小于 `0.001`，不使用相对误差放宽。

| 指标 | 实测 |
| --- | --- |
| 最大绝对误差 | 8.6784363e-5 |
| 均方根误差 | 6.1010377e-6 |
| 达到或超过门槛的元素 | 0 |
| 单次完整前向耗时 | 7.73 秒 |
| 编译耗时 | 129.14 秒 |

本地脚本和结果保存在 `output/issue109-overall-host/`，复用现有解析、
Host 构建、缓冲区、ORT 参考与比较函数。原有完整模型命令没有外部库参数，
直接运行其默认命令不能证明外部库参与。

这是一例完整前向的数值证据，不等同于全部七个固定用例、连续生成或语言质量验收。
耗时仅为本次测量；没有据此宣称相对原实现的公平性能提升。

### 完整 RISC-V/QEMU 模型：混合调用已实际运行并通过

使用同一完整模型和 NMSIS-DSP 1.6.0，`prefer_external` 选择结果为：
**253 个 MatMul 调用 NMSIS，最后 1 个投影使用原实现**。
原因是词表输出维度 `151936` 大于 NMSIS 的每维上限 `65535`。

这次完整运行的 `matmul_policy="sequential"`，最后一层采用顺序累加。
下面的精度和耗时记录属于这个配置；后续改用 `blocked_fma` 的结果单独记录，
不能直接继承本次完整模型验收结论。

`require_external` 已验证会在生成时明确失败。当前接口无法承诺整个 Qwen3
都使用 NMSIS；若未来要求这一点，需要另选支持大维度的实现，或单独设计并验证分块适配。

完整图已通过实际交叉编译，ELF 中同时确认了适配函数和官方 NMSIS 函数，
构建结果位于 `output/issue109-overall-riscv/`。本轮复用该构建；运行前已确认
当前分支生成的模型代码、启动代码、内存布局和逐节点选择一致，并将权重包中的
每项数据与当前解析模型直接比较。没有重新编译或另跑默认计算路径。

随后实际执行了完整 28 层 Qwen3 的 `full_seed_0`，输入长度为 256，
使用 RV64GC / LP64D、QEMU virt 单核和 NMSIS-DSP 1.6.0。
外部库数值选项为 `allow_provider_fp32`；完整模型验收仍沿用原有严格门槛，
没有放宽误差。输入已逐数组核对，与前轮完整 ONNX Runtime 参考输入完全一致。

QEMU 正常完成并以 0 退出。对形状 `[1, 256, 151936]` 的全部
**38,895,616** 个输出检查有限性并逐元素比较参考结果：

| 指标 | 实测 |
| --- | --- |
| 最大绝对误差 | 0.0001220703125 |
| 原有门槛 | 最大绝对误差严格小于 0.001，相对容差为 0 |
| 均方根误差 | 8.8258389e-6 |
| 达到或超过门槛的元素 | 0 |
| QEMU 启动至完成帧校验的耗时 | 11288.77 秒，约 3 小时 8 分钟 |
| 本次验证总耗时 | 11328.58 秒，约 3 小时 9 分钟 |

复现脚本、输入、实际输出和最终记录位于 `output/issue109-full-qemu/`，
其中 `result.json` 为完整比较结果，`qemu/run.json` 为运行记录，
`qemu/output.bin` 为完整 FP32 输出。参考输出保存在
`output/issue109-overall-host/ort.npy`。复用现有运行器和全输出比较函数，
临时权重运输副本由现有运行器在成功后清理，原始权重包继续保留。

该结果证明：**允许最后一层保留原实现时，当前混合调用可以完成这个固定输入的
完整模型，并满足原有精度要求。** 这仍是一次完整前向，不涵盖全部七个固定用例、
连续生成或真实硬件。耗时只记录本次模拟器运行，不用于宣称性能提升。

## 代表矩阵性能与分块回退（2026-10-24）

本轮比较三种配置：原有顺序累加、原有分块乘加、优先 NMSIS 且回退时使用分块乘加。
目标是确认保留外部库调用时，最后一层能否恢复原有优化。已有代码支持这个组合，
本轮补充混合调用回归用例、生成结果和计时记录，没有新增一套矩阵乘法实现。

### 相同条件下的结果

三种配置使用相同输入、权重、内存布局和工具链：
NMSIS-DSP 1.6.0、Zig 0.14.1、QEMU 8.2.2、RV64GC / LP64D、单核、`-O2`、
外置权重。每种配置先预热一次，再交替顺序测量三次；以下为中位耗时，单位秒。

| 代表矩阵 | M × K × N | 原顺序累加 | 原分块乘加 | NMSIS + 分块回退 |
| --- | --- | ---: | ---: | ---: |
| Q 投影 | 8 × 1024 × 2048 | 1.470 | 0.409 | 1.305 |
| 注意力输出投影 | 8 × 2048 × 1024 | 1.511 | 0.489 | 1.372 |
| 前馈升维 | 8 × 1024 × 3072 | 1.980 | 0.639 | 2.012 |
| 前馈降维 | 8 × 3072 × 1024 | 1.914 | 0.604 | 1.926 |
| 注意力 QK，单个头 | 256 × 128 × 256 | 0.090 | 0.069 | 0.091 |
| 注意力 AV，单个头 | 256 × 256 × 128 | 0.096 | 0.074 | 0.094 |
| 最后投影，单行 | 1 × 1024 × 151936 | 13.672 | 4.741 | 4.886 |

最后一列前六项已通过选择记录和 ELF 函数符号确认实际调用 NMSIS；
最后一项因维度超限使用原分块代码。这个末层生成的 C 与直接选择原分块实现完全一致。
两者本次耗时范围分别为 4.351–5.559 秒与 4.364–4.798 秒，
中位数的小幅差异不足以证明外部调用配置引入固定开销。

**末层单行从顺序累加改为分块回退后，耗时约减少 64.3%。**
其余六类矩阵中，NMSIS 耗时仍为原分块实现的约 **1.27–3.19 倍**。
NMSIS 与原顺序累加处于同一量级，三次样本及重叠的耗时范围不足以证明稳定优势。
因此，当前可保留外部库调用并改善末层耗时，但不能据此宣称完整模型恢复了原分块方案的速度。

### 数值和选择检查

7 类矩阵、3 种配置共完成 84 次小图运行，包括 21 次预热和 63 次计时，全部通过。
每次均检查有限值，并与单线程 ONNX Runtime 及 FP64 参考比较。
最大 ORT 绝对误差为 **8.1062317e-6**，严格小于原门槛 `0.001`，
达到或超过门槛的元素为 0；所有结果也满足既有
`1e-4 + 2e-5 × |参考值|` 误差界，最大只占该界的 5.82%。

另外，5 项定向回归检查通过。其中新增的小图在同一程序里先调用真实 NMSIS，
再回退到分块实现，覆盖 K=129 和列尾部，实际在 QEMU 中执行；
对独立参考的最大绝对误差为 3.0665876e-7。

代表矩阵测试阶段，完整 Qwen3 只重新生成代码并检查选择：
仍为 **253 个外部调用 + 1 个原实现**，最后的 `v_5909` 使用分块乘加。
随后按用户要求完成了这个配置的一次完整运行，结果见下节；
此前顺序回退和代表矩阵的数据均保留。

### 计时范围与本地记录

计时复用现有 `forward_wall_s`，从启动 QEMU 到收到完成帧，
包含加载、模拟器即时编译和程序检查；不含模型编译、启动前权重准备、
输出转存和参考计算。每个样本都会新启 QEMU，所以不是纯算子内核计时，
短矩阵尤其容易受启动开销和机器负载影响。

输入和权重均为固定随机数据。前四项保留模型的 K、N，将 M 从 256 减到 8；
末层保留完整 K、N，仅运行一行；注意力采用一个头的矩阵尺寸，
但本次右矩阵存为外置权重，完整模型中它来自中间结果。
这些取舍适合定向比较，不能按行数直接推算完整模型耗时。
分块乘加会改变末层的浮点计算顺序，因此另行完成了下节记录的完整模型数值检查；
代表矩阵的耗时不能代替完整模型的实际记录。

本地保留以下文件，未提交生成产物：

- `output/issue109-matmul-performance/benchmark.py`：复用现有构建和运行入口的基准脚本。
- `output/issue109-matmul-performance/measurements/report.json`：所有计时样本、误差、选择与汇总。
- `output/issue109-matmul-performance/measurements/`：各矩阵的输入、权重、参考输出、构建记录和实际输出。
- `output/issue109-matmul-performance/full-fallback-check/`：完整图生成结果、末层代码及定向测试日志。

## 完整模型使用分块回退（2026-10-24）

**253 个 NMSIS 调用加最后一层原分块优化，已完成完整 Qwen3 并通过原误差门槛。**
本次使用当前本地分支的实现，重新编译完整程序，沿用 28 层模型、
长度 256 的 `full_seed_0`、原始权重和同一份 ONNX Runtime 参考输出。
输入已逐数组确认与此前完整运行一致。NMSIS-DSP 1.6.0、Zig 0.14.1、
QEMU 8.2.2、RV64GC / LP64D、单核保持不变。

选择仍为 253 个外部实现和 `v_5909` 一个原实现，最后一层改用
`blocked_fma`、`1x4` 分块和原始权重布局。新 ELF 中确认了 NMSIS 函数，
生成的末层代码确认使用分块乘加。没有为补计时重跑模型。

### 精度与前后变化

全部 **38,895,616** 个 FP32 输出均为有限值，逐元素使用原有
`max_abs < 0.001`、`rtol=0` 标准比较，没有放宽门槛。

| 指标 | 此前：末层顺序累加 | 本次：末层分块乘加 |
| --- | ---: | ---: |
| 最大绝对误差 | 0.0001220703125 | 0.0001220703125 |
| 均方根误差 | 8.8258389e-6 | 8.5976271e-6 |
| 达到或超过误差门槛的输出 | 0 | 0 |
| 完整前向耗时 | 11288.77 秒，3 小时 8 分 9 秒 | 估算 9364.47 秒，2 小时 36 分 4 秒 |

按这些记录粗略比较，前向约少 **1924.30 秒，即 32 分 4 秒，约 17.05%**。
这支持末层分块回退能缩短本次混合调用的运行时间，但不能作为严格、可重复的提速保证，
也不表示其余 NMSIS 算子已经达到原分块实现的速度。

本次从解析、编译到比较结果保存共约 **9478.37 秒，即 2 小时 37 分 58 秒**，
其中编译约 86.70 秒；该总时间由保存的开始和结束 UTC 时间重建，不含随后整理文档与归档。
此前脚本复用了已编译程序，因此两个脚本的总耗时不能直接用于比较计算速度。
本次验证于北京时间 12:19:24 完成。

### 计时接管影响

原结果收集进程在计算期间退出，QEMU 本身一直继续运行。独立服务接管了同一个进程，
等待原有完成帧，再通过现有 QMP 接口转存完整输出并复用原比较函数。
没有中断或重新执行模型，数值检查针对这次实际生成的完整输出。

原始“启动 QEMU 前”的计时起点已丢失，记录中的
`qemu.forward_wall_s_estimate` 从 Linux 记录的进程启动时间重建，
终点是收到并检查完成帧。启动时间记录的粒度为 0.01 秒，
但这不能恢复原始计时起点，也不能消除不同运行时机器负载的差别。
因此上述耗时和百分比必须标为估算或粗略比较，不能表述为严格同口径性能基准。

接管进程无法读取原 QEMU 的操作系统退出码，故 `exit_code=null`；
已确认完成帧正确、输出完整、QMP 退出响应收到并观察到进程结束。
不将它记作“退出码为 0”。运输用权重分片继续保留，
并通过原有结束校验，没有因接管而丢弃运行数据。

### 已保存的数据

本次目录为 `output/issue109-full-qemu-blocked/`：

- `result.json`、`qemu/run.json`：最终数值、估算耗时、实现选择和接管限制。
- `inputs.npz`、`qemu/inputs.bin`、`qemu/output.bin`：输入和全部实际输出；输出为 155,582,464 字节。
- `build/`：新 ELF、生成的 C、编译命令、内存布局和编译日志。
- `full_qemu.py`、`recover_collector.py`、`run.log`、`recovery.log`、
  `result-before-recovery.json`：运行入口及接管前后的记录。

原始权重仍在 `output/issue109-overall-riscv/weights/`，
参考结果仍在 `output/issue109-overall-host/ort.npy`。
本次归档同时包含这些复用数据、模型、源代码、全部新运行数据和历史比较依据；
使用已有归档脚本写入普通 tar，再直接回读比较并落盘，不新增哈希体系。
该结果仅覆盖一个固定输入的完整前向，不代表七个用例、连续生成或真实硬件都已验证。

## NMSIS 串行与原算子多 hart 混合运行（2026-10-24）

本轮完成第一步：允许外部 MatMul 与已有多 hart 运行器同时使用。
NMSIS 始终由主 hart 按图中的顺序同步调用；其他 hart 只执行原有分块乘法和
达到并行门槛的 Softmax。复用原有调度及结束等待，没有改写 NMSIS 算法、
拆分它的计算，也没有新增调度框架。默认仍为 1 hart，需要在构建时显式选择 2 或 4。

实现只解除构建和生成代码中对外部调用的整体多 hart 限制。
分块乘法按输出块分工，Softmax 按独立行分工，保留每个输出原来的累加顺序。
外部调用仍由图入口同步调用的普通内核执行，不放入工作 hart 的回调；
前面的并行步骤全部结束后，才会进入下一次库调用。
同一张图仍允许维度超限的 MatMul 回退到原分块实现。

### 精度与错误处理

本轮定向回归共 **57 项通过**：56 项接口、构建和主机执行检查，加 1 项真实
NMSIS/QEMU 混合回归。后者实际启动 QEMU 5 次，包含 1／2／4 hart 成功运行、
4 hart 下的预期库调用失败，以及使用相同 ELF 的新运行恢复。
3 项原有通用 QEMU 冒烟没有重复执行。

真实混合回归依次执行 NMSIS、原分块乘法、Softmax、再次 NMSIS。
测试适配函数读取实际 hart 编号，确认两次库调用均在 hart 0 且顺序正确，
同时覆盖分块尾部、17 行 Softmax 的不均匀分工，以及并行后的外部错误传播。
1／2／4 hart 结果逐位相同，对 FP64 参考的最大绝对误差为 **9.7234085e-9**。

另完成四类性能用例，每类每个 hart 数先预热一次，再轮换顺序测量三次，
共 **48 次小图运行**。全部 FP32 输出有限，所有运行与对应的 1 hart 输出逐位相同。
相对单线程 ONNX Runtime 的最大绝对误差为 **4.7683716e-6**，
严格小于原门槛 0.001，相对容差为 0，超限元素为 0。
全部通过既有 FP64 误差界 `1e-4 + 2e-5 × |参考值|`，最大占该界的 2.83%。

### 同条件耗时比较

以下单位为秒，列出三次计时的中位数。所有配置复用同一图、输入和权重，
保持 RV64GC / LP64D、原始权重布局、1×4 分块、外置权重和相同编译选项。

| 用例 | 尺寸或计算链 | 1 hart | 2 hart | 4 hart |
| --- | --- | ---: | ---: | ---: |
| 串行 NMSIS 投影，对照项 | M=8，K=1024，N=2048 | 1.3831 | 1.3564 | 1.4092 |
| 最后投影，单行分块回退 | M=1，K=1024，N=151936 | 4.7135 | 2.9768 | 2.1965 |
| Softmax，单个注意力头 | 256×256 | 0.02375 | 0.02184 | 0.02177 |
| 混合小图 | NMSIS → Softmax → 分块回退 | 1.7967 | 1.0418 | 0.6415 |

最后投影单行的 4 hart 耗时约为 1 hart 的 46.6%，即本次约 **2.15 倍速度**；
混合小图约 **2.80 倍速度**。这说明在保留串行外部库调用时，已有并行部分
可以缩短本轮小图的总耗时。串行 NMSIS 对照的耗时范围重叠，没有获得可确认的提速。
Softmax 耗时仅约 22–24 毫秒，范围也重叠，不能把小幅中位数差别当作稳定收益。

末层单行保留真实的 K、N，N=151936 超过 NMSIS 的 65535 限制，实际选择原分块代码。
混合小图输入为 17×129，第一次权重为 129×2049，Softmax 为 17×2049，
最后权重为 2049×2051。为了以较小数据验证选择过程，仅在这个实验的库描述中将
维度上限设为 2049，使最后一步回退；没有修改 NMSIS 本身的上限或正式适配器。

计时复用运行器原有 `forward_wall_s`，由启动 QEMU 前的单调时钟量到完成帧校验，
包含启动、加载、即时编译和计算，不含构建、启动前权重准备、输出转存和参考计算。
本轮计时起点完整保留，没有发生进程接管。每个样本启动新的 QEMU，预热仅影响
主机文件和缓存。按现有运行器，2／4 hart 使用 QEMU 多线程模式，
并为每个 hart 保留 1 MiB 栈；各构建的实际内存布局均已保存。

这里只比较一次实验中的三个样本，输入为固定随机数据，投影行数和混合图均有缩减。
**小图阶段没有运行多 hart 完整模型；随后完成的完整验证见下节，不能把小图的 2.15 或 2.80 倍换算成完整模型耗时。**
NMSIS 的 253 个调用仍串行；完整模型的收益会受这部分计算占比限制。
此前单 hart 完整模型的精度、耗时及计时接管说明保持有效。

### 本地复现与数据

沿用前文的编译器、QEMU、NMSIS 路径配置，定向回归入口为：

```bash
python -m pytest -q tests/test_external_matmul_build.py \
  tests/test_tensor_c_external_matmul.py \
  -k "not rv64_external_c_function_smoke and not rv64_precompiled_provider_executes"
python -m pytest -q -s \
  tests/test_tensor_c_nmsis.py::test_nmsis_main_hart_between_parallel_kernels
```

本轮证据保存于 `output/issue109-multihart/`：

- `benchmark.py`：复用现有构建、运行和参考计算的本地实验脚本。
- `measurements/report.json`、`benchmark.log`：全部计时、误差、实现选择及汇总。
- `measurements/`：每项输入、权重、ORT/FP64 参考、生成代码、ELF、实际输出及运行记录。
- `unit-tests.log`：56 项测试已观察输出的转录，注明未为保存日志重复测试。
- `nmsis-pytest-final.log`、`nmsis-pytest-final/`：真实混合回归日志、构建和运行数据；
  测试搭建阶段的日志另行保留。

本地性能脚本固定读当前工作树的 `output/nmsis/NMSIS`，默认写入尚不存在的
`measurements/` 目录，并拒绝覆盖。原始实测目录继续保留；如需后续新实验，
先修改脚本输出位置，避免覆盖本次证据。小图阶段没有重跑完整模型；后续完整验证单独记录如下。
所有工作仍保留在本地，没有提交、推送或创建 PR。

## 完整模型使用 4 hart（2026-10-24）

**4 hart 完整 Qwen3 已通过原误差门槛，全部输出与单 hart 分块配置逐位一致。**
本次在北京时间 16:21:56 完成，仍使用 28 层模型、长度 256 的 `full_seed_0`，
相同输入、权重、ONNX Runtime 参考和工具链。只运行了一次，没有为补计时重跑。

配置保留 253 个选择 NMSIS 的 MatMul 节点，在主 hart 同步串行执行；
最后的 `v_5909` 仍因维度上限回退到 `blocked_fma`、1×4 分块及原始权重布局。
生成代码确认有 28 个 Softmax 和一个末层分块的并行调度点，工作回调中没有 NMSIS 调用。
与之前单 hart 生成的模型 C 相比，只移除了已验证的多 hart 编译限制，
没有为完整运行改写算子算法。新 ELF 已确认真实 NMSIS 函数，
QEMU 实际使用 `-smp 4 -accel tcg,thread=multi`。

### 精度与单 hart 输出变化

全部 **38,895,616** 个 FP32 输出均为有限值。验收仍为每个输出
严格满足 `max_abs < 0.001`、`rtol=0`，没有放宽门槛。

| 指标 | 单 hart 分块配置 | 本次 4 hart |
| --- | ---: | ---: |
| 对 ORT 最大绝对误差 | 0.0001220703125 | 0.0001220703125 |
| 对 ORT 均方根误差 | 8.597627126070227e-6 | 8.597627126070227e-6 |
| 达到或超过 0.001 的输出 | 0 | 0 |

除比较参考结果外，还读取已保存的单 hart 完整输出逐元素比较。
两者最大差值、均方根差值均为 0，并通过分块逐位检查，
因此可明确确认本次 4 hart 配置没有改变这个固定输入的输出。
QEMU 完成帧有效、全部输出转存成功、正常退出码为 **0**，收集服务也已成功结束。

### 耗时与计时口径

| 指标 | 单 hart 分块配置 | 本次 4 hart |
| --- | ---: | ---: |
| QEMU 前向耗时 | 估算 9364.47 秒，2 小时 36 分 4 秒 | 9249.00 秒，2 小时 34 分 9 秒 |
| 验证总耗时，含构建和比较 | 重建 9478.37 秒，2 小时 37 分 58 秒 | 9382.71 秒，2 小时 36 分 23 秒 |
| 编译耗时 | 86.70 秒 | 93.86 秒 |

按前向记录粗略比较，本次少 **115.47 秒，即约 1 分 55 秒、1.23%**，
速度比约 1.0125。**本次完整模型没有显示出小图那样的大幅提速，也不足以确认稳定收益。**
253 个 NMSIS 节点仍串行执行，并行仅覆盖已有的末层和 Softmax。
本轮没有逐算子计时，不能精确划分各部分的耗时或把全部差值归因于多 hart。

本次从开始就由独立用户服务保留原 Python 收集进程，直接使用原始
`qemu.forward_wall_s`：从启动 QEMU 前的单调时钟到收到并校验完成帧，
包含加载、模拟器即时编译和计算；不含启动前的权重运输准备、输出转存及数值比较。
本次没有发生计时接管，验证总耗时同样来自原始单调时钟起点，
包含解析、编译、运行和比较，不包含之后整理文档与归档。

单 hart 基准的 `forward_wall_s_estimate` 则由进程启动 ticks 重建，
总时间由保存的 UTC 起点重建；它在接管后无法取得操作系统退出码，
原记录 `exit_code=null` 继续保留，不改写为 0。
两个配置还使用不同 QEMU 线程模式及 1／4 MiB 栈预留，
历史主机负载没有控制，且各只有一次完整测量。
因此 1.23% 只能作为历史记录差异，不能表述为严格同口径、可重复的性能提升。

### 完整数据保存

本次运行目录为 `output/issue109-full-qemu-4hart/`，旧运行和归档未覆盖：

- `result.json`：精度、单 hart 输出比较、实现选择、前向/总耗时与计时限制。
- `qemu/run.json`、`qemu/uart.bin`、`qemu/qemu.stdout`、`qemu/qemu.stderr`：完成和退出记录。
- `inputs.npz`、`qemu/inputs.bin`、`qemu/output.bin`：输入与全部实际输出；输出为 155,582,464 字节。
- `build/`：ELF、生成的 C、启动代码、编译命令、布局及日志。
- `full_qemu.py`、`run_service.sh`、`launch.json`、`run.log`：运行脚本、启动身份与完整日志。
- `archive_results.py`：复用普通 tar 及直接回读比较的归档脚本，不新增哈希体系。

原始权重继续保存在 `output/issue109-overall-riscv/weights/`，
ORT 基准在 `output/issue109-overall-host/`；
本次正常退出后，现有运行器完成运输校验并清理临时分片，原始权重未删除。
归档同时收录新运行、源代码、模型、NMSIS、权重、ORT 基准，
以及单 hart 实际输出和接管元数据。
归档收据保存在运行目录外的 `output/issue109-full-qemu-4hart-archive-receipt.json`，
记录实际文件路径、大小及直接回读核对结果。

这次结论只覆盖一个固定输入的完整前向，不代表七个用例、连续生成、其他库或真实硬件验收。
已有代码改动保留在本地，没有提交、推送或创建 PR。

## NMSIS 按输出行并行调用（2026-10-24）

第二步在现有接口中增加显式选项：多个 hart 分别计算一段输出行，
每段都调用原来的 NMSIS 函数。复用现有工作分配和结束等待，
没有复制矩阵、重写库算法或增加新的调度器。默认关闭，原有串行调用方式保持不变。

### 用法和边界

在前文 NMSIS 示例中只需改为：

```python
provider = nmsis_matmul("/path/to/NMSIS", parallel_rows=True)
```

后续仍将它传给 `TensorCCodegen(..., external_matmul=provider)`，
并通过既有 `build_external(..., harts=2)` 或 `harts=4` 构建。
数值许可、回退策略和链接依赖沿用前文配置；没有新增命令行入口。
通用描述字段 `ExternalMatMul.parallel_rows` 默认为 `False`，
OpenBLAS 工厂不会自动打开它。

当前只在外置权重路径、输出行数不少于 hart 数，且每组
`M × N × K >= 32768` 时分工。这个工作量下限是保守保护，并非自动寻找最佳性能。
单 hart、行数不足、小矩阵及内嵌权重仍串行。
每个 hart 读取对应的 A 行，共享只读 B，写互不重叠的 C 行；
每个输出仍计算完整 K，未拆分累加过程。多组矩阵和广播逐组处理，
例如 17 行在 4 hart 下分为 5／4／4／4 行。

选择外部实现时仍先检查完整 M、N、K，维度上限保持 65535，
不会通过拆行绕过限制；完整模型末层仍应回退到原分块实现。
通用外部库只有确认可同时调用、共享输入不被修改、拆行不改变每个输出的运算，
且不会在内部嵌套使用 ScratchV 工作 hart，才可以声明这个选项。
当前 NMSIS 标量实现使用局部矩阵描述和局部计算状态，符合这些条件。

任何 hart 返回错误，均先等待这次所有调用结束，再返回图状态 4；
不执行后续算子、不重算原实现、不复制未完成输出。
非有限结果继续返回状态 2。

### 定向验证

本轮 **78 项测试通过**：75 项接口、构建和主机执行检查，以及 3 项真实 NMSIS/QEMU 检查。
另有 3 项原有通用 QEMU 冒烟按范围未重复运行。
3 项真实检查实际启动 QEMU 13 次，包含两次预期错误。
主机与 QEMU 定向检查合计覆盖：

- 默认串行方式兼容，1／2／4 hart 行并行及不均匀分工。
- 多组矩阵和广播、同一输入同时用作 A/B、输入保持不变。
- NMSIS → 原分块乘法 → Softmax → NMSIS 的混合计算。
- 工作 hart 报错后等待其他调用完成，不运行后续算子；使用相同程序的新运行可恢复。
- 内嵌权重、小矩阵、行数不足等串行路径，以及输出非有限值的处理。

真实混合图用实际 hart 编号和每个 hart 的调用计数确认分工，
不只依赖生成代码中的函数名。混合回归对 ORT 最大绝对误差为 **7.6834112e-9**；
广播回归为 **5.7220459e-6**，1／2／4 hart 输出均逐位一致。
通用主机回归另检查同一进程内错误后的恢复。

### 相同条件下的耗时

在同一轮实验中比较按行并行的 1／2／4 hart，以及 **4 hart、NMSIS 仍串行**的对照。
后者保留 Softmax 和原分块乘法的并行，直接对应第一步行为。
每类每种配置预热一次、正式测量三次，轮换测量顺序，只运行一个 QEMU，
没有同时执行编译或其他测试。下表为正式测量的中位秒数：

| 用例 | M × K × N 或计算链 | 1 hart 行调用 | 2 hart 行并行 | 4 hart 行并行 | 4 hart 串行 NMSIS |
| --- | --- | ---: | ---: | ---: | ---: |
| Q 投影 | 8 × 1024 × 2048 | 1.3339 | 0.7215 | 0.4578 | 1.3123 |
| 注意力输出投影 | 8 × 2048 × 1024 | 1.2398 | 0.7009 | 0.3822 | 1.2418 |
| 前馈升维 | 8 × 1024 × 3072 | 1.8790 | 1.0198 | 0.5807 | 1.8560 |
| 前馈降维 | 8 × 3072 × 1024 | 2.0502 | 1.0498 | 0.6620 | 1.9134 |
| 注意力 QK，单头 | 256 × 128 × 256 | 0.0932 | 0.0552 | 0.0505 | 0.0991 |
| 注意力 AV，单头 | 256 × 256 × 128 | 0.0935 | 0.0564 | 0.0434 | 0.0937 |
| 混合小图 | NMSIS → Softmax → 分块回退 | 4.2944 | 2.3025 | 1.3155 | 3.0022 |

六类 NMSIS 矩阵中，4 hart 相对 1 hart 的本次速度比为 **1.85–3.24 倍**；
相对同轮 4 hart 串行 NMSIS 对照为 **1.96–3.25 倍**。
混合图的速度比分别为 **3.26 倍**和 **2.28 倍**。
这些小图表明按行分工能够缩短当前 QEMU 下的耗时，
但不能把这些倍数直接套到完整模型。

7 类、4 种配置共 **112 次运行**，包括 28 次预热和 84 次正式测量，全部通过。
所有 FP32 输出有限，包含串行对照在内的全部输出与各自 1 hart 基准逐位一致。
对单线程 ORT 的最大绝对误差为 **8.1062317e-6**，
严格满足 `max_abs < 0.001`、`rtol=0`，达到或超过门槛的元素为 0。
全部通过既有 FP64 误差界 `1e-4 + 2e-5 × |参考值|`，
最大只占误差界的 **5.82%**。每个实际程序均核对真实 NMSIS 函数，
2／4 hart 行并行程序另确认存在行任务函数，串行对照不存在该函数。

### 尺寸、计时与结论限制

前四类保留模型 K/N，M 从 256 缩为 8；注意力采用一个头的完整矩阵尺寸，
但右矩阵在本次实验中是外置权重，完整模型中它来自中间结果。
混合图为 17×1024 输入，1024×2049 外部乘法，17×2049 Softmax，
再接 2049×2051 分块回退。只有该实验的库描述将维度上限设为 2049，
用于触发回退；正式 NMSIS 接口未改变。
本轮混合图的 K 从第一步实验的 129 增至 1024，所以不同阶段的混合图耗时不能直接比较。

工具链为 NMSIS-DSP 1.6.0、Zig 0.14.1、QEMU 8.2.2、RV64GC / LP64D、`-O2`，
外置权重、原始布局及原实现 1×4 分块。输入和权重是固定随机数据；
同一用例的各配置复用相同数据，4 hart 串行与行并行的内存布局也相同。

本次保留原始 `forward_wall_s`：启动 QEMU 前到完成帧校验，
含启动、加载、即时编译、计算及程序检查；不含构建、启动前权重准备、输出转存和参考计算。
没有发生进程或计时接管。每次启动新的 QEMU，预热仅影响主机文件和缓存。
1 hart 使用原默认线程模式，2／4 hart 使用多线程模式，每个 hart 的栈为 1 MiB。
三次样本轮换四种配置，位置并非完全均衡；计时仍受模拟器启动和机器负载影响，
所有原始样本与范围都已保留。**本轮未运行完整模型，不能宣称完整模型获得相同提速。**
已有完整模型数值和耗时仍属于 NMSIS 串行配置。

### 本地复现与保存

沿用前文工具链和 `SCRATCHV_NMSIS_ROOT` 配置，定向测试入口为：

```bash
python -m pytest -q tests/test_external_matmul_build.py \
  tests/test_tensor_c_external_matmul.py \
  -k "not rv64_external_c_function_smoke and not rv64_precompiled_provider_executes"
python -m pytest -q -s \
  tests/test_tensor_c_nmsis.py::test_nmsis_main_hart_between_parallel_kernels \
  tests/test_tensor_c_nmsis.py::test_nmsis_parallel_rows_join_mixed_graph \
  tests/test_tensor_c_nmsis.py::test_nmsis_parallel_rows_broadcast_on_qemu
```

所有本轮数据保存在 `output/issue109-nmsis-row-parallel/`：

- `benchmark.py`、`benchmark.log`：复用既有入口的性能脚本和完整执行日志。
- `measurements/report.json`：全部样本、误差、选择、符号和时间汇总。
- `measurements/`：输入、权重、ORT/FP64 参考、生成源码、ELF、编译日志及全部实际输出和运行记录。
- `generic-tests.log`、`generic-pytest/`：接口、构建和主机回归的日志与产物。
- `qemu-tests.log`、`qemu-tests/`：真实 QEMU 回归日志、参考结果、程序和运行产物。

性能脚本读取本地 `output/nmsis/NMSIS`，拒绝覆盖已有 `measurements/`；
如需后续实验，先指定新的输出位置。已有工作区改动、此前完整模型数据和归档全部保留，
本轮仅在本地完成，没有提交、推送或创建 PR。

## 保留原优化并按实测选择（2026-10-24）

本轮解决两个问题：外部调用不能连带关闭其他节点的原优化；不能只按“库支持”
就替换更快的原实现。默认行为不变，新增显式的 `measured` 模式。

### 先把比较基准校正

原有 W4 记录确实包含完整模型的快速配置：4 hart、`blocked_fma`、
`2x4` 分块、`column-panel` 权重排布，前向 **410.53 秒**，完整验证
**527.44 秒**，其中编译 **83.12 秒**。证据见
[W4 优化记录](llm-deploy-v1.0/W4/softsim-optimization-2026-10-22.md)和
[完整模型 hart 对比](llm-deploy-v1.0/W4/reports/full-hart-comparison-2026-10-22.json)。

此前外部库的完整模型是另一种配置：253 个 NMSIS 节点串行运行，
只有末层的 `1x4/original` 和 Softmax 使用多 hart。前面的按行并行小图
也主要比较 NMSIS 串行与并行。因此，它们不能说明外部库比原有
`2x4/column-panel` 快。本轮以这个更强的原实现重新进行定向对比。
历史完整模型没有重跑，410.53 秒也不是本轮取得的新成绩。

### 这次代码改变了什么

- 每个 MatMul 独立选择实现。外部库继续读取按行连续的数据；
  其他原实现节点继续使用 `2x4` 和 `column-panel`，无需整张图退回原始布局。
- 复用已有权重排布和多 hart 调度。共享同一权重的节点若会互相影响排布，
  保守保留原实现；不新增重复权重或运行时转换层。
- `measured` 读取调用方提供的 `MatMulPerformance`。记录必须匹配完整输入形状、
  权重来源、实际排布、库描述、目标和生成配置；默认外部耗时至少减少 **10%** 才选择它。
- 未测过、配置失配、库不支持或收益不足时保留原实现，并在
  `artifact.matmul_selections` 中说明原因。扁平数组另带 `m/n/k` 的旧式输入
  也保守回退，避免仅凭数组长度错误复用测量。
- 有效记录对应的 hart 数由现有构建入口和生成的 C 编译检查共同约束。
  即使全部节点保留原实现，也不能把 4 hart 的记录静默用于 1 hart 构建。
  整份记录与生成环境失配时则不附加这项约束，原实现能够正常构建。

`builtin`、`prefer_external`、`require_external` 的语义和默认值不变。
`prefer_external` 仍然表示“支持就优先调用”，不会自动获得性能判断。
NMSIS 和 OpenBLAS 的库内核未改写，项目的分块方式也不会被强行套进库内部。

### 怎样使用

数据结构位于 [external_matmul.py](../scratchv/backend/external_matmul.py)。
每条 `MatMulMeasurement` 保存两侧完整形状、右侧数据类别、原实现的实际布局，
以及同条件测得的两种耗时。`MatMulPerformance` 保存这些记录与对应的库和运行配置。

新检出可先使用 [可迁移示例](../examples/external_matmul_measured.py)，查看随仓库保存的
[Host 历史配置](../examples/external_matmul_profiles/host-historical.json)和
[RISC-V 历史配置](../examples/external_matmul_profiles/riscv64-historical.json)：

```bash
python -m examples.external_matmul_measured --target host
python -m examples.external_matmul_measured --target riscv64
```

默认仅显示历史配置，不加载真实算子库、不生成或执行模型。
需要演示生成时，显式指定 `--generate`，并按前文准备本机库路径，例如：

```bash
python -m examples.external_matmul_measured --target riscv64 \
  --nmsis-root /path/to/NMSIS --generate output/matmul-example/model.c
```

这只生成一个未测尺寸的小图，不编译、不运行；缺少相同形状的测量时保守选回原实现。
生成的 C 用于查看选择结果，示例没有导出外置权重包，也不覆盖已有输出文件。
所有示例输出都会注明 `current_environment_validated=false`，不把历史数据当作本机验证。
额外传入 `--use-historical-profile` 才会明确使用保存的历史测量，且必须与 `--generate`
一起指定。示例的小图仍没有匹配记录，因此仍选原实现；这不是一次性能测试，
也不能据此声称在新机器上复现了历史收益。Host 生成可通过 `--openblas-include`、
`--openblas-lib` 或前文环境变量提供本机路径。示例直接构造现有 `MatMulPerformance`，
没有增加通用配置加载框架。

下面保留原始历史记录的读取方法，适用于已有本地 `output/` 材料及对应库路径的工作区。
这些原始 JSON 含当时的本机依赖路径，不能把它与上面的可迁移摘要文件直接互换：

~~~python
import json
from pathlib import Path
from scratchv.backend.external_matmul import (
    ExternalMatMul, MatMulMeasurement, MatMulPerformance,
)
from scratchv.backend.tensor_c_codegen import TensorCCodegen

saved = json.loads(Path(
    "output/issue109-performance-selection/"
    "riscv-selected-verification/profile.json"
).read_text())["profile"]
saved["provider"] = ExternalMatMul(**saved["provider"])
saved["measurements"] = tuple(MatMulMeasurement(**row)
                              for row in saved["measurements"])
profile = MatMulPerformance(**saved)

artifact = TensorCCodegen(
    program, initializers,
    constant_storage="external", kernel_calls=True,
    matmul_policy="blocked_fma", matmul_tile="2x4",
    matmul_rhs_layout="column-panel",
    matmul_implementation="measured",
    matmul_performance=profile,
    external_matmul=profile.provider,
    external_matmul_numerics="allow_provider_fp32",
    matmul_target=profile.target,
).generate()
~~~

随后沿用 `write_weight_bundle`、`build_external(..., harts=4)` 和 `run_external`。
Host 使用独立的 `host-selected-verification/profile.json` 和已有 `build_library`。
本轮 Host 配置要求应用和 OpenBLAS 均为单线程；运行时实际查询库线程数为 1。

`CompilerConfig` 同步补齐显式设置：
`tensor_constant_storage`、`tensor_kernel_calls`、`tensor_matmul_policy`、
`tensor_matmul_tile`、`tensor_matmul_rhs_layout`、`tensor_qemu_workspace_layout`、
`tensor_matmul_performance`。原默认值保留，调用方需要显式设置同样的优化组合。
没有新增命令行入口。

`environment` 是调用方填写的测量条件说明，不会自动探测或验证机器、库版本、
CPU 负载和库线程数。配置文件中的本地库路径也必须仍有效。
更换库、编译配置或运行条件后，应重新取得对应测量；不能直接复用旧成绩。

### RISC-V/QEMU 实测

两种实现均使用 4 hart、RV64GC/LP64D、同样输入和严格浮点编译参数。
原实现使用 `2x4/column-panel`，NMSIS 使用按输出行并行及原始布局。
每种配置预热一次，再测三次，下面为前向墙钟时间中位数：

| 用例（M×K×N） | 原优化实现／秒 | NMSIS 行并行／秒 |
| --- | ---: | ---: |
| 投影 8×1024×2048 | 0.06360 | 0.41612 |
| 输出投影 8×2048×1024 | 0.07659 | 0.46577 |
| MLP 升维 8×1024×3072 | 0.08771 | 0.62085 |
| MLP 降维 8×3072×1024 | 0.10388 | 0.62800 |
| 单头 QK 256×128×256 | 0.03818 | 0.05405 |
| 单头 AV 256×256×128 | 0.04178 | 0.04714 |
| 混合小图 | 0.32271 | 1.27562 |
| 实际行数投影 256×1024×2048 | 1.04548 | 11.72680 |

这组数据中原实现全部更快；最后一项耗时约为 NMSIS 的 1/11.2。
共 64 次 QEMU 执行全部通过，最大 ORT 绝对误差为
`8.106231689453125e-6`，满足严格 `< 0.001`、`rtol=0`；
对独立 FP64 参考的误差最多占预设预算约 5.82%。

QK/AV 在此使用固定的二维右侧数据，不能把其结果套用于完整模型中的动态、
多批次注意力矩阵。混合图总耗时也不会被当成某个单算子的测量。
实际行数投影原实现的三个样本为约 0.976～2.118 秒，慢样本保留，
未为了改善结论另测；NMSIS 对应约 11.698～11.973 秒。

计时复用原运行器，从启动 QEMU 前到完成帧验证结束，包含模拟器启动开销，
不包含生成、静态权重排布、编译、启动前运输准备、输出转存和参考计算。
这些准备成本分别记录。工具为 Zig 0.14.1、QEMU 8.2.2、`-O2`；
本次没有计时接管，结果仍仅代表本机 QEMU 的定向实验。

### 开发电脑/OpenBLAS 实测

原实现同样启用 `2x4/column-panel`；OpenBLAS 0.3.26 使用原始布局及单线程。
两种实现预热两次，轮换顺序测五轮，每轮调用十次。
下面为每次调用耗时的中位数，单位是毫秒：

| 用例 | 原优化实现／毫秒 | OpenBLAS／毫秒 | 选择 |
| --- | ---: | ---: | --- |
| 投影 8×1024×2048 | 1.70149 | 1.16390 | OpenBLAS |
| MLP 降维 8×3072×1024 | 2.77525 | 1.75255 | OpenBLAS |
| 单头 QK | 0.74253 | 0.16043 | OpenBLAS |
| 单头 AV | 0.86425 | 0.16520 | OpenBLAS |
| 尾部尺寸 17×129×19 | 0.01408 | 0.00393 | OpenBLAS |
| 小矩阵 1×16×16 | 0.00144 | 0.00181 | 原实现 |
| 混合小图 | 1.64485 | 0.26543 | 整图对比，未写入单算子配置 |

共执行 728 次 C 调用，所有返回状态正常；报告保留 98 条数值检查记录，
另保存 84 份输出数组（14 份第二次预热输出、70 份测量批次末输出）。
最大 ORT 绝对误差为 `2.384185791015625e-6`，独立 FP64 误差最多占预算约 1.90%。
Host 计时只围绕连续函数调用，包含 Python 调用边界，不含编译、库加载、
数据准备和参考计算；不能把它与 QEMU 墙钟时间直接比较。
权重排布和准备成本另有记录，以上收益不等同于首次部署总耗时收益。

### 选择逻辑与原优化的验证

将上述单算子的真实中位数读入配置后，额外各执行一次进行正确性检查，
这些执行时间不再用于更新性能结论：

- RISC-V 七种已测矩阵全部保留原实现，不链接 NMSIS；输出与已保存的原实现输出逐位一致。
- Host 五种矩阵选择 OpenBLAS，小矩阵保留原实现；输出与所选实现的已保存输出逐位一致。
- Host 的“外部 MatMul → Softmax → 未测过的 MatMul”实际运行通过，
  前者使用原始布局，后者仍使用原分块权重布局。
- QEMU 另有真实混合路径用例，验证外部按行并行、原分块和 Softmax 连接、
  共享输入、尾部尺寸、错误等待与重复调用。强制覆盖路径的合成耗时仅用于功能测试，
  不作为性能证据。

本轮累计 **187 个定向 pytest 用例通过**：通用接口与构建 118 项、额外边界 3 项、
真实 OpenBLAS 混合调用 1 项、QEMU 4 项、原分块/权重布局回归 61 项。
其中 QEMU 用例包含两次预期的错误状态验证，并非运行失败。

另对完整 Qwen3 图仅做生成与布局检查：**254 个 MatMul 全部保留原实现，
196 组权重保留分块排布**。去除性能配置新增的 hart 编译约束后，生成源码
与显式选择原实现的源码逐字相同。完整图中的未测形状、转置权重和动态右侧
均保守回退；没有把二维矩阵测量推广到这些节点。
本项没有编译或执行完整模型，因此不声称本轮已经复现 410.53 秒或完整模型精度。

### 数据保存与复查

本轮所有新数据位于 `output/issue109-performance-selection/`（本地目录；[验收摘要](external-matmul-evidence.json)）。
旧完整模型数据、旧归档和本轮开始前的工作区改动均保留。

| 位置（相对于本轮目录） | 内容 |
| --- | --- |
| `before/` | 本轮修改前的相关文件副本及工作区状态 |
| RISC-V 报告（本地：`output/issue109-performance-selection/riscv-measurements/report.json`；[验收摘要](external-matmul-evidence.json)） | 全部样本、环境、数值；对应目录保存源码、ELF、输入、权重、参考、输出与运行记录 |
| Host 报告（本地：`output/issue109-performance-selection/host-measurements/report.json`；[验收摘要](external-matmul-evidence.json)） | 全部样本与准备成本；对应目录保存源码、动态库、输入、权重、参考和检查点输出 |
| RISC-V 选择验证（本地：`output/issue109-performance-selection/riscv-selected-verification/report.json`；[验收摘要](external-matmul-evidence.json)） | 实测配置、逐节点选择与一次正确性执行 |
| Host 选择验证（本地：`output/issue109-performance-selection/host-selected-verification/report.json`；[验收摘要](external-matmul-evidence.json)） | 实测配置、逐节点选择与混合图验证 |
| 完整图生成检查（本地：`output/issue109-performance-selection/full-graph-codegen-only/report.json`；[验收摘要](external-matmul-evidence.json)） | 两份源码、254 个选择原因、196 组排布及一致性检查 |
| `generic-tests-fixed.log`、`generic-boundary-tests.log`、`openblas-measured-test.log` | 接口、构建、边界与 Host 定向测试 |
| `qemu-tests.log`、`builtin-regression.log` | QEMU 混合路径与已有优化回归 |
| `benchmark_riscv.py`、`benchmark_host.py`、`verify_*_selection.py`、`check_full_graph_codegen.py` | 复用现有接口的本地实验脚本 |

首次 Host 加载因缺少 `libgfortran` 的搜索路径而中止，其日志和未完成目录保留；
配置现有本地依赖后，在新目录完成正式测量，未混入失败尝试。
通用测试首次的五处测试假设错误也保留了原日志，修正后全部通过。

复查时从已有报告开始，不必重跑。需要以后复现 Host 时，除了前述线程变量和
头文件/库目录，还要将本地展开包的 `usr/lib/x86_64-linux-gnu` 及其中
`openblas-pthread` 加入 `LD_LIBRARY_PATH`，确保库的动态依赖可找到。
实验脚本使用新的输出目录并拒绝覆盖已有数据，不应直接重跑覆盖本次记录。

## 补测真实矩阵形状和权重布局（2026-10-24）

本轮将完整模型的 254 个 MatMul 归并为 9 类，保留实际批次和全部 16 个注意力头。
8 类普通记录对应完整图的 **253 个节点**；另补清共享词嵌入权重末层的独立成本，
Host 最终配置有 9 类、覆盖全部 254 个节点。RISC-V 的末层按接口上限回退。
本节当时只做定向测试和完整图生成；后续完整运行结果见下一节。

这里的“覆盖节点”指它们的形状、权重类别和布局已有匹配记录，
不是逐一执行了 254 个真实中间激活。固定权重取自模型第一层对应算子，
末层取实际共享权重；输入使用固定种子的代表数据，全部保存。
本轮不据此承诺所有层、所有输入或整个混合模型的数值结果。

### 实际补齐了哪些情况

| 类别 | 节点数 | 左侧完整形状 | 右侧完整形状 | 原实现的右侧布局 |
| --- | ---: | --- | --- | --- |
| 位置矩阵 | 1 | 1×64×1 | 1×1×256 | 动态输入，原始布局 |
| Q 投影 | 28 | 1×256×1024 | 1024×2048 | 转置常量，可分块 |
| K/V 投影 | 56 | 1×256×1024 | 1024×1024 | 转置常量，可分块 |
| QK 注意力 | 28 | 1×16×256×128 | 1×16×128×256 | 动态输入，原始布局 |
| AV 注意力 | 28 | 1×16×256×256 | 1×16×256×128 | 动态输入，原始布局 |
| 输出投影 | 28 | 1×256×2048 | 2048×1024 | 转置常量，可分块 |
| MLP 升维 | 56 | 1×256×1024 | 1024×3072 | 转置常量，可分块 |
| MLP 降维 | 28 | 1×256×3072 | 3072×1024 | 转置常量，可分块 |
| 末层输出 | 1 | 1×256×1024 | 1024×151936 | 转置常量、共享权重，保留原始布局 |

普通固定权重用例保留“实际原权重 N×K → 转置 → MatMul”的结构：
原实现可把转置合入静态分块排布，外部路径需要执行的转置计入运行耗时。
动态右侧保持真正的第二个运行输入，没有改成常量，也没有缩小到单个注意力头；
其计时边界为 MatMul 接收连续缓冲区处，不含上游生成这些输入的算子。

末层的 `model.lm_head.weight` 同时用于词嵌入 Gather，原实现不能像普通投影那样
折叠成专用分块权重。本轮 Host 子图保留同权重的 Gather 和转置，确认两端均为原始布局。
该 Gather 读取一行 1024 个元素，不等于完整模型读取 256 行词嵌入的工作量。
因此另做独立的“转置 → MatMul”补测，显式保持该节点在完整模型中的原始布局，
移除额外 Gather。已有输入、权重和参考直接复用。对每种实现，独立探针与原共享子图中的转置、乘法主体
在仅统一变量和函数编号后逐字一致，确认没有换成更便宜的分块布局或改变算法。
完整图偏好 `column-panel`、此探针显式 `original`，两者在末层的**实际布局相同**；
最终记录使用 `transposed_constant/original`，没有放宽其他节点的匹配规则。

### 定向实测结果

每类、每种实现预热一次，再交替顺序测三次。表内为中位数。
RISC-V 使用 4 hart，NMSIS 按输出行并行；Host 的应用和 OpenBLAS 均为单线程。
原实现启用 `blocked_fma/2x4`，只有实际允许的固定权重使用 `column-panel`。

| 类别 | RISC-V 原实现／秒 | NMSIS／秒 | Host 原实现／毫秒 | OpenBLAS／毫秒 |
| --- | ---: | ---: | ---: | ---: |
| 位置矩阵 | 0.02223 | 0.02763 | 0.01086 | 0.01120 |
| Q 投影 | 1.07626 | 12.22915 | 46.13094 | 20.36373 |
| K/V 投影 | 0.54880 | 6.11310 | 23.06451 | 8.97631 |
| QK 注意力 | 0.32419 | 0.56538 | 13.74026 | 2.96618 |
| AV 注意力 | 0.29399 | 0.49331 | 11.89525 | 2.85968 |
| 输出投影 | 0.98412 | 11.84636 | 47.06436 | 20.27518 |
| MLP 升维 | 1.50564 | 18.17630 | 81.82041 | 26.07019 |
| MLP 降维 | 1.46588 | 17.93193 | 74.39245 | 29.81584 |
| 末层独立转置＋乘法 | 未测 | 接口不支持 | 8495.40031 | 2246.11568 |

RISC-V 的 8 类都应保留原实现；末层 N=151936 超过 NMSIS 的 65535 上限，
无需运行必然不兼容的外部用例。Host 的位置矩阵保留原实现，其余 7 类常规矩阵选择 OpenBLAS。
末层独立探针的 OpenBLAS 中位耗时约为原实现的 1/3.78，
最快原实现为 8.21977 秒、最慢 OpenBLAS 为 2.31177 秒，仍通过保守门槛，
已写入最终 Host 配置。此前共享子图的 8.55865／2.26916 秒数据仍保留，
但未把包含额外 Gather 的成绩用于最终选择。

RISC-V 计时从启动 QEMU 前到完成帧验证结束，含启动和模拟执行开销。
Host 计时围绕一次已加载的 C 函数调用，含生成的转置、数值检查等工作。
两者都把参考计算、生成、静态权重排布、编译、缓冲准备和输出保存另列记录。
本次没有计时接管；两平台时间不能直接横比，也不能把各节点时间相加当作完整模型耗时。
位置矩阵很短，尤其容易受 QEMU 启动和调用开销影响。

### 怎样避免把波动误判为收益

本轮继续使用原有 `measured` 接口，没有放宽形状或布局匹配，也没有改写库内核。
新配置选择更保守的实测值：

- `builtin_seconds` 取三次正式测量中的最小值。
- `external_seconds` 取三次正式测量中的最大值。
- 仍要求外部耗时至少减少 10%。也就是最慢外部样本仍比最快原实现样本至少快 10%，才采用外部库。

原始样本、最小值、最大值和中位数全部保留；没有为了改善结果追加测量或丢掉慢样本。
这能避免本组重叠样本产生误选，但三次本机测量不等于统计意义上的稳定性保证。
配置仍依赖明确的目标、hart 数、库版本、线程和编译条件；更换环境不能直接套用。

沿用上一节读取 `MatMulPerformance` 的代码，将文件换成：

- RISC-V 配置（本地：`output/issue109-real-matmul-shapes/riscv/profile.json`；[验收摘要](external-matmul-evidence.json)）：4 hart、8 条真实形状记录。
- Host 最终配置（本地：`output/issue109-real-matmul-shapes/host/profile-complete.json`；[验收摘要](external-matmul-evidence.json)）：单线程、9 条记录，末层单独标明测量来源。

旧 Host 的 `profile.json` 保留当时的 8 条结果及末层排除原因；最终配置使用 `profile-complete.json`。

它们是本轮真实模型形状的独立配置，未覆盖或改写旧实验配置。
默认模式仍不改变；使用时须显式设置 `matmul_implementation="measured"` 及匹配的生成参数。

### 数值与选择检查

RISC-V **64 次**运行全部通过；Host 共 **80 次**调用全部通过，
包括初次 9 类的 72 次，以及为拆清末层成本新增的 8 次独立探针。
两平台最大 ORT 绝对误差均为 `7.748603820800781e-7`，
全部输出有限，沿用严格 `max_abs < 0.001`、`rtol=0`。
独立 FP64 对照也全部通过：最大误差预算占比约为 RISC-V 0.731%、Host 0.803%。
RISC-V 实验流程用时 **303.31 秒**；Host 初次 9 类用时 **58.90 秒**，
末层独立补测另用 **47.18 秒**。这些时间包括各自准备与验证，不是模型的前向耗时。

重新解析完整 ONNX，仅做源码生成和权重布局核对，结果如下：

| 目标 | 按测量选择的外部节点 | 保留原实现 | 原分块权重数 | 尚无可直接使用的测量 |
| --- | ---: | ---: | ---: | --- |
| RISC-V | 0 | 254 | 196 | 无意外缺口；末层按维度上限回退 |
| Host | 253 | 1 | 0 | 无，末层独立成本已补齐 |

Host 仅保留位置矩阵的原实现。固定权重改用 OpenBLAS 后保留实际运行时转置，
所以 Host 的分块权重数为 0；这是这些节点的实测选择，不是全局关闭原优化。
RISC-V 则保持全部 196 组原分块布局。两目标的形状、节点数量和实际权重映射均重新核对，
Host 的 9 类配置精确匹配 254 个节点；RISC-V 的 8 类匹配 253 个节点，末层按接口上限回退。
两目标都没有意外的未知项。

另对两目标共 16 个选中子图进行源码和物理权重对照：与已经实测的对应实现一致，
源码仅增加 hart 数量约束；复用它们的数值证据，没有再次执行。
末层另核对独立探针与原共享子图的计算主体，以及完整图中保留的原始权重和转置。
新增两个定向回归通过，确认三维输入配转置常量、四维动态批次可以精确匹配，
且降维的旧记录不会被误用。本轮没有更改生产代码或重复上轮全套测试。

### 保存位置与剩余验收

全部新数据在 `output/issue109-real-matmul-shapes/`（本地目录；[验收摘要](external-matmul-evidence.json)）。
旧改动、历史测试和归档保持原样，本轮修改前的相关文件副本位于 `before/`。

| 文件或目录 | 内容 |
| --- | --- |
| inventory.json（本地：`output/issue109-real-matmul-shapes/inventory.json`；[验收摘要](external-matmul-evidence.json)） | 9 类完整形状、254 个节点、实际权重名称和布局 |
| `cases/` | 固定输入、实际权重、ORT 与 FP64 参考，末层外置权重文件 |
| RISC-V 报告（本地：`output/issue109-real-matmul-shapes/riscv/report.json`；[验收摘要](external-matmul-evidence.json)） | 64 次输出、计时、数值、ELF、源码、权重包和日志的记录 |
| Host 初次报告（本地：`output/issue109-real-matmul-shapes/host/report.json`；[验收摘要](external-matmul-evidence.json)） | 初次 72 次调用、输出、动态库、源码、分阶段时间 |
| 末层独立报告（本地：`output/issue109-real-matmul-shapes/host-lm-head-isolated/report.json`；[验收摘要](external-matmul-evidence.json)） | 另 8 次输出与独立成本，原始／统一编号的内核文本及差异检查 |
| 完整图最终选择检查（本地：`output/issue109-real-matmul-shapes/full-graph-selection-complete/report.json`；[验收摘要](external-matmul-evidence.json)） | 9 条 Host 配置和 8 条 RISC-V 配置的最终覆盖；旧检查结果保留 |
| `full-graph-selection-complete/host-model.c`、`riscv64-model.c` | 最终完整图生成结果，未编译、未执行 |
| `benchmark-riscv.log`、`benchmark-host.log`、`exact-rank-tests.log` | 本次实验及两个新增回归的日志 |

本轮已补齐这些实际形状和布局的选择依据，未通过缩小形状或预先转置来省去运行成本。
本节的定向证据不代替完整模型验收；后续单用例完整运行、精度和总耗时见下一节。
多组输入和真实 RISC-V 硬件仍需要各自证据。

## 按实测配置运行完整模型（2026-10-24）

本轮按用户指定的两种新配置，各执行一次完整模型，均通过。
用例为 `full_seed_0`，序列长度 256，模型包含全部 28 层；
每端比较完整的 **38,895,616 个 FP32 输出**，不是只检查最后一个词。
只在本地运行，没有提交、推送或创建 PR，也没有覆盖旧结果。

### 本次实际执行的配置

| 目标 | MatMul 选择 | 原优化与库设置 |
| --- | --- | --- |
| Host | 253 个 OpenBLAS + 1 个原实现 | 应用和 OpenBLAS 均为单线程；仅位置矩阵保留原实现 |
| RISC-V/QEMU | 254 个原实现，0 个外部调用 | 4 hart、`blocked_fma/2x4`、196 组 `column-panel` 权重、已有 Softmax 并行 |

两端都显式使用 `measured` 和上一轮冻结的配置。RISC-V 末层因共享词嵌入权重继续
保留原始布局，其他可用分块权重完整保留，工作区布局为 `none`。
NMSIS 的接口和测量配置仍保留，但本次 RISC-V 可执行文件未链接 NMSIS。
因此这轮验证的是“按实测选回原优化”的完整链路，不是 NMSIS 按行并行的完整模型。

生成的 C 与上一轮最终完整图选择检查逐字一致。编译器为 Zig 0.14.1；
QEMU 为 8.2.2，目标 RV64GC/LP64D，实际参数包含 `-smp 4` 和 `tcg,thread=multi`。
Host 除设置线程环境变量外，还读取 OpenBLAS 的实际线程数，确认是 1。

### 精度结果

| 目标 | 对 ORT 最大绝对误差 | 均方根误差 | 达到或超过 0.001 的元素 |
| --- | ---: | ---: | ---: |
| Host | 0.00008678436279296875 | 0.000006101037650925549 | 0 |
| RISC-V/QEMU | 0.0000858306884765625 | 0.000005391918773554923 | 0 |

两端全部输出有限，均满足原有严格门槛 **`max_abs < 0.001`、`rtol=0`**。
RISC-V 与历史 4 hart 原优化结果逐位一致，最大差异为 0；
与此前“主 hart 串行 NMSIS + 末层原分块”的 4 hart 结果不逐位相同，
最大差异为 0.00008487701416015625。验收仍以 ORT 为准。

本轮复用保存的 ORT 结果，没有额外运行一次完整 ORT 模型。
复用前已核对输入、完整图结构和当前权重：2148 个权重条目及填充共
2,384,319,432 字节与旧参考权重直接比较一致，比较时同时复制为本次 Host 权重包。
RISC-V 按自己的分块布局另写一份物理权重包。沿用已有检查，没有增加哈希体系。

### 分阶段耗时

单位为秒。两个目标顺序执行，未并行跑其他基准。

| 阶段 | Host | RISC-V/QEMU |
| --- | ---: | ---: |
| 解析模型 | 0.430 | 0.337 |
| 生成代码，含布局登记与检查 | 0.408 | 2.449 |
| 权重准备与保存 | 2.736 | 4.091 |
| 完整构建阶段 | 128.905 | 100.008 |
| 前向运行 | 7.611 | 435.050 |
| 各自验证流程总耗时 | 140.603 | 560.685 |

RISC-V 构建阶段内，“启动编译器至 ELF 验证完成”为 **97.324 秒**；
100.008 秒还包含权重包检查、源文件和链接脚本准备等。
Host 的 128.905 秒是已有动态库构建接口的完整计时，未额外拆出纯编译器时间。
生成代码中的 RISC-V 子计时为布局规划 0.014 秒、分块视图登记和有限值检查 2.182 秒，
二者已包含在 2.449 秒内，不重复相加。实际分块权重序列化归入权重准备阶段。

Host 的 7.611246 秒只围绕一次同步 C 函数调用；加上动态库加载、绑定和关闭后为
7.611619 秒。RISC-V 的 435.049753 秒从启动 QEMU 前计到完成帧校验结束，
包含模拟器启动、即时翻译和模型内检查，不能与 Host 的纯函数时间直接横比。
RISC-V 整个运行接口阶段为 452.659 秒，还包括权重运输准备、输出转存和结束校验。

从启动 Host 子进程到两端收集完成，整个顺序任务用时 **702.804 秒，约 11 分 43 秒**。
各端总耗时包含各自准备、构建和数值验证；总任务还含子进程启动与退出。
这些时间均不含事先复制公共参考/配置、文档更新和最终归档。
本次全程保留原始单调时钟计时，**没有收集器接管或起点重建**；
两端子进程和 QEMU 均真实返回 0，QEMU 完成帧与完整输出已保存。

### 与旧结果的比较

| 比较对象 | 旧前向／秒 | 本次前向／秒 | 单次历史变化 |
| --- | ---: | ---: | --- |
| Host，旧 254 个 OpenBLAS | 7.725460 | 7.611619 | 近似同口径少 1.47% |
| RISC-V，旧 4 hart 原优化 | 410.530255 | 435.049753 | 多 5.97%，输出逐位一致 |
| RISC-V，旧 4 hart 串行 NMSIS 混合配置 | 9249.003713 | 435.049753 | 少 95.30%，旧耗时约为本次 21.26 倍 |

Host 比较使用含加载/关闭的时间，接近旧 `invoke_library` 口径，仍有少量包装步骤差异。
本次 1.47% 差异不能证明稳定收益。RISC-V 两个主要旧基准都保留原始计时、没有接管；
旧原优化配置连内存布局也与本次一致，但本次确实测得慢约 6%，不据此声称刷新性能纪录。
旧 NMSIS 配置则是 253 个主 hart 串行调用和末层 `1x4/original`，
本次变为全部原实现及 `2x4/column-panel`，不能把 21.26 倍解释为单独某一项优化的收益。

旧 Host 构建约 129.145 秒、流程共 144.263 秒；旧原优化 4 hart 编译约 83.122 秒、
流程共 527.442 秒；旧 NMSIS 4 hart 流程共 9382.714 秒。
旧流程包含的 ORT 计算、检查与本次权重复制、历史输出比较不同，总耗时不作同口径加速比。
所有比较都是单次历史记录，机器负载、缓存和二进制布局未做受控重复实验。
旧单 hart NMSIS 的 9364.473 秒是接管后的估算，仅保留为历史证据，未用于上表。

结果说明，在当前输入和环境下，按实测选回原优化后，RISC-V 完整前向已回到分钟级；
Host 继续使用适合开发电脑的 OpenBLAS。它不意味着外部库本身已在当前 QEMU 目标上加速。

### 数据保存与复查

全部本轮数据保存在 `output/issue109-full-measured/`（本地目录；[验收摘要](external-matmul-evidence.json)）。
脚本拒绝覆盖已有运行目录；如需以后复现，应使用新目录，避免重复启动本轮任务。

| 文件或目录 | 内容 |
| --- | --- |
| 总结果（本地：`output/issue109-full-measured/result.json`；[验收摘要](external-matmul-evidence.json)） | 两端状态、子进程退出码和顺序任务总耗时 |
| Host 结果（本地：`output/issue109-full-measured/host/result.json`；[验收摘要](external-matmul-evidence.json)） | 精度、253+1 选择、实际线程数、生成/编译/运行计时 |
| RISC-V 结果（本地：`output/issue109-full-measured/riscv/result.json`；[验收摘要](external-matmul-evidence.json)） | 精度、254 个原实现、196 组分块、旧输出比较与分阶段耗时 |
| QEMU 原始记录（本地：`output/issue109-full-measured/riscv/qemu/run.json`；[验收摘要](external-matmul-evidence.json)） | 启动参数、完成帧检查、原始前向时间、真实退出码 |
| 历史基准核对（本地：`output/issue109-full-measured/baseline-audit.json`；[验收摘要](external-matmul-evidence.json)） | 5 项旧记录的位置、输入与参考一致性、计时差异 |
| `reference/`、`profiles/` | 完整输入、ORT 基准和本次冻结的实测选择配置 |
| `host/`、`riscv/` | 完整输出、生成 C、动态库/ELF、物理权重、编译与运行记录 |
| `host_full.py`、`riscv_full.py`、`run_all.py`、`launch.sh` | 本次实际执行的脚本与启动环境 |
| `git-status-before.txt`、`git-diff-before.patch` | 本轮开始前的本地状态 |

完整归档还包含模型原文件、当前源码及未提交改动、库依赖、实测选择的原始数据和旧比较输出。
采用普通 tar、直接回读逐文件比较和落盘同步；不覆盖旧归档。
归档路径、字节数及成功状态记录在输出目录下的
`issue109-full-measured-archive-receipt.json`，归档日志另存，均位于本轮运行目录之外。

本轮只验证一个完整输入用例。其他输入、七用例覆盖、生成循环和真实 RISC-V 硬件，
以及同条件重复性能测量，仍需各自证据。后续只做文档链接检查和 `git diff --check`，
不为补计时重复运行完整模型。

## RISC-V 约 6% 耗时差异的定向排查（2026-10-24）

**目前没有发现计算代码回退，但也不能把这次差值全部认定为运行波动。**
旧原优化 4 hart 前向为 410.530255 秒，新配置为 435.049753 秒，
相差 24.519498 秒，即 5.9726%。两者都是原始计时、事件完成、无接管；
这是真实记录的墙钟差异，不能改写或忽略。

本轮只进行源码/制品对照、三类小图和完成通知测试，没有重跑完整模型。
没有修改生产代码，也没有调整原优化、选择规则或撤回 `poll` 修复。

### 已经排除的计算路径差异

直接比较旧、新 ELF 的全部三个加载段，指令、常量、初始化数据、地址及大小完全相同。
全部参与执行的节与函数地址也一致。生成的 `model.c` 只多了 150 字节的
编译期 hart 数检查；ELF 文件整体的差异位于不参与执行的调试/符号信息。

两次构建命令、工具版本、4 hart 参数、内存布局、196 组分块权重和
`blocked_fma/2x4/column-panel` 一致。已有报告记录的输入、权重身份值也一致，
前一轮已经直接确认完整输出逐位一致。本轮只读取既有身份记录，没有重做哈希。

因此，`measured` 在这次完整模型中没有引入运行时选库工作，也没有丢掉原分块优化。
仍需区分的是宿主运行条件和完成观察方式，而不是假定计算内核变慢了。

### 相同程序下，新旧等待方式的对照

宿主侧确有一项差异：UART 完成等待由 `select.select` 改成了 `select.poll`，
用于避免文件描述符数量上限。前向计时函数、完成帧解析、QMP 和内存采样代码保持相同。

复用已有输入、权重与 ORT/FP64 参考，每类各构建直接 `builtin` 和 `measured` 回退版本，
确认两版加载段完全一致。正式对照统一运行同一个 `measured` ELF，
只替换宿主等待器，不把两个因素混在一起。

每种等待方式预热一次，然后交替先后顺序运行五对，三类共 **36 次**。
所有运行均为 4 hart、事件等待且没有回退，旧 `select` 实际使用的描述符均小于 1024。

| 样本 | 旧 select 中位／秒 | 新 poll 中位／秒 | 两组中位数变化 | 五对比值的中位变化 | poll 较慢的配对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 真实 MLP 升维：1×256×1024 乘 1024×3072 | 1.530502 | 1.546712 | +1.06% | +0.26% | 3/5 |
| 动态注意力：1×16×256×128 乘 1×16×128×256 | 0.311212 | 0.323212 | +3.86% | +0.004% | 3/5 |
| 原分块 MatMul → Softmax → 原分块 MatMul 混合图 | 0.307985 | 0.333961 | +8.43% | +2.81% | 4/5 |

“配对比值”先比较相邻一对的 poll/select，再取五对的中位数，
与分别取两组中位数后相除不是同一种统计。两者都保留，不能只挑有利结果。
尤其混合图的分组中位数确实偏慢 8.43%，五对中四对偏慢；
这组样本不足以宣称等待器完全没有影响。

另一方面，不改变程序和等待器的重复运行本身也有波动：
旧 select 的最大/最小耗时差分别为 11.23%、16.90%、18.31%，
新 poll 分别为 3.58%、14.15%、28.28%。这些是短图样本范围，
不能当成完整模型的误差范围或据此断言完整模型的 5.97% 全是波动。

36 次输出全部有限且同图逐位一致；最大 ORT 绝对误差为
0.00000014901161193847656，严格小于 0.001，FP64 对照也全部通过。
整个定向流程用时 31.585 秒。每次保存原始输出、计时、布局、构建记录和环境快照；
本轮系统级换入/换出计数均未增加。这只说明这些定向样本的条件，不能反推旧完整模型。

### 完成通知本身是否增加了长时间等待

由于混合图仍有偏慢样本，再复用已有 UART 测试的管道同步方式：
真实 Python 子进程写入合法完成帧，写完后等待父进程确认，避免进程退出先唤醒等待器。
不启动 QEMU，不进行矩阵计算。旧、新方式各预热一次、正式测试 20 次，交替共 **42 次**。

| 从写入前到等待返回，正式样本 | 旧 select | 新 poll |
| --- | ---: | ---: |
| 最小值 | 51.718 微秒 | 78.469 微秒 |
| 中位数 | 93.447 微秒 | 93.623 微秒 |
| 最大值 | 114.467 微秒 | 261.597 微秒 |

两组中位数仅差 0.176 微秒；20 对中 poll 较慢 8 对。
42 次全部通过，均为事件等待、无回退，子进程最终正常退出；流程共 1.438 秒。
还保存了写入前、写入后、等待返回三个原始时间戳。
写入后时间有一次晚于父进程返回，产生负差值；它是记录顺序受调度影响，
不是物理负延迟，原值完整保留，没有截零。

本场景没有复现足以解释完整模型额外 24.52 秒的通知尾部延迟。
但它不模拟持续多 hart 运算、逐字节 UART 流或宿主竞争，
不能单独排除等待方式对 QEMU 调度的间接影响。
完整前向只等待一次顶层完成帧，不能把通知延迟按 254 个 MatMul 倍乘。

### 仍不能从旧记录确定的原因

本次完整验证服务的日志记录了 `9.0G memory peak` 和
`556.3M memory swap peak`，覆盖 Host、编译及 RISC-V 全部阶段。
它说明服务期间存在交换内存驻留，不能确定发生在 QEMU 前向，也不能解释其中多少耗时。
新 QEMU 的进程内存峰值反而比旧记录低约 413 MB。

旧运行保存了 QEMU CPU 时间和线程采样，新完整运行缺少同口径的独立 CPU 时间、
历史 CPU 亲和与压力序列。现有数据不能拆出“CPU 得到的运行机会变少”
“频率或缓存变化”“换页”各占多少。计时边界与时钟核对未见足以解释差值的异常。

当前决定是保留计算代码和 `poll` 修复，不为一次历史差值盲目调整内核或选择规则。
**已确认计算程序等价；尚未确认完整运行存在稳定性能回退，也尚未定位全部 24.52 秒。**
以后若另获授权运行完整模型，应在同一环境、相邻交替运行中同时记录
QEMU CPU 时间、前向阶段换页/负载和墙钟时间，再判断可重复差异。
本轮不为补这些证据重跑完整模型。

### 保存位置

全部新数据位于 `output/issue109-riscv-regression-analysis/`（本地目录；[验收摘要](external-matmul-evidence.json)）；
此前完整模型数据和归档保持原样。

- 源码与可执行程序对照（本地：`output/issue109-riscv-regression-analysis/source-audit.json`；[验收摘要](external-matmul-evidence.json)）
- 宿主运行与资源审计（本地：`output/issue109-riscv-regression-analysis/runtime-audit.json`；[验收摘要](external-matmul-evidence.json)）
- 36 次小图原始结果（本地：`output/issue109-riscv-regression-analysis/measurements/report.json`；[验收摘要](external-matmul-evidence.json)）：各次输入、输出、参考、权重、源码、ELF 和运行日志同目录保存。
- 42 次通知探针结果（本地：`output/issue109-riscv-regression-analysis/notification-latency/report.json`；[验收摘要](external-matmul-evidence.json)）：原始完成帧和时间戳全部保留。
- `benchmark.py`、`notification_latency.py` 及各自日志：本轮可复查脚本；拒绝覆盖已有测试目录。
- `git-status-before.txt`、`git-diff-before.patch`：本轮开始前的本地状态。

这些报告的 PASS 指执行、数值、等价条件或事件协议检查通过，
**不表示“性能回退已被证明不存在”**。

## 完整模型 select / poll 重复对照（2026-10-24）

**本轮没有复现 poll 稳定慢约 6% 的现象；保留现有 poll 修复和原计算优化。**
三组相邻对照的 poll 耗时变化介于 −1.09% 与 +1.04%，方向不一致。
旧 select 自身也出现较慢运行，最后一组两种方式同时变慢。
这支持运行条件会影响完整模型耗时，但不能反推历史 24.52 秒全部由某一种宿主因素造成，
也不能用三组数据承诺两种等待方式在所有环境中完全等速。

### 对照条件与精度

本轮按用户授权执行完整 28 层的 `full_seed_0`，有效长度 256。
两种等待方式各预热一次，再做三组正式对照，共 **8 次完整前向**。
正式顺序为 select→poll、poll→select、select→poll；预热保留但不计入性能统计，
没有失败重试、剔除样本或追加运行。

八次使用同一份冻结 ELF、输入、ORT 参考和物理权重：4 hart、
254 个原实现 MatMul、196 组分块权重、原 `blocked_fma/2x4/column-panel`。
唯一有意改变的执行选择是宿主 UART 等待器；没有重新生成代码、编译或运行 ORT。
QEMU 仍为 `tcg,thread=multi`，可用 CPU 范围统一为 0–3；该设置不等于独占 CPU。
旧 select 的实际描述符均不超过 10，未触及其 1024 限制。

8 次全部通过：每次 **38,895,616 个 FP32 输出全部有限**，
对 ORT 最大绝对误差均为 **0.0000858306884765625**，
RMSE 为 0.000005391918773554923，严格满足 `max_abs < 0.001`、`rtol=0`。
与保留的旧原优化输出逐位一致。每次均有有效的 20 字节完成帧、
155,582,464 字节实际输出、真实退出码 0，事件等待没有回退。

### 正式前向耗时

| 配对 | 执行顺序 | 旧 select／秒 | 当前 poll／秒 | poll 相对变化 |
| --- | --- | ---: | ---: | ---: |
| 1 | select → poll | 429.183341 | 430.194628 | +0.2356% |
| 2 | poll → select | 433.656858 | 428.932779 | −1.0894% |
| 3 | select → poll | 447.721421 | 452.388985 | +1.0425% |
| 分别取组中位数 | 每种 3 次 | 433.656858 | 430.194628 | −0.7984% |

三组配对比值的中位变化为 **+0.2356%**，其中 2/3 组 poll 较慢；
它与表中“分别取组中位数再相除”的 −0.7984% 含义不同，两项都保留。
同一等待方式的最大/最小差为 select **4.32%**、poll **5.47%**。
预热为 select 432.066349 秒、poll 431.233291 秒，不参与上述统计。

全部时间来自原 `run_external.forward_wall_s`，**无接管、无起点重建**。
该区间从创建 QEMU 前开始，到 UART 完成校验返回为止，含 QEMU 启动、
即时翻译和 guest 内部检查；不等于只计矩阵内核。
权重运输准备、输出转存及数值比较另在执行流程中计时。
八次累计前向 **3485.377652 秒**；本次执行验证调用总计 **3623.908516 秒**
（约 1 小时 23.91 秒），包含各次准备和检查，不含之后的分析、文档和归档。
提前执行的冻结数据准备另为 2.342705 秒；两次调用之间的等待不混入连续计时。

### 前向阶段的 CPU、换页与负载

两种方式都每 0.5 秒采样，复用一个原有内存监控线程。
每次记录 QEMU 进程/线程 CPU、缺页、内存、CPU 亲和，以及系统 CPU、换页、
负载和等待资源的压力指标。八次共保存 **6949 条原始采样**，分析无缺项或采样错误。
日志写在被等待器监视的 QEMU 目录之外，避免日志本身制造额外通知。

下表 CPU 为完成边界前最后一次、边界后第一次的累计观察值，
八次身份和计数边界检查均有效；这是有采样粒度的观察区间，不是精确积分。
系统换入/换出以页数计，范围覆盖整个 WSL。

| 正式运行 | QEMU CPU／秒，边界前—后 | 交换区驻留峰值／MiB | 系统换入／换出页 | 1 分钟负载范围 |
| --- | ---: | ---: | ---: | ---: |
| 第 1 组 select | 1501.62—1502.03 | 0 | 24／801 | 3.96—4.98 |
| 第 1 组 poll | 1499.20—1499.65 | 0 | 27／408 | 3.77—5.03 |
| 第 2 组 poll | 1502.57—1502.67 | 270.07 | 2304／75964 | 3.96—5.05 |
| 第 2 组 select | 1509.77—1510.16 | 0 | 204／25 | 3.97—5.00 |
| 第 3 组 select | 1548.47—1548.68 | 282.47 | 2201／76402 | 4.48—5.23 |
| 第 3 组 poll | 1559.03—1559.45 | 315.26 | 6570／94823 | 4.10—5.19 |

第三组两者不仅墙钟变慢，进程 CPU 累计也增加，平均使用的逻辑 CPU
约降到 3.46 和 3.45；前两组约 3.48–3.50。系统 CPU 忙碌比例约 92.37%–93.29%，
各运行窗口都有 CPU 资源等待压力。不能据此把全部等待时间归到 QEMU，
或区分 Windows 宿主调度、频率、缓存等未直接测量的原因。

换页是真实观测，但**不是单独解释变慢的充分证据**：
第二组 poll 同样出现约 270 MiB 交换区驻留，却是本轮最快的正式样本。
`VmSwap` 是驻留量，主要缺页次数不等于交换换入次数；
系统换页、负载和压力包含其他 WSL 活动，且各次观测窗口长短不同。
因此不能简单宣布“慢了多少秒就是换页造成的”。

正式样本最后一次完整读取距完成边界为 0.094–0.451 秒，
首个边界后样本距边界 0.00137–0.02632 秒；后者可能已包含少量输出转存。
CPU 计数以 0.01 秒为单位，边界前后的原始值和间隔全部保留。
采样线程每次消耗约 0.921–1.174 CPU 秒，为前向墙钟的 0.214%–0.259%，
完成返回钩子为 0.028–0.852 毫秒；这不是对墙钟干扰的精确上界，也未从耗时中扣除。
两种方式使用相同采样仍不能视作零开销，历史未加此采样的运行不完全同口径。
线程排队统计的可用性未确认，不能把缺失或零值解释成没有排队。

### 本轮结论与保存位置

本轮没有证据要求撤回 poll，也没有发现计算优化或选库规则的性能回退。
原历史 410.530255 秒与 435.049753 秒仍按原样保留：
现在旧 select 的三次也达到 429.18–447.72 秒，
所以不能把历史约 6% 的差值直接认定为 poll 引入的稳定开销。
具体宿主原因尚未唯一定位，三组也不是统计等价证明；不据此继续扩大模型运行。

- `output/issue109-full-waiter-ab/`（本地目录；[验收摘要](external-matmul-evidence.json)）：固定条件、脚本、服务起止记录及本轮开始前的本地状态。
- 八次执行与精度结果（本地：`output/issue109-full-waiter-ab/run/result.json`；[验收摘要](external-matmul-evidence.json)）。
- 耗时与资源对照（本地：`output/issue109-full-waiter-ab/comparison.json`；[验收摘要](external-matmul-evidence.json)）：全部配对、预热、CPU 边界、每线程观察、负载与压力。
- `run/trials/`：八次完整输出、输入、UART 帧、运行日志、逐次结果及原始资源采样。
- `run/frozen/`、`run/reference/`、`run/waiters/`：冻结的源码/ELF/物理权重、输入/参考/历史输出和精确等待器源码。
- 归档成功后，文件位置和回读验证结果写入输出目录下的
  `issue109-full-waiter-ab-archive-receipt.json`；归档日志放在实验目录之外。

本轮归档包含新实验全部数据、当前源码及未提交改动，以及历史比较所需的制品和报告。
原 ONNX 与更广的依赖材料保留在此前完整归档中，由既有成功回执引用，不重复嵌套旧大归档。
采用普通 tar、直接回读比较和落盘同步，不增加哈希框架，不覆盖旧数据。
本轮未修改生产代码；收尾仅做文档本地链接检查和 `git diff --check`，不增加模型运行。

## 三类矩阵的 NMSIS 耗时拆分（同步主干后）

本轮在 `735dd00` 加现有本地实现上，针对 QK、AV 和第一层 Q 投影继续检查。
运行目标仍为 RV64GC / LP64D、4 hart、QEMU 多线程；NMSIS 按输出行并行调用。
这是三类子图的定向诊断，**没有重新运行完整模型**，也没有改变当前 poll 或 `measured` 选择配置。

复用先前保存的输入、实际第一层权重、ORT 与 FP64 参考。QK 和 AV 保留完整批次维度，
右侧是运行时输入；它们的测量边界不包含上游生成或转换输入布局的操作。
Q 投影保留真实原始权重及运行时转置。输入激活仍是固定种子的代表值，不是真实模型中间值的捕获。

### 先区分时间花在哪里

仅在新生成的 C 和适配器副本中添加计时，官方 NMSIS 计算源码与生产运行器保持原样。
每批记录整体调度起止、4 个 hart 的适配器调用区间及真正 NMSIS 函数区间；
Q 投影另将转置数据搬移和转置结果检查分开。
结束后通过已有 QMP 通道额外转存计数，计算中途不写 UART，原完成帧和 `forward_wall_s` 继续保留。

每例 1 次预热、3 次正式运行，共 **12 次通过**；总准备与验证流程为 **54.056704 秒**。
以下各列分别取三次正式中位数，单位秒：

| 时间范围 | QK | AV | Q 投影 |
| --- | ---: | ---: | ---: |
| 原前向墙钟 `forward_wall_s` | 0.495973 | 0.467668 | 12.181768 |
| guest 函数入口至结束 | 0.474160 | 0.439363 | 12.153997 |
| NMSIS 计算区间并集 | 0.434255 | 0.414069 | 11.976454 |
| 每批调度区间累计 | 0.436235 | 0.416019 | 11.976916 |
| 调度区间内没有适配器调用活跃的时间 | 0.001749 | 0.001731 | 0.000452 |
| 转置数据搬移 | 0 | 0 | 0.148624 |
| 转置结果检查 | 0 | 0 | 0.006052 |

各 hart 的计算同时发生，表中的“并集”只对重叠时间算一次，再累计顺序执行的各批。
逐次计算 NMSIS 并集占 guest 时间的比例后取中位数，QK、AV、Q 投影分别为
**91.86%、94.24%、98.54%**。在这三个测量边界内，主要时间花在 NMSIS 计算，
优先检查库计算路径比优先修改调度更有依据。QK/AV 的转换为 0 只说明本子图没有转换步骤。

12 次输出全部有限，严格满足 ORT `max_abs < 0.001`、`rtol=0`，
并通过既有 FP64 误差预算。最大 ORT 误差分别为
QK **2.980232238769531e-7**、AV **7.748603820800781e-7**、
Q 投影 **3.725290298461914e-7**；每次与旧 NMSIS 实际输出逐位一致。
完成帧、真实退出码 0、4 hart 配置以及事件等待无回退均已核对。

本轮从同一 QEMU 导出的设备树确认计时器频率为 **10 MHz**。
`rdtime` 读取的是 QEMU 虚拟时间，**不是 CPU 周期**；函数区间可能包含即时翻译和宿主调度等待。
“没有适配器调用活跃的时间”包含唤醒、同步、计时记录及调度影响，不能称为纯调度 CPU 成本。
各列是独立中位数，不能强行相加或相减；也不能把四个 hart 的耗时相加当作墙钟时间。
计时代码本身有开销，本轮只用于诊断，不能直接拿来判定外部实现是否达到 10% 选用门槛。

脚本、原生成代码、计时版本、适配器副本、ELF、设备树、计数表、输入、输出及原运行日志
保存在本地 `output/issue109-nmsis-profile/baseline/`；入口脚本为
`output/issue109-nmsis-profile/profile_riscv.py`，日志为
`output/issue109-nmsis-profile/baseline.log`。可分享数值见
[验收摘要](external-matmul-evidence.json)，完整原始材料仍是本地数据。

### 去掉计时代码后的三方对照

根据拆分结果，候选只启用 NMSIS 官方已有的 `RISCV_MATH_LOOPUNROLL` 编译开关。
它将内层循环展开，减少循环控制；没有另写矩阵算法，也没有改动 NMSIS 源码。
对照的三方是现有原优化实现、现有 NMSIS、开启此开关的 NMSIS，均为 4 hart。
两份 NMSIS 的生成代码、运行时、内存布局完全相同，构建命令唯一差异是该宏。
ELF 反汇编确认选中了不同路径：矩阵函数从 148 字节变为 278 字节，
静态乘法、加法指令各从 1 处变为 5 处，均没有融合乘加指令；这不是动态执行次数统计。

本轮全部使用**未插桩**代码和原 `forward_wall_s`，每例每种实现各 1 次预热、3 次正式运行，
三种实现按轮次轮换顺序，同一时刻只运行一个 QEMU。共 **36 次通过**，
总流程 **112.277007 秒**。正式前向中位数如下：

| 矩阵 | 原优化实现／秒 | 原 NMSIS／秒 | 循环展开 NMSIS／秒 | 展开后相对原 NMSIS |
| --- | ---: | ---: | ---: | ---: |
| QK | 0.331448 | 0.529714 | 0.452790 | −14.52% |
| AV | 0.297934 | 0.447408 | 0.397326 | −11.19% |
| Q 投影 | 1.028714 | 11.869110 | 11.777665 | −0.77% |

QK 和 AV 在这三次正式样本中缩小了与原优化实现的差距，但候选仍分别约为原实现的
1.37 倍、1.33 倍耗时。Q 投影候选约为原实现的 11.45 倍耗时；
其原 NMSIS 范围是 11.716802–11.973469 秒，候选是 11.520893–12.006542 秒，
范围重叠，不能把约 0.77% 的中位差宣称为稳定收益。

沿用保守门槛：**候选最慢正式时间 ≤ 原实现最快正式时间 × 0.9**。
三个候选均未达到门槛，因而本轮**不修改生产选用策略、默认 NMSIS 构建、既有实测配置或 poll**。
Q 投影中原实现沿用 `blocked_fma / 2x4` 和静态列分块权重，NMSIS 保留所需的运行时转置；
QK/AV 是动态 RHS，两边均使用原布局。对比的是各实现完成同一子图所需的时间。

36 次输出全部有限，严格通过同一 ORT 门槛和 FP64 预算；
最大 ORT 误差为 **7.748603820800781e-7**，最大 FP64 预算占比约 **0.007309**。
循环展开与原 NMSIS 的输出逐位一致，真实退出码均为 0，事件等待均未回退。
计时包含 QEMU 启动、即时翻译、guest 检查、必要转置和矩阵计算；
准备、静态权重排布、编译、运输准备、输出转存和数值比较不在前向时间中。
这仍是本机 QEMU 的三次正式样本，不能外推为完整模型或真实硬件提速。

如需单独复现候选构建，在已有 NMSIS 制品和权重包的构建调用中显式加入开关即可：

```python
from scratchv.runtime.riscv_external import build_external

executable = build_external(
    artifact, bundle_dir, build_dir, toolchain,
    harts=4,
    extra_cflags=("-DRISCV_MATH_LOOPUNROLL",),
)
```

这里的 `artifact` 沿用显式 NMSIS、`parallel_rows=True`、`allow_provider_fp32` 配置；
它是本轮实验候选，不是现在的默认生产配置。
现有 provider 描述不会自动把额外编译开关纳入实测配置匹配，不能给旧测量补上该宏后仍当作同一配置。
本轮将候选证据单独保存，没有覆盖旧配置，也没有生成新的选用配置。

未插桩对照及其脚本保存在本地 `output/issue109-nmsis-profile/variant-comparison/`
和 `output/issue109-nmsis-profile/compare_variants.py`；日志为
`output/issue109-nmsis-profile/variant-comparison.log`。
[验收摘要](external-matmul-evidence.json) 保留各组正式最小值、中位数、最大值、误差和准入结果；
完整源码、构建命令、反汇编、输入、输出与运行记录仍保留在本地目录。

### 补查循环尾部和抵消样本

另做 10 次只检查正确性的 QEMU 小图：固定 `M=N=17`，在 4 hart 上分别关闭、开启循环展开。
普通样本使用 `K=127、128、129、130`，覆盖除以 4 后余数 0、1、2、3 的循环尾部，
同时覆盖不能均分给 4 个 hart 的 17 行输出；另有一组 `K=129` 的强抵消样本。

普通样本的 8 次运行全部有限，满足 ORT `max_abs < 0.001`、`rtol=0` 和既有 FP64 预算。
最大 ORT 误差为 **7.62939453125e-6**，最大 FP64 绝对误差为
**1.216086167943331e-5**，误差与相应预算之比最大约 **0.053742**。

强抵消样本的 2 次运行使用原有 `gamma_K × sum(|各乘积|)` 预算，
并以按 K 顺序执行、分开乘法和加法的 FP32 结果作逐位检查；**此样本不采用 ORT 绝对误差门槛**。
它对 FP64 的最大绝对误差为 **1.0**，预算占比约 **0.003876**，仍通过该项既定标准。
不能把这 2 次归入“ORT 误差小于 0.001”的结论。

全部 10 次都与独立顺序 FP32 参考逐位一致；开关前后的结果也逐位一致，
输入和权重未被修改，真实退出码均为 0，事件等待均无回退。
总流程 **1.780947 秒**只说明这批检查的执行时间，不用于性能判断。

小图脚本为本地 `output/issue109-nmsis-profile/candidate_correctness.py`，
结果与各图输入、权重、参考、误差预算、生成源码、ELF、运行日志和输出保存在
`output/issue109-nmsis-profile/candidate-correctness/`，日志为
`output/issue109-nmsis-profile/candidate-correctness.log`；
可分享结果见 [验收摘要](external-matmul-evidence.json)。

本轮合计 58 次定向执行，三批数据分别保存；没有运行完整模型，
没有把候选宏设为默认，没有覆盖此前测量配置或历史归档。

## 按四列整理输入与本地分块候选

沿用上一轮的 QK、AV、第一层 Q 投影，以及 RV64GC / LP64D、4 hart、按输出行并行配置。
本轮保留原优化和已启用 `RISCV_MATH_LOOPUNROLL` 的官方 NMSIS 参照，只比较两个候选：

| 候选 | 输入整理与实际计算 | 逐位一致参照 |
| --- | --- | --- |
| A：四列整理后调用官方算子 | 将 B 的四列复制成连续小矩阵，调用未改动的官方循环展开算子，再写回对应输出列 | 官方 NMSIS 循环展开 |
| B：四列整理加本地分块内核 | 使用相同整理；在 NMSIS 标量源码副本中复用 ScratchV `blocked_fma / 2x4`，每 128 个 K 元素使用融合乘加，再单独累加各块 | 原优化 `blocked_fma / 2x4` |

**A 仍调用官方算子；B 是本地派生实现，不能称为原版 NMSIS 加速。**
B 保留许可证、上游参数检查和来源副本。两者不改生产源码或官方检出，仍使用严格浮点编译参数；
B 的融合乘加来自明确的内建调用。反汇编中 A 的官方矩阵函数仍为 278 字节、无融合乘加，
B 的本地函数为 808 字节、有 8 处融合乘加；这是静态代码证据，不是执行次数。

两者复用六参数 C 接口和普通行布局，不拆 K，末尾不足四列时按实际宽度紧凑存放。
每个 hart 自行整理 B，自动栈缓冲区固定 **20 KiB／调用**，另有普通函数栈开销；
沿用每 hart 1 MiB 栈，无堆分配或共享可写缓冲区。
实验逐次调用上限为 `M≤256、K≤1024、N≤2048`，M 是分给该 hart 的行数。
现有单一 `max_dimension=2048` 不能完整表示逐轴上限，候选尚不是通用生产 provider。
上限、空指针、零维和失败后不散写的分支本轮只作静态审查，没有新增调用负例。

### 性能与选用结论

复用已保存的固定种子输入、实际第一层权重和参考；激活不是真实模型中间值的捕获。
QK/AV 保留真实批次与动态右侧，Q 投影保留运行时转置；原实现使用原有静态列分块权重。
没有新增性能插桩或计时回调，原按行并行回调仍在使用。
每例每种实现 1 次预热、3 次正式运行，顺序轮换且 QEMU 独占。**48 次全部通过**，
总准备与验证流程 **81.780086 秒**。下表是正式前向中位数，单位秒：

| 矩阵 | 原优化实现 | 官方 NMSIS 循环展开 | A：整理后调用官方 | B：本地分块候选 |
| --- | ---: | ---: | ---: | ---: |
| QK | 0.365956 | 0.445717 | 0.439477 | 0.396808 |
| AV | 0.299955 | 0.390322 | 0.394557 | 0.381827 |
| Q 投影 | 1.010894 | 11.886615 | 1.497894 | 1.446597 |

Q 投影的 A 比本轮官方循环展开参照缩短 **87.40%**，说明数据布局对这一形状影响显著。
但 A、B 仍分别耗费原实现的 **1.48 倍、1.43 倍时间**。
B 相对官方参照缩短 87.83%，其中包含本地内核与数值顺序变化，不能全部归为官方库收益。
本轮没有硬件计数器证据，不能进一步定位为缓存或 TLB 的具体改善。

QK 的 A 比官方参照仅少 1.40%，AV 的 A 反而多 1.09%，样本范围重叠；
QK 原实现本身的正式范围是 **0.300290–0.391551 秒**，短样本波动明显。
不以这些中位差宣称稳定加速，继续使用**候选最慢正式时间 ≤ 原实现最快正式时间 × 0.9**。
A、B 在三类矩阵上的六项均未达到门槛，生产策略、默认构建、poll 和旧实测配置保持原样。

沿用原 `forward_wall_s`，从 QEMU 启动前到 UART 完成校验结束。
启动、即时翻译、guest 检查、必要转置以及候选每次整理、包装调用、结果散写**全部计时**；
代码生成、静态权重排布、编译、运输准备、输出转存和数值比较不在前向时间内。
这些是本机的三次正式样本，没有运行完整模型，也不能外推为真实硬件收益。

48 次输出全部有限，通过 ORT `max_abs < 0.001`、`rtol=0` 和原 FP64 预算；
最大 ORT 误差 **7.748603820800781e-7**，最大预算占比约 **0.007309**。
A 与官方循环展开、B 与原优化实现分别逐位一致；全部退出码为 0，事件等待未回退。

### 边界验证与数据

另有 **28 次只检查正确性的小图通过**，流程 **5.459337 秒**，不参与性能准入。
六个普通图覆盖 K=127–130、不同列尾部、不均匀行分配和四维批次广播，各测四种实现。
这 24 次使用相同 ORT/FP64 门槛，最大 ORT 误差 **7.62939453125e-6**，
最大 FP64 绝对误差 **1.8405275522326292e-5**，预算占比最大约 **0.053167**。
另一个图覆盖次正规数、带符号零和跨 K=128 分界的融合乘加抵消，四种实现均有限且通过各自逐位检查：
A 对官方顺序实现，B 对原分块实现及既有精确融合乘加参考。
这四次只记录 FP64 最大绝对差 **1.0**，**没有使用 ORT 或 gamma 预算门槛**，
不能并入“全部误差小于 0.001”。28 次输入和权重未变，退出码均为 0，事件等待未回退。

本轮共 76 次定向执行。候选源码、上游副本和差异、输入、参考、权重、全部输出、ELF、
反汇编、构建命令及日志保存在本地 `output/issue109-nmsis-blocking/`。
性能明细为 `measurements/report.json`，边界明细为 `boundary-correctness/report.json`；
入口脚本为 `compare_blocking.py`、`boundary_correctness.py`，同目录保留各自日志。
[验收摘要](external-matmul-evidence.json) 保存完整形状、各次正式耗时、最小值／中位数／最大值及门槛。
本轮没有新增 tar 归档，没有覆盖旧数据、测量配置或历史归档；原始大文件仍是本地材料。

## Q 投影静态权重复用与直接输出

本轮只测同一 Q 投影 `[1,256,1024] @ [1024,2048]`，继续 RV64GC / LP64D、4 hart。
原优化与两个新候选共用已有生成器和序列化器产生的 **8 MiB `[512,1024,4]` 静态权重包**。
取消原有的运行时权重转置，以及每个 hart 的四列整理；共用的权重、输入校验和公共输出拷贝均保留。

- **静态 A**：直接把只读 K×4 面板传给官方 NMSIS 循环展开算子；仍需每次调用 4 KiB 的紧凑结果缓冲和散写。
- **静态 B**：本地派生的 K128 分组融合乘加内核，明确四列限制与输出行跨度，直接写最终矩阵，去掉上述额外缓冲和散写。

这是明确命名的本地面板接口，**没有把静态面板冒充原六参数接口中的普通 B 矩阵**。
现有生产接口、官方检出和旧实验不变。候选限单个常量 MatMul、二维或单批次三维输入，
`M≤256、K≤1024、N≤2048` 且 N 必须为 4 的倍数；并行仍复用原行分配、错误汇总和等待全部线程结束。
B 的四列约束还可能使编译器进一步简化代码，收益属于整个候选，不能单独归因为删除散写。
公共“工作区到输出”的最后一次拷贝没有删除；B 也仍是本地派生内核，不是官方 NMSIS。
删除运行时转置后，工作区从旧候选的 **10 MiB 降到 2 MiB**；每次调用的额外数组由
旧候选 20 KiB 降为 A 的 4 KiB、B 的 0，B 仍有普通局部变量和函数栈帧。

### 四方实测

原优化、上一轮“运行时整理＋本地分块”候选及新 A/B 在本轮一起重测，
各 1 次预热、3 次正式运行，顺序轮换、QEMU 独占。**16 次通过**，总流程 **23.184959 秒**。
下表的准备成本各计一次，前向为正式中位数：

| 实现 | 准备成本／毫秒 | 前向／秒 | 前向相对原优化 |
| --- | ---: | ---: | ---: |
| 原优化静态面板 | 35.483 | 1.084294 | — |
| 旧本地候选，本轮重测 | 21.879 | 1.474226 | +35.96% |
| 静态 A，官方算子 | 36.396 | 1.356229 | +25.08% |
| 静态 B，本地直接输出 | 36.012 | 1.215148 | +12.07% |

静态 B 比**本轮重测**的旧本地候选缩短 **17.57%**，但仍慢于原优化。
这里不拿前轮绝对时间代替同期对照，也不宣称完整模型或真实硬件有相同收益。

共用准备成本 P=**35.483 毫秒**，包括生成视图和有限值检查 12.763 毫秒、
实际按面板顺序序列化并落盘 22.720 毫秒；仅建立数组视图不算完成物理整理。
A/B 另计各自局部源码适配 0.913／0.529 毫秒。P 只测了一次，三种静态实现各归属一次同样成本，
不是准备成本的重复测量分布，也没有将复用后的准备成本算作零。

主门槛是**候选 P＋最慢正式前向 ≤ 0.9×（原优化 P＋最快正式前向）**。
A/B 左侧分别为 **1.577472／1.276328 秒**，原优化的比较值为 **1.097821 秒**，两者均未通过。
只看前向和“准备一次、复用三次”的附加估算也未通过；后者是阶段计时的组合，
不是同一 QEMU 内连续热运行。生产策略、默认构建、poll 和旧实测配置保持不变。
前向仍包含启动、即时翻译、图检查、计算和实际存在的拷贝；编译、运输准备和输出转存另记。

16 次全部有限，满足原 ORT `max_abs < 0.001`、`rtol=0` 与 FP64 预算；
最大 ORT 误差 **3.725290298461914e-7**，最大 FP64 预算占比约 **0.003975**。
静态 A 与已保存的官方循环展开结果逐位一致，旧／新本地候选均与本轮原优化逐位一致。
实际符号、4 hart、退出码 0 和事件等待无回退均已核对。

### 边界与保存位置

另有 **12 次边界运行和 8 项生成拒绝检查通过**。三种实现在 K=127／128／129 的普通图上
共运行 9 次，最大 ORT 误差 **7.62939453125e-6**、FP64 绝对误差 **1.6121500038934755e-5**，
预算占比约 **0.057368**；覆盖奇数行分片、多个面板与输出行跨度。
特殊浮点图的 3 次按各自逐位参考检查，B 另对精确分组融合乘加参考，**没有套用 ORT 或 gamma 门槛**。
候选在同一次 guest 中另检查输出前后保护值、M=3/N=8/K=3 的直接调用、非法 N=3 拒绝且不写输出，
以及共享面板内容不变；生成期 N=3／5／6／7 两候选全部拒绝。边界流程 **2.333447 秒**不用于性能判断。

原始材料保存在本地 `output/issue109-q-projection-static/`：
`measurements/report.json` 保存准备成本、全部样本与准入计算，`boundary-static/report.json` 保存边界结果；
候选与共享 helper、源码快照、输入／权重／参考／输出、ELF、反汇编、构建和运行记录均保留。
[验收摘要](external-matmul-evidence.json) 另保存可分享数值。本轮没有运行完整模型或新增 tar 归档，
没有覆盖任何旧实验、旧配置或历史归档。

## Q 投影热循环整理与内联对照

继续只测上述 Q 投影、RV64GC / LP64D、4 hart，共用同一 8 MiB 静态权重包和 2 MiB 工作区。
两个新候选均为**本地派生实现**，保留原 K128 分组融合乘加顺序、公共校验和最后输出拷贝：

- `paired_core`：把完整两行和奇数尾行分开，缓存描述符中的基址；保留独立核心函数及调用边界。
- `inline_core`：复用完全相同的核心源码，仅将核心内联到包装函数；模型到包装函数的边界仍保留，不启用全局 LTO。

旧核心在每个 K 步仍有 5 次第二行是否存在的条件判断，而本例每个 hart 恰好处理 64 行。
新完整行对的热循环已移除这些判断，并缓存输出基址；奇数尾行另有独立路径。
按本图的实际循环范围推算，旧版和 `paired_core` 每次前向各有 2048 次核心调用。
反汇编确认成对行版仍实际调用独立核心，内联版没有独立核心或对应调用。
四种实现均为 **1 次并行派发、4 次范围回调**，输出行归属相同；本轮没有修改调度。
内联还让编译器看见更多代码，耗时差不能全部归为调用／返回指令的成本。

每种实现 1 次预热、3 次正式运行，顺序轮换且 QEMU 独占，**16 次通过**。
下表只比较本轮同期数据；此前约 12% 的差距不是固定常数：

| 实现 | 准备成本／毫秒 | 前向中位／秒 | 正式前向范围／秒 | 中位相对原优化 |
| --- | ---: | ---: | ---: | ---: |
| 原优化 | 35.139 | 1.075534 | 1.074071–1.085592 | — |
| 上轮静态直接输出，本轮重测 | 35.981 | 1.262352 | 1.261343–1.343355 | +17.37% |
| 分离完整行对与尾行 | 35.668 | 1.089382 | 1.035867–1.142011 | +1.29% |
| 同核心内联到包装函数 | 35.625 | 1.028367 | 1.002810–1.093196 | −4.39% |

两候选相对本轮旧静态实现的中位耗时分别减少 **13.70%／18.54%**。
内联候选中位值低于原优化，但正式范围重叠，不能据三个样本宣称稳定加速。
共用准备成本 P=**35.139 毫秒**，包含生成与检查 11.821 毫秒、实际序列化与落盘 23.318 毫秒；
两候选另计源码适配 0.529／0.486 毫秒。P 仍只测一次，每个独立部署各归属一次。

沿用准备成本加最慢正式前向的主门槛，两候选左侧为 **1.177679／1.128822 秒**，
原优化最快前向加 P 为 **1.109211 秒**，其 90% 为 **0.998290 秒**，均未通过。
只看前向及“准备一次、复用三次”的附加估算也未通过；复用估算不是同一 guest 的热缓存测量。
**生产实现、默认构建、poll 和旧实测配置保持不变。** 计时边界沿用上节，
没有新增性能插桩或计时回调，原并行回调继续使用；本轮总准备与验证流程为 20.154848 秒。

16 次输出全部有限且与本轮原优化逐位一致，严格 ORT 最大绝对误差为
**1.341104507446289e-7**（门槛 `<0.001`、`rtol=0`），FP64 预算占比最大约 **0.000861**。
另有 **12 次边界运行、8 项生成拒绝检查通过**：9 次普通图最大 ORT 误差
**1.9073486328125e-6**，FP64 预算占比最大约 **0.052413**；覆盖 K=127／128／129、
奇数行、多面板和输出行跨度。3 次特殊浮点图均与精确分组融合乘加参考逐位一致，
**没有套用 ORT 或 gamma 门槛**。保护值、非法 N=3 不写输出及共享面板不变检查均通过。
全部运行退出码为 0、事件等待无回退；边界流程 2.180957 秒不参与性能判断。

本轮材料保存在本地 `output/issue109-q-projection-kernel/`：
`measurements/report.json`、`boundary-kernel/report.json` 分别保存性能和边界结果，
`analysis/` 与运行目录保存静态诊断、反汇编、实际构建和调用证据。
候选共用的 `shared_core.h`、源码快照、输入／权重／参考／输出、ELF 和日志均保留。
[验收摘要](external-matmul-evidence.json) 保存可分享统计。本轮没有运行完整模型、
新增 tar 归档或覆盖旧阶段数据，不能把结果外推为真实硬件收益。

## Q 投影内联版的遍历与分块复用对照

**本轮两个候选均未达到计入准备成本后至少快 10% 的门槛，保留当前生产选择。**
仍只测第一层 Q 投影：代表激活 `[1,256,1024]`、实际权重 `[2048,1024]`，
RV64GC / LP64D、4 hart。沿用原 8 MiB 静态列分块权重包和 2 MiB 工作区，
直接包含上一轮未修改的内联核心，不改 K128 分组乘加顺序。

上轮按“四列块→全部本地行对”遍历；本轮只增加两个受控候选：

- 两行优先：两行→四列块，集中访问这两行输入。
- 八行组：最多八行→四列块→行对，在组内重复使用相同权重块，并保留最后不足八行的处理。

本例每个 hart 处理 64 行。上述输入行的逻辑地址跨度分别为上轮 256 KiB、
两行版 8 KiB、八行版 32 KiB；这是源码推导，**不是缓存占用或命中率测量**。
计算量和逻辑加载量未减少，编译器也可能利用较小行组边界优化代码，不能把时间差全部归因于缓存。
四方仍各有一次并行派发、四次回调；本地版保留模型到包装函数的调用，内部核心均已内联。
这是本地派生的显式 panel4 实验接口，不是未经修改的官方 NMSIS 或现有行主序 v1 接口。

反汇编中四方完整行对的最内层循环均为 6 次浮点加载、8 次融合乘加，没有核心调用或栈读写。
原优化每次循环有 18 条静态指令，三个本地版为 19 条，差别在地址更新／循环控制。
八行版另有外层整数状态的栈暂存，并非最内层浮点累加器溢出。
这些事实没有显示出可仅通过改变遍历顺序消除的大量最内层冗余，不能直接换算为耗时占比。

四种实现同期重建、顺序轮换，每种一次预热、三次正式运行；同一时刻只运行一个 QEMU。
**16 次性能运行通过**，准备与验证流程共 18.989817 秒。前向正式结果如下：

| 实现 | 准备成本／毫秒 | 前向中位／秒 | 正式前向范围／秒 | 中位相对原优化 |
| --- | ---: | ---: | ---: | ---: |
| 原优化 | 32.981 | 1.070705 | 1.034697–1.075771 | — |
| 上轮内联版，本轮重测 | 33.760 | 1.070682 | 1.017093–1.088346 | 约持平 |
| 两行优先 | 33.393 | 1.051373 | 1.016511–1.063378 | −1.81% |
| 八行组 | 33.362 | 1.063585 | 1.011116–1.075956 | −0.66% |

共用准备 P=32.981 毫秒，含生成与检查 11.677 毫秒、权重实际序列化／检查／落盘 21.304 毫秒；
本地三版另计源码适配 0.779／0.411／0.381 毫秒。P 每条路径只测一次。
沿用原始前向计时（创建 QEMU 前至完成帧校验，含启动、即时翻译、计算和最终输出拷贝），
编译、运输准备、输出转存和数值比较不在 P+前向准入时间中。

保守主门槛为“候选 P+最慢正式前向 ≤ 原优化 P+最快正式前向的 90%”：
原优化右侧基线为 1.067678 秒，90% 为 **0.960910 秒**；
两行版为 **1.096771 秒**，八行版为 **1.109318 秒**，均不满足。
纯前向和“准备一次、复用三次”的附加估算也未通过；复用估算不是同一 guest 热缓存实测。
所有前向范围重叠，上轮约 4.39% 的中位差在本轮也没有保留，不能宣称稳定提速。
当前证据不支持仅靠这两种遍历顺序替换原优化；生产代码、poll 和既有实测配置保持原样。

16 次输出全部有限、与本轮原优化逐位一致，最大 ORT 绝对误差 **1.341104507446289e-7**，
严格满足 `<0.001`、`rtol=0`；FP64 最大绝对误差 **8.85283508722523e-8**，
既有预算占比最大约 0.000861。另 **12 次边界运行、8 项生成拒绝检查通过**，
普通图最大 ORT 误差 **1.9073486328125e-6**，FP64 预算占比最大约 0.052413。
覆盖 K=127／128／129、奇数行、多面板与输出跨度；特殊浮点图沿用精确 K128-FMA 位比较，
不套用 ORT 或 gamma 门槛。原 M=3 探针保留，同一次 guest 新增 M=9/N=8/K=3 探针，
检查八行组后的尾一行、输入和权重未变、输出保护及非法 N=3 不写输出，全部通过。
28 次均退出 0、事件等待无回退；边界流程 2.470200 秒不用于性能判断。

数据保存在本地 `output/issue109-q-projection-reuse/`：
`measurements/report.json`、`boundary-reuse/report.json` 保存性能和正确性记录；
两份脚本、冻结源码快照、共享核心、输入／真实权重／参考／输出、ELF、反汇编和日志均已保留。
[验收摘要](external-matmul-evidence.json) 保存可分享统计。历史实验与归档未覆盖，
未重跑完整模型，未提交、推送或提 PR；结果限于本机 QEMU 的三次正式样本。

## Q 投影内联版的指令与地址计算对照

**两个新候选均未达到计入准备成本后至少快 10% 的门槛，不替换当前生产实现。**
本轮基于 `735dd00` 和已有本地改动，只测同一第一层 Q 投影：代表激活 `[1,256,1024]`、
实际权重 `[2048,1024]`，RV64GC / LP64D、4 hart。四方共用原 8 MiB 静态列分块权重包与
2 MiB 工作区，沿用行对优先遍历、K128 分组融合乘加、输入检查和最后输出拷贝。

A 改用指针终点控制循环；B 使用同一计算体，仅在 `N==2048 && K==1024` 时传入常量，
其他受支持尺寸走 A 的通用路径。这是本地派生的显式 panel4 接口实验，不是自动选库，
也不代表未经修改的官方 NMSIS。原调度仍为一次派发、四次回调，每个 hart 分配 64 行。

实际 ELF 显示：旧行对版与 A 的最内层行对循环体都是 **19 条静态指令**，
仍有 6 次加载、8 次融合乘加，指针写法没有减少该循环体的指令数。
B 的限定尺寸路径降到 **18 条**，但包装函数增至 **1152 字节**，旧版为 632 字节、A 为 666 字节。
核心和行组函数均已内联，已检查的计算循环没有核心调用或栈读写；模型到包装函数的调用保留。
静态指令数下降没有转化为本次提速，不能把时间差单独归为代码大小、缓存或即时翻译成本。

四方各一次预热、三次正式运行，轮换顺序且独占 QEMU，**16 次全部通过**：

| 实现 | 准备成本／毫秒 | 前向中位／秒 | 正式前向范围／秒 | 中位相对原优化 |
| --- | ---: | ---: | ---: | ---: |
| 原优化 | 40.752 | 1.007092 | 0.995884–1.070494 | — |
| 上轮行对优先，本轮重测 | 41.637 | 1.048140 | 1.037677–1.069266 | +4.08% |
| 指针终点 A | 41.169 | 1.054396 | 1.022796–1.114251 | +4.70% |
| 限定尺寸常量路径 B | 41.166 | 1.092860 | 1.092486–1.126010 | +8.52% |

共同准备 P=**40.752 毫秒**，含生成与检查 11.693 毫秒、实际权重序列化／检查／落盘
29.059 毫秒；三个本地版本另计源码适配 0.885／0.417／0.414 毫秒。P 每条路径只测一次。
主门槛“候选 P+最慢前向 ≤ 原优化 P+最快前向的 90%”右侧为 **0.9329724468 秒**，
A 为 **1.155419582 秒**、B 为 **1.167176558 秒**，均未通过；纯前向和复用三次的成本估算也未通过。
复用估算不是同一 guest 的热缓存实测。前向保留原计时：创建 QEMU 前至完成帧校验，含启动、
即时翻译、计算与输出拷贝；编译、运输准备、输出转存和比较不计入 P+前向。性能验证流程为
19.403478 秒，三次正式样本不足以给出稳定的硬件性能或单条指令成本结论。

16 次性能输出全部有限且逐位等于原优化；最大 ORT 误差 **1.341104507446289e-7**，
满足严格 `<0.001`、`rtol=0`，FP64 最大绝对误差 **8.85283508722523e-8**，预算占比最大约 0.000861。
另 **12 次边界运行、8 项生成拒绝检查通过**：9 次普通图仍用 ORT／FP64 门槛，
最大 ORT 误差 **1.9073486328125e-6**、FP64 预算占比约 0.052413；3 次特殊浮点图用精确
K128-FMA 逐位比较，不套用 ORT 或 gamma 门槛。候选边界运行还在同一次 guest 中检查
`M=9/N=2048/K=1024` 限定路径和只匹配一个条件的邻近尺寸 `M=1/N=2048/K=129`、
`M=1/N=8/K=1024`，用精确小整数参考核对完整输入、权重、输出及保护值，未增加 QEMU 次数。
探针缓冲仅加入边界制品的工作区，性能制品仍为 2 MiB。合计 **28 次运行＋8 项拒绝检查**，
均通过；运行退出码为 0、事件等待无回退。边界流程 4.187492 秒不作性能结论。

材料保存在本地 `output/issue109-q-projection-address/`：`measurements/report.json`、
`boundary-address/report.json`、`analysis/assembly-review.json`、`candidate-notes.json`，
以及脚本、源码快照、输入／权重／参考／输出、ELF、反汇编和日志。
[验收摘要](external-matmul-evidence.json) 保存可分享统计，原始文件仍为本地数据。
生产代码、poll、既有实测配置和历史归档保持不变；没有重跑完整模型、新增归档、提交、推送或提 PR。

## Q 投影冷启动与同一 QEMU 重复执行对照

**本轮不支持“限定尺寸版稳定慢约 8.5%”的结论，也没有达到替换原优化的门槛。**
同一 QEMU 内后续调用的 public（单次公开函数）耗时，两方汇总中位数只差约 **0.058%**，
计算与同步的 dispatch 区间也近似相等。冷启动中候选中位数慢 **2.88%**，三对正式运行中
候选两次较慢、一次较快。历史实验与本轮的宿主采样方式不同，不能精准归因百分比的变化。

仍只使用同一真实第一层 Q 权重与代表激活，RV64GC / LP64D、4 hart、CPU 亲和集合 0–3、
当前 poll，基于 `735dd00` 加已有本地实现。两份未插桩生成 C 均与上轮地址实验逐字相同。
两方共用 8 MiB panel4 权重与 2 MiB 工作区；候选仍是本地派生接口，不是未经修改的官方 NMSIS。

冷启动各一次预热、三次正式运行，正式顺序为原优化→候选、候选→原优化、原优化→候选，
共 **8 个 QEMU 进程**。原始前向计时仍从创建 QEMU 前到完成帧校验：

| 实现 | 准备 P／毫秒 | 冷启动前向中位／秒 | 三次正式范围／秒 |
| --- | ---: | ---: | ---: |
| 原优化 | 33.872 | 1.023162 | 1.015428–1.036781 |
| 限定尺寸常量版 | 34.819 | 1.052606 | 1.024562–1.086968 |

共同 P 包括生成／检查 11.856 毫秒与实际权重序列化／检查／落盘 22.016 毫秒，
候选另加适配 0.947 毫秒；准备只测一次。编译、诊断源码改写、运输准备、输出转存与比较不计入 P。
**唯一采用门槛仍是未插桩冷启动的 P+最慢前向 ≤ 原优化 P+最快前向的 90%**：
右侧 **0.9443698677 秒**，候选 **1.121786539 秒**，未通过。

诊断另用 **6 个 QEMU 进程**，两方各三个，按相同的三对顺序运行。每个进程连续调用五次，
每次结果直接写入独立输出槽并保存；运输输出因此由 2 MiB 增为 10 MiB，工作区仍为 2 MiB。
从同一 QEMU 的设备树读得时钟频率 **10 MHz**；guest 的 rdtime 表示虚拟时间，不是 CPU 周期。

| 诊断区间／秒 | 原优化 | 限定尺寸常量版 |
| --- | ---: | ---: |
| 首次 public：三个进程的中位 | 1.034090 | 1.087940 |
| 后四轮 public：先各进程取中位，再汇总三个中位 | 1.005499 | 1.006079 |
| 后四轮 dispatch：同样按进程汇总 | 0.982382 | 0.981636 |

后续 public 的三个进程中位范围分别为 **0.998308–1.017467** 与 **0.987070–1.011937 秒**，
明显重叠；不能把进程内的十二个后续调用当作十二份独立样本。后续输入检查约 6.5–6.7 毫秒、
有限性检查约 1.4 毫秒、最终拷贝约 15 毫秒。每轮先向工作区和该输出槽写入 NaN，
其约 **14.4–14.5 毫秒**另记为准备，不在 public 内，但会影响执行状态。
独立输出槽、外围计时及额外运输改变了诊断制品；空计时对也不能代表全部插桩扰动。

实际 ELF 的直接字节比较确认，原优化热回调 422 字节、候选完整包装函数 1152 字节及其行回调
74 字节在各自冷启动／重复制品中完全相同，启动汇编与相应 worker 代码也保留。
重复 driver 是一个真实函数调用点循环五次；这不代表整个 ELF 或外围内存布局相同。

首次并不总比后续慢，不能把变化全部归于即时翻译。host 前向与 guest driver 包络的差值为
**28.66–32.70 毫秒**，只是未覆盖的启动、完成处理及两种时钟等残差，不是独立测得的启动或 JIT 成本。
本轮两阶段都用既有资源采样器，每 0.5 秒采一次，替换原先每 0.2 秒的 RSS 采样线程。
冷进程只有三个前向采样点，重复进程各十一个；采样线程自身约占 4.8–6.8／13.3–16.1 毫秒 CPU。
已采样边界中重大缺页和进程换页驻留均为零，系统观测窗口的换入／换出增量也为零，
没有本次换页造成差异的证据；系统忙率约 94.8%–97.0%，负载与压力指标包含其他 WSL 工作，
不能据此排除宿主调度影响，更不能将系统活动全归于 QEMU。资源边界也不是精确的每轮 CPU 计时。

合计 **14 个模型进程、38 次 Q 子图调用全部通过**，另一次设备树导出不计模型运行，
准备与验证总流程 **42.455073 秒**。所有输出有限、逐位等于本轮冷启动原优化；
最大 ORT 误差 **1.341104507446289e-7**（严格 `<0.001`、`rtol=0`），
FP64 最大绝对误差 **8.85283508722523e-8**，预算占比最大约 0.000861。
每个进程结束时，输入与完整物理权重读回字节一致；这是末态检查，不证明运行中从未暂时写入。
所有进程均退出 0、事件等待无回退，没有残留 QEMU。

本地材料在 `output/issue109-q-projection-warm/`：`measurements/report.json`、`measurements.log`、
`analysis/assembly-control.json`、两份脚本、原始／诊断 C、ELF、共享权重、输入与参考、
38 份独立输出、逐轮时钟表、输入／权重末态转存及资源 JSONL 均保留。
[验收摘要](external-matmul-evidence.json) 提供可分享统计，不是原始数据替代品。
本轮未重跑完整模型，未修改生产策略、poll 或既有实测配置，也未新增归档。

## Q 投影真正 3×4 内核对照

**真正共享三行权重加载的候选已通过机器码与正确性验证，但未达到计入准备成本后快 10% 的门槛。**
本轮仍为同一第一层 Q 真实权重与代表激活 `[1,256,1024] @ [1024,2048]`，
RV64GC / LP64D、4 hart、CPU 亲和集合 0–3、严格浮点参数与当前 poll。
候选保留 8 MiB 四列权重包、2 MiB 工作区、递增 K 的融合乘加顺序及每 K128 块后的独立加法，
一次真正计算三行四列，不足三行复用两行／单行尾部；每个 hart 的 64 行为 21 个三行组加一行。
它仍是本地派生 panel4 接口，不是官方 NMSIS 或生产 provider；生成器基底的 2×4 元数据未改，
实际计算由候选源码和机器码确认，公共校验、输出拷贝与一次派发／四次回调保持原样。

先构建审查，再执行。原两行主循环每 K 为 6 次加载、8 次融合乘加、18 条指令；
候选固定尺寸三行主循环为 **7 次加载、12 次融合乘加、24 条指令**。
按每 hart 的 64 行和尾行完成同量输出，加载从 **192 降至 152（−20.83%）**，
最内层循环指令从 **576 降至 517（−10.24%）**，融合乘加仍为 256 次。
这是由已编译循环推导的数量，不含外层、检查和启动，不是硬件缓存流量或时间预测。
所有最内层乘加循环无栈读写或独立核心调用；包装函数仍有 **224 字节栈帧**，
保存／恢复调用约定要求的寄存器并暂存外层整数状态，不能称整个函数“无栈访问”。
候选包装函数 2096 字节含尺寸分支和通用尾部，原 422 字节是范围回调，职责范围不同。
四个最终冷／重复制品各自完整热函数字节均与预审相同，才允许启动 QEMU。

冷启动每方一次预热、三次正式运行，按 AB／BA／AB 轮换；下表时间单位秒：

| 范围 | 原优化 | 真正 3×4 |
| --- | ---: | ---: |
| 准备 P，仅一次观察 | 0.035207 | 0.036137 |
| 冷启动前向中位 | 1.061845 | 1.034265 |
| 冷启动正式范围 | 1.026494–1.105366 | 0.963700–1.034818 |
| 同 guest 首次 public，三个进程中位 | 0.980231 | 1.030739 |
| 后四轮 public，先各进程取中位再汇总 | 1.024081 | 0.988554 |
| 后四轮计算与同步 dispatch，同样汇总 | 1.002015 | 0.965869 |

共用 P 包括生成／检查 12.378 毫秒、真实权重序列化／检查／落盘 22.829 毫秒，
候选另加适配 0.930 毫秒；编译、运输及诊断辅助工作不混入 P。
冷启动中位少 **2.60%**；后续 public 少 **3.47%**，其三个进程中位范围分别为
**0.971277–1.044533／0.980527–0.995719 秒**。两种范围都重叠，配对方向也不全一致。
主门槛仍是 `P候选+最慢冷前向 ≤ 0.9×(P原优化+最快冷前向)`：
候选 **1.070955602 秒**，要求不超过 **0.955530833 秒**，未通过，不采用。

重复诊断每方三个进程、每进程五次，时钟为同一 QEMU 设备树确认的 10 MHz。
后四轮相关，不能当作十二个独立样本；每轮 NaN 准备约14毫秒另计，10 MiB 独立输出槽及外围计时会影响状态。
冷启动与内部计时范围不同；host 与 guest driver 的 **26.11–37.36 毫秒**残差不是纯启动或即时翻译成本。
0.5 秒资源采样共102份、无错误；前向每进程仅2–11份完整样本，末端空隙 **20.6–499.5毫秒**。
采样边界未见重大缺页／进程换页驻留，系统窗口换入／换出增量为零，仍不能排除宿主调度影响。

性能批次 **14个进程、38次Q调用通过**，另1次设备树导出，流程 **42.175809秒**。
全部输出有限、逐位等于本轮原优化；严格 ORT 最大误差 **1.341104507446289e-7**（`<0.001`、`rtol=0`），
FP64最大误差 **8.85283508722523e-8**，预算占比最大约0.000861；输入／权重末态读回一致。
另 **14次边界运行＋4项生成拒绝通过**，流程 **2.849102秒**不参与性能判断：
12次普通图覆盖M=2/3/4/63/64/65、K=127/128/129和分片余数0/1/2，
最大ORT误差 **1.9073486328125e-6**，FP64最大误差 **1.1812826475221527e-5**，预算占比约0.0477255。
2次特殊浮点图只按精确K128融合乘加位参考检查，不套用ORT／gamma门槛。
候选同guest内的M1–8直接探针、非法N=3不写输出、M5/N2048/K1024固定分支两行尾部及保护值均通过。
合计 **28个模型进程、52个主图输出**，额外直接调用不混入主图计数；退出码均0、事件等待无回退。

完整材料保存在本地 `output/issue109-q-projection-3x4/`，
主记录为 `preflight/machine-review.json`、`measurements/report.json`、`boundary/report.json`；
源码、ELF、输入／权重／参考／输出、逐轮时钟、资源JSONL与日志均保留。
[验收摘要](external-matmul-evidence.json) 提供可分享统计。本轮未改生产实现、poll、旧配置或官方NMSIS，
未重跑完整模型、未覆盖历史数据、未新增归档，未提交、推送或提PR。

## Q 投影原 2×4 的 K 两步展开对照

**两步展开通过机器码和数值验证，但没有达到至少快 10% 的采用门槛，也未满足继续与 3×4 组合的条件。**
仍只测第一层真实 Q 权重与代表激活 `[1,256,1024] @ [1024,2048]`，
保持 RV64GC / LP64D、4 hart、CPU 亲和集合 0–3、严格浮点参数和当前 poll。
本轮直接从原生 2×4 生成 C 修改串行／并行两处 K 循环，没有新增外部包装函数或编译参数。
每次依次执行原来的 K 与 K+1 计算体，奇数余项仍走原体；不跨 K128 边界，不拆分累加器。
权重布局、2 MiB 工作区、任务划分、行列尾部检查、公共检查与最后拷贝均沿用原实现。

先审查机器码：相同两步 K 工作量从 **36 条静态指令降为 32 条（−11.11%）**，
**12 次加载、16 次融合乘加没有减少**；减少的是循环控制和地址更新。
热循环没有栈读写、寄存器溢出或函数调用，但外围仍有尾部检查。
完整范围函数从 **422 增至 620 字节**，调用约定栈帧从 **96 增至 128 字节**，
不能称整个函数无栈访问。四份最终冷／重复制品的热函数字节都与各自预审一致后才运行。

冷启动每方一次预热、三次正式运行；重复诊断每方三个独立进程，每进程五次，
两阶段正式顺序均为 AB／BA／AB，独占 QEMU；下表单位为秒：

| 范围 | 原优化 2×4 | K 两步展开 |
| --- | ---: | ---: |
| 准备 P，仅一次观察 | 0.033187 | 0.033629 |
| 冷启动前向中位 | 1.045556 | 1.014245 |
| 冷启动正式范围 | 1.002507–1.055632 | 0.997607–1.017878 |
| 同 guest 首次 public，三个进程中位 | 0.965670 | 0.970484 |
| 后四轮 public，先各进程取中位再汇总 | 0.999895 | 0.983538 |
| 上项三个进程中位的范围 | 0.986370–0.999982 | 0.961047–0.984640 |
| 后四轮计算与同步 dispatch，同样汇总 | 0.977603 | 0.960703 |

共用 P 包含生成／检查 **11.554 毫秒**、真实权重序列化／检查／落盘 **21.633 毫秒**，
候选另计适配 **0.441 毫秒**；编译、运输及诊断辅助工作不混入 P。
冷中位少 **2.99%**，后续 public 中位少 **1.64%**；冷范围仍重叠，配对方向不全一致。
采用门槛仍为 `P候选+最慢冷前向 ≤ 0.9×(P原优化+最快冷前向)`：
候选 **1.051506977 秒**，要求不超过 **0.932124344 秒**，未通过。
事先声明的组合条件另外要求冷／热均呈不重叠改善：冷侧 **1.051507 > 1.035694 秒**，不通过；
热侧 **0.984640 < 0.986370 秒**通过，但合取条件失败，因此**没有构建或运行 3×4 组合版**。

冷前向从创建 QEMU 前计至完成帧校验；内部计时用设备树确认的 10 MHz 虚拟时钟，不是 CPU 周期。
每进程后四轮相关，不能当作十二份独立样本；NaN 准备另计，独立输出槽和外围计时会影响状态。
首次不总比后续慢，host／guest 时间残差不是纯启动或即时翻译成本，也不能把静态指令减少换算为提速。
资源每 0.5 秒采样，共98份、无错误，末端空隙约7.8–500.0毫秒；不能排除宿主调度或推断逐轮 CPU 成本。

性能批次 **14个进程、38次Q调用通过**，另1次设备树导出，流程 **40.997613秒**。
输出全部有限且逐位等于本轮原优化；严格 ORT 最大误差 **1.341104507446289e-7**（`<0.001`、`rtol=0`），
FP64最大误差 **8.85283508722523e-8**，预算占比最大约0.000861；输入／完整物理权重末态读回一致。
另 **14次边界运行＋1项重复变换拒绝通过**，流程 **2.532747秒**，不用于性能判断：
12次普通图覆盖 K=1/2/3/127/128/129、奇数行与 N=3/5/7/9 的合法补齐列尾部，
最大ORT误差 **1.9073486328125e-6**、FP64最大误差 **1.421002257373516e-5**，预算占比约0.040646。
2次特殊浮点图只核对精确K128融合乘加参考位，不套用ORT／gamma门槛；输出保护和输入／权重末态检查通过。
合计 **28个模型进程、52个主图输出**，退出码均0、事件等待无回退；末态一致不证明运行中从未暂时写入。

完整材料保存在本地 `output/issue109-q-projection-unroll2/`：
`preflight/machine-review.json`、`measurements/compiled-machine-control.json`、`measurements/report.json`、
`boundary/report.json` 保存机器码、性能和边界证据；源码、输入／权重／参考／输出、ELF、反汇编与日志均保留。
[验收摘要](external-matmul-evidence.json) 提供可分享统计。未采用候选，生产实现、poll、旧配置和历史数据保持不变；
未运行完整模型，未新增归档、提交、推送或提PR。

## Q 投影两步展开的尾部与地址开销

**精简版数值通过，冷／热中位耗时有所下降，但计入准备成本仍未达到至少快 10% 的采用门槛。**
本轮保留原优化 2×4、旧两步展开作为同期对照，只精简展开产生的多余控制路径。
仍为真实首层 Q 权重和代表激活 [1,256,1024] @ [1024,2048]，
RV64GC / LP64D、4 hart、CPU 亲和集合 0–3、严格浮点及当前 poll，原权重布局和 K128 累加顺序不变。

机器码分析表明，旧展开的外围净增约 **29 条／2×4任务**，仅占展开任务指令约 **0.174%**，
包括每任务地址准备和每 K128 块的计数、尾检查；不能单独解释此前指令减少却仅小幅提速。
旧版74字节尾体在 K1024 下不执行，但其指针准备仍保留；栈帧增大来自多保存三个寄存器。
范围函数每 hart 只调用一次，整个 Q 投影多出 **24 次栈访问、192字节流量**，不是每个矩阵块都付出该成本。
主环两步 K 的32条指令中仍有 **12次加载、16次融合乘加**；静态条数不是耗时占比，尚不能确定缓存或即时翻译瓶颈。

精简版利用已知 K 的奇偶性：任意固定偶数 K 不生成单步尾部，奇数 K 先算成对区间再执行至多一步；
只替换原串行／并行 K 循环，原累加器、每128步的块加法、任务划分、行列保护及公共检查／拷贝均保留。
预审确认偶 K 尾检查、尾体及相关指针准备消失：函数 **620→484字节**，栈帧 **128→96字节**；
主环仍为32条、没有热循环栈访问或函数调用，原实现为422字节／96字节。
按固定形状执行路径推算，每任务原版／旧展开／精简版为 **18703／16684／16655条**；这不是运行周期测量。
六份最终冷／重复制品的完整热函数字节均与各自预审相同后才运行。

每方一次冷预热、三次正式冷运行；重复诊断每方三个独立 QEMU，每进程五次。
两阶段正式顺序均为原版／旧展开／精简版轮转，三方各占一次首、中、末；下表单位为秒：

| 范围 | 原优化 2×4 | 旧两步展开 | 精简两步展开 |
| --- | ---: | ---: | ---: |
| 准备 P，仅一次观察 | 0.034505 | 0.034921 | 0.034929 |
| 冷启动前向中位 | 1.078203 | 1.032701 | 1.006975 |
| 冷启动正式范围 | 0.999359–1.177280 | 1.030853–1.058711 | 0.995447–1.056147 |
| 同 guest 首次 public，三个进程中位 | 1.023589 | 0.986693 | 1.001127 |
| 后四轮 public，先各进程取中位再汇总 | 1.025738 | 1.009164 | 0.983200 |
| 上项三个进程中位的范围 | 1.014235–1.034361 | 0.971370–1.029448 | 0.961706–0.988608 |
| 后四轮计算与同步 dispatch，同样汇总 | 0.999021 | 0.986094 | 0.959827 |

共用 P 为生成／检查11.886毫秒、权重序列化／检查／落盘22.618毫秒；
旧展开和精简版适配另计0.416／0.424毫秒，编译、运输及诊断辅助工作单列，不混入 P。
精简版相对原版冷／热中位少 **6.61%／4.15%**，相对旧展开少 **2.49%／2.57%**。
冷范围仍重叠；精简版热范围与原版分离、与旧展开重叠，不能据此声称长期稳定收益或把差值全部归于尾检查。
采用仍要求 P精简+最慢冷前向 ≤ 0.9×(P原优化+最快冷前向)：
本轮 **1.091075696秒 > 0.930477695秒**，因此不采用，也不追加3×4组合或完整模型。
冷前向从 QEMU 创建前到完成帧校验；内部10MHz虚拟时钟不是CPU周期，后四轮也不是十二份独立样本。
独立输出槽、每轮准备和计时会影响状态；153份0.5秒资源样本无采样错误，末端空隙约4.0–498.0毫秒。
后续计算与同步占public时间约97.6%，尚不能拆分加载／乘加耗时，也不能确定缓存、即时翻译或宿主调度的各自影响。

性能批次 **21进程、57次Q调用通过**，另1次设备树导出，流程 **63.356027秒**。
全部输出有限且逐位等于同期原实现；严格ORT最大误差 **1.341104507446289e-7**（<0.001、rtol=0），
FP64最大误差 **8.85283508722523e-8**，输入／完整物理权重末态一致。
另 **16次边界运行＋1项重复变换拒绝通过**，流程 **2.714726秒**，不作为性能数据：
14次普通输出覆盖K=1/2/3/127/128/129/130及奇数行列，K130专门覆盖最后一个K128块只有两步的偶K路径；
最大ORT误差 **3.814697265625e-6**、FP64最大误差 **1.421002257373516e-5**，预算占比约0.0500813。
2次特殊浮点输出只按精确K128融合乘加位参考检查；候选均与原版逐位相同，末态和输出保护检查通过。
合计 **37个模型进程、73个主图输出**，退出码均0、事件等待无回退；末态一致不证明运行中从未暂时写入。

完整资料保存在本地 output/issue109-q-projection-unroll2-tight/：
analysis/overhead.json、preflight/machine-review.json、measurements/report.json、boundary/report.json，
以及源码、机器码、ELF、输入／权重／参考／输出、原始时钟和资源日志。[验收摘要](external-matmul-evidence.json)记录统计。
仅保留实验候选；生产实现、poll、旧配置、旧实验及归档不变，未重跑完整模型，未提交、推送或提PR。

## 仍未承诺的能力

真实硬件、其他库版本或编译配置、向量优化、七用例覆盖，
以及更广环境和长期重复性能结论，仍需要各自的运行证据。
当前只完成这一固定模型、机器和配置下的三组完整 select / poll 对照。
NMSIS 按行并行已通过代表矩阵和混合小图，但该配置的完整模型仍未验证。
其他外部库是否可安全按行并行，需要单独确认，不能直接沿用 NMSIS 的结论。
显式实测配置已能按节点选择；自动采集测量、跨环境复用、预测未测形状及全图最优选择仍未实现。
动态尺寸、其他数值格式和其他算子也不会因 MatMul 接入而自动获得支持。
真实模型 9 类形状与布局已补测，其中 NMSIS 末层由接口上限排除。
更多输入和线程配置变化仍需要对应证据，不承诺外部库必然更快。
