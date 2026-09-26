# Topic 18：CNN 指令调度验证

验证状态：**passed**。
输入：`models/graph/cnn.onnx`，15 个节点；SHA-256：`169586fb90fc4a2fe1a45e3c27ac95e9c03cec58616b6dcc3b7ab7b4c24fc264`。
主路径为 standalone CNN 编译器，固定启用常量合并；CompilerDriver 两种分配器仅作补充回归。
A/B 两侧使用相同版本的代码生成器，仅切换调度；生成器哈希保存在 JSON 中。
评测显式请求符号化汇编，并校验关闭调度时的二进制与默认输出一致；默认汇编格式保持原有行为。
CPU 模型：LLVM 18.1.3 / `sifive-e76`，RV32IMFD，禁用压缩指令。
调度候选与原序均由 llvm-mca 分析；仅在完成周期严格减少时采用候选。指令延迟、旁路和资源占用由 LLVM 提供。
配置：`-iterations=1 -noalias=false -timeline -timeline-max-cycles=0 -timeline-max-iterations=1`。模型定义链接、工具版本、输入和原始 LLVM JSON 均保存在报告 JSON 中。

## 指标与结果

| 指标 | 定义 |
|---|---|
| 最大指令并行数 | LLVM Timeline 中单周期发射指令数的峰值；跨区域取最大值 |
| 流水线气泡 | 发射区间内没有指令发射的周期数 B=S−U；S 为首末发射周期覆盖的长度，U 为有发射的周期数。排除尾部排空，不代表逐流水级气泡数 |
| 完成周期 | 最后执行完成与最后发射加一的较大值，减去首发射周期；排除 LLVM 最后一个管理周期，包含排空 |
| 汇总 | 周期和气泡跨区域求和；气泡率为 ΣB/ΣS，IPC 为建模指令数/完成周期合计 |

表中数值均为调度前→后。

| 路径 | 建模/输入指令 | 换序区域 | 峰值并行数 | 周期合计 | 气泡周期 | 气泡率 | IPC |
|---|---:|---:|---:|---:|---:|---:|---:|
| standalone/const-merge | 877/878 | 29 | 2→2 | 1116→927 | 230→151 | 25.1%→20.8% | 0.786→0.946 |
| CompilerDriver/greedy | 79/88 | 0 | 2→2 | 125→125 | 28→28 | 30.1%→30.1% | 0.632→0.632 |
| CompilerDriver/linear | 51/60 | 0 | 2→2 | 92→92 | 4→4 | 7.7%→7.7% | 0.554→0.554 |

主路径局部周期合计减少 189（16.94%），气泡减少 79 个周期。

## 胜/平/负区域

对照对象：同一编译路径、同一静态区域，A 为关闭调度的原始顺序，B 为最终输出顺序；两侧使用相同 LLVM 配置。不跨分配器比较。研究目标是判断换序能否减少该 CPU 模型下的局部等待。
区域由标签、控制转移、未知指令和其他汇编文本边界划分，可能只是基本块的一部分；区域内终止指令固定。
胜：T_B < T_A；平：T_B = T_A；负：T_B > T_A。T 为完成周期，与峰值并行数或气泡是否变化无关。每个成功分析的静态区域计一次，包含未换序区域；不按循环次数或收益大小加权。
无收益和退化候选恢复原序，因此最终结果的负区域应为零；JSON 保留候选周期，可审查被拒绝的候选。工具失败会使 benchmark 失败，不视为持平。

| 路径 | 对照区域数 | 胜/平/负区域 |
|---|---:|---:|
| standalone/const-merge | 100 | 29/71/0 |
| CompilerDriver/greedy | 22 | 0/22/0 |
| CompilerDriver/linear | 21 | 0/21/0 |

## 副作用与执行

| 路径 | 物理寄存器数 | 栈加载/存储 | CFG 活跃峰值 | 结构检查 | 完整执行 |
|---|---:|---:|---:|---|---|
| standalone/const-merge | 30→30 | 0/0→0/0 | 24→25 | passed | passed |
| CompilerDriver/greedy | 14→14 | 16/24→16/24 | N/A | passed | not_run |
| CompilerDriver/linear | 22→22 | 0/12→0/12 | N/A | passed | not_run |

结构检查包括指令/寄存器集合、读值来源、最终写入、访存顺序、控制流边界及 sp 操作。
CFG 活跃性用固定点求解循环，出口契约为内存输出、sp 保留、ret 读取 ra；含未知语义的补充路径标为 N/A。
调度发生在寄存器分配之后，没有再次分配；栈访存包含保存/恢复，不能直接当作精确 spill 数。
完整 standalone CNN 使用 QEMU 执行 3 组固定 Q16.16 输入，比较全部整数寄存器、工作区及输出：passed。
同时验证汇编重新编码与二进制一致、权重字节不变及缓冲区边界。执行结果在这些输入下前后一致；该检查不验证定点 CNN 与 ONNX 浮点参考的数值精度。

| 输入种子 | 调度前输出（Q16.16 原始整数） | 调度后输出 | 全部状态一致 |
|---|---|---|---|
| 0 | [27990] | [27990] | True |
| 1 | [29997] | [29997] | True |
| 7 | [29999] | [29999] | True |

## 覆盖与限制

覆盖率 = 纳入模型的源汇编指令条数 / 输入源汇编指令条数。标签和注释不计数，未展开的伪指令按一条计数；覆盖率不表示动态执行占比或发生换序的比例。
未建模指令仍保留在输出中，但不计入周期、峰值并行数和气泡指标；未估算的开销不视为零周期，指令形式是否可合法编码需另行验证。

| 路径 | 建模/输入指令 | 覆盖率 | 未建模指令及条数 |
|---|---:|---:|---|
| standalone/const-merge | 877/878 | 99.9% | `auipc` × 1 |
| CompilerDriver/greedy | 79/88 | 89.8% | `max` × 4、`slt` × 5 |
| CompilerDriver/linear | 51/60 | 85.0% | `max` × 4、`slt` × 5 |

未建模原因与处理：

- **AUIPC：地址相关，固定原位。** 它计算“自身指令地址 + 立即数偏移”，主路径用它建立数据基地址。移动指令而不重算偏移会改变结果；当前不实现该地址重算，也不对 AUIPC 估算周期。活跃性分析仍可识别其寄存器写入。
- **max：尚未展开的项目伪指令。** 当前 RV32IM 路径中的 `max rd, rs, 0` 会展开成比较、分支和复制等多条指令。调度器在展开前处理文本，尚未对该展开序列建模，不能直接赋予一条普通指令的周期。
- **slt：寄存器操作数形式无效。** CompilerDriver/greedy 第 17 行：`slt t1, 0, t0`；CompilerDriver/greedy 第 42 行：`slt t1, 0, t0`；CompilerDriver/greedy 第 60 行：`slt t1, 0, t0`；CompilerDriver/greedy 第 83 行：`slt t0, t1, 1`；CompilerDriver/greedy 第 95 行：`slt t1, 0, t0`；CompilerDriver/linear 第 21 行：`slt s7, 0, s6`；CompilerDriver/linear 第 32 行：`slt s7, 0, s10`；CompilerDriver/linear 第 43 行：`slt s7, 0, t0`；CompilerDriver/linear 第 58 行：`slt s8, t4, 1`；CompilerDriver/linear 第 65 行：`slt s10, 0, t4`。`slt` 的两个输入必须是寄存器：零寄存器应写为 `zero/x0`，寄存器与立即数比较需要 `slti` 等合法形式。调度器保留原有生成结果，将这些指令作为边界；公共指令选择的修复需独立处理。

未建模指令作为调度边界：自身固定，前后指令分别在各自区域内分析，禁止跨越该边界移动。其他未建模原因及所在行见 JSON 的 `scheduling.diagnostics`；CFG 活跃性遇到未知寄存器语义时标为 N/A，不能解释为零活跃寄存器。

测试对象与结论范围：

- **CompilerDriver 是补充编译回归。** 它将 Conv/Gemm 等算子简化为少量乘加，没有展开完整张量计算；与 standalone 的静态指令数差异不能归因于调度。完整执行列的 `not_run` 表示未执行机器码做结果验证；静态结构检查通过不等于完整 CNN 推理正确。
- **静态区域只计一次。** 完整 standalone 代码也包含循环；例如 4 条循环体指令执行 1000 次，静态仍是 4 条，动态执行量为 4000 次。报告不按循环次数或分支频率加权，各区域从周期 0、入口操作数就绪开始独立估算，没有贯通跨区域的动态等待。
- **性能结论限于模型。** llvm-mca 使用 SiFive7 的资源约束；报告未测量真实硬件周期、缓存未命中或分支预测开销，因此局部周期合计下降不能直接换算为整网运行时间下降。
- **执行验证覆盖已测输入。** QEMU 用于比较调度前后执行状态，不用于测量硬件加速；固定输入上的一致性不等于所有输入的证明，也不验证定点结果与 ONNX 浮点参考的精度。

## 关闭调度的主分支对照

以 `11a2c3c` 为独立基线，比较 46 个 DSL 与仓库 CNN、三种分配器、DAG 开关、none/all 优化，共 564 个 CompilerDriver 配置；另比较 standalone 常量合并开关的两组默认输出。

566 组结果全部一致，涵盖汇编、可编码机器码、寄存器映射及 spill/reload。24 组标量用例的 TinyFive 执行结果与计数一致，且全部符合独立预期。基线已有 272 组编译或编码失败，同样记录并对照；一致不表示这些路径正确。完整数据由 `benchmarks.compare_schedule_disabled` 生成在 `benchmark_reports/schedule_disabled.json`，摘要为同名 Markdown。

同一对照工具在范围收缩前的 `26295bf` 上检测到 300 组差异，并检出 `loop_add_4` 的 load 10→15、store 6→7；当前工作版本差异为零。该验证使用独立版本，预期结果不由当前代码生成器自行构造。

## 本地集成验证

- Python 3.12.13：LLVM 18.1.3 与 22.1.8 下，CI 主测试集各通过 1294 项，无失败或跳过；CI 单独选取的模拟器测试另通过 7 项。
- 本地执行现有 benchmark Job 的 17 个业务步骤，退出码全部为 0。已有 LLVM/TinyFive 对比步骤提示使用静态 fallback，该报告不能当作完整实测；Topic 18 的 LLVM 与 QEMU 检查没有使用此 fallback。
- Python 3.11.16：Topic 06 集成契约 12 项通过，activation/elementwise/loop 门禁共 12 个用例通过；此前失败的三个循环用例已恢复。
- Topic 06 全量诊断为 15 个通过、8 个数组输入 ABI 不支持，与独立主分支对照一致。此诊断不要求所有未支持用例通过，不能写成 23 项全绿。
- LLVM 22.1.8 下，standalone 局部完成周期为 1114→921，三组完整 CNN 执行状态一致。上文主表仍使用 LLVM 18.1.3。

测试集由此前的 1308 项变为 1294 项：撤出依赖通用后端修复的 18 个参数化用例，新增 4 个非标准操作数保持原位的回归。原测试与修改保存在本地备份中。公共后端修复不再作为调度器测试的隐含前提；原有主分支测试及 Topic 06 门禁没有删减或放宽。

本地验证执行了相应测试与 benchmark 命令，不包括 GitHub Runner 部署、上传 artifact 或 Pages 发布。

## 复现

```bash
# 精确复现主表时确认实际版本为 18.1.3。
export LLVM_MCA=llvm-mca-18
"$LLVM_MCA" --version
PYTHONHASHSEED=0 python -m benchmarks.bench_cnn_schedule
PYTHONHASHSEED=0 python -m benchmarks.compare_schedule_disabled --baseline-root /path/to/base-checkout
PYTHONHASHSEED=0 SCRATCHV_REQUIRE_RISCV_EXECUTION=1 python -m pytest \
  tests/ -q --ignore=tests/test_simulator.py
python -m pytest -q tests/test_simulator.py::TestStubProfiledMachine \
  tests/test_simulator.py::TestRealProfiledMachine::test_lw_preserves_all_four_bytes
python scripts/run_topic06_benchmarks.py --category activation --fail-on-test-failure
python scripts/run_topic06_benchmarks.py --category elementwise --fail-on-test-failure
python scripts/run_topic06_benchmarks.py --category loop --fail-on-test-failure
```

实现与工具配置见[代码说明](18-指令调度器代码说明.md)，逐区域 LLVM 输出及执行证据见[原始数据](cnn-scheduling-results.json)。生成报告位于 `benchmark_reports/inst_scheduler_cnn.md`；汇编、二进制及默认输出保存在 `benchmark_reports/cnn_schedule/`。
