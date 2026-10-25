# W4 分层验证与 CI

## 日常最小验证与多 hart

先按改动选择直接相关的测试；文档和配置改动只做对应检查。矩阵生成或并行运行时
改动可先运行以下小图回归（按[复现步骤](reproduction.md)设置 Python、编译器和 QEMU）：

```bash
python -B -X utf8 -m pytest -q \
  tests/test_tensor_c_external.py \
  tests/test_riscv_external_harts.py tests/test_w4_fma_cross_backend.py
```

外置运行时 `build_external(..., harts=1)`、完整探测及检查点诊断入口的
`--harts {1,2,4,auto}` 均默认单核，手动 1/2/4 保持原值。`--harts auto`
（或 `build_external(..., harts="auto")`）在构建前按当前进程可用 CPU 数向下选择：
1 个选 1 hart，2–3 个选 2 hart，4 个及以上选 4 hart。Linux 优先使用 CPU affinity；
不可用时回退进程/系统 CPU 数，无法确定时保守选 1。此口径不估计瞬时空闲率或
cgroup CPU 带宽配额，也不受 `OMP_NUM_THREADS` 等 BLAS 线程设置影响。

选择只发生在启动/构建时，运行中不切换；CLI 会打印自动选择结果，报告用
`harts_requested` 记录请求、`harts` 记录实际整数。当前数学策略的默认值仍是
`sequential`；需要矩阵并行时必须同时显式使用 `--matmul-policy blocked_fma`。
MatMul 的多核路径只用于 `blocked_fma` 中足够大的矩阵，按独立输出块分工；
每个元素的 K 顺序和 K=128 分组保持不变，算子结束后才复用工作区。
`sequential` 下的 MatMul 仍串行。每个 hart 独立使用 1 MiB 栈。

外置运行器在 Linux 上用 inotify 与 pidfd 等待已有 UART 完成帧，启动前注册目录监视，
不再固定每 50ms 轮询；平台或内核不支持时保留轮询回退。`run.json` 的 `timing_mode`
记录实际等待方式，`completion_wait_fallback_reason` 说明回退原因。`forward_wall_s`
仍表示 QEMU 启动至宿主校验完整完成帧，包含装载与通知开销，不是纯算子执行时间。
QMP 服务端连接启用 `nodelay=on`，减少 STOP 事件与命令回复之间的小包等待；仍等待
匹配回复，并在停止后复核 UART 帧，再转储输出、检查退出和清理资源。

外置 Softmax 在实际 hart 数大于 1、总元素数至少为 32768，且独立行数不少于
hart 数时可并行；它不依赖 `matmul_policy`，因此显式选择多 hart 后，即使 MatMul
策略为 `sequential`，符合条件的 Softmax 也会并行。默认仍为单 hart、`sequential`。
独立行对应固定的 `(outer, inner)` 坐标，每行完整保留原有求最大值、逐项 `expf`
与求和、逐项归一化顺序，不拆分行内归约。输出有限性检查紧跟每个 FP32 归一化值
的写回，不再由主 hart 在算子结束后重复扫描整个输出；检查本身及输入、权重、其它
算子的校验仍保留。工作线程通过原子状态回传数值错误，`sv_parallel_for` 等全部回调
完成后，主 hart 检查错误状态，再进行后续工作区复用。inline 存储、单 hart、小图
及独立行数不足时仍走串行路径，并在同样的逐行写回位置检查输出。
32768 元素只是保守的启用门槛，尚未校准为各形状的最佳阈值；继续复用已有门铃/WFI
同步和 `expf` 页对齐。此代码生成改动需要重新构建 ELF，复用旧 ELF 不会加入
Softmax 并行或写回时的有限性检查融合。

同一次生成中，输入形状、dtype 和规范化 axis 相同的外置并行 Softmax 复用回调代码。
只共享函数与结构体定义；每次算子调用仍使用独立的输入、输出指针和初始化为零的
错误状态，完成同步后才检查状态和复用工作区。此复用不改变单核默认、手动/自动
hart 选择、并行门槛或行内数值顺序；也不代表真实硬件缓存或完整模型的等比例收益。

代码生成 API 可显式比较 `TensorCCodegen(..., constant_storage="external",
matmul_policy="blocked_fma", matmul_tile="2x4")` 或 `matmul_tile="1x8"`。
`matmul_tile` 默认仍为 `"1x4"`；候选仅适用于外置 `blocked_fma`，不改变 hart 的
默认、手动或自动选择方式。`2x4` 复用两行之间的 B 加载，`1x8` 复用八列之间的
A 加载；每个输出仍按原 K 顺序执行显式 FP32 FMA，保留 128 项分组和块末加法。
尾行、尾列有边界保护，不使用 RVV 或 fast-math；面板布局由下述独立实验选项控制。
完整探测与检查点 CLI 可使用 `--matmul-tile {1x4,2x4,1x8}`；选择 `2x4` 或 `1x8`
必须同时显式传入 `--matmul-policy blocked_fma`，不匹配时在构建前报错。报告与构建证据
记录实际 tile；`--reuse-build` 同时核对旧报告和 ELF 构建证据，旧记录缺少该字段仅按
`1x4` 兼容。选择变化需要重新构建，不会自动挑选最快内核。应先用同条件小图比较输出
位模式、耗时、累计 CPU 时间和峰值内存，不能把短图收益直接当作完整模型或真实硬件收益。

外置 `blocked_fma` 可显式选择 `--matmul-rhs-layout column-panel`，API 对应
`matmul_rhs_layout="column-panel"`；默认 `original` 保留原布局。该选项只打包二维
常量 RHS，允许 LHS 带批维；每个面板的列宽沿用 `matmul_tile`，物理布局为
`[ceil(N/NR), K, NR]`。它复用相邻列的权重访存，仍按递增 K 顺序执行 FMA，保留
128 项分组与块末加法，不改变默认单核或手动、自动 hart 选择。

只有常量的全部消费者都把它用作 MatMul RHS 时才选择面板。独占常量经过合法二维
转置后仅供 MatMul RHS 使用时，在局部复制的 IR 容器中折叠该转置，交由原有生命周期
规划分配工作区；原始 IR 和绑定不修改。共享给 Gather 等其他算子的权重、动态 RHS、
批量 RHS 和空张量保持原布局；报告列出实际选中的权重，不能把请求选项当作全图覆盖。

列数整除面板宽度时，使用非连续数组视图，由现有权重包写入器分块序列化，避免复制
完整权重数组。尾面板需要补零，分配前按物理字节数检查容量；尾列可能增加权重容量。
这与 `qemu_workspace_layout` 的 4KiB 工作区间隔选项相互独立，后者仍默认关闭。
所有物理权重（含补零区域）的有限性和地址重叠检查仍保留。打包的权重包必须与对应
生成代码/ELF 配对，不能直接作为原布局权重或其他部署后端的权重使用。

`matmul_rhs_layout` 与 `matmul_rhs_panels` 在报告和构建证据中记录逻辑形状、物理形状、
原权重、折叠的转置及视图/复制方式；复用同时校验布局和面板记录。旧记录缺字段只兼容
`original`，面板构建必须重新生成；选择相同选项不意味着可以绕过已有源码、权重和
构建身份校验。小图收益需与同 hart、同微内核、同计时方式的原布局比较，常量转置链
还包含省去转置复制的收益，不能把总差值全部归为缓存效果。

外置 ELF 构建将频繁调用的 `expf` 代码段起始地址按 4096 字节页对齐，避免当前
实现因生成图代码长度变化而跨页、引入额外的 QEMU TCG 开销。这里只调整链接布局，
不改变 `expf` 的数值实现、默认单核或手动/自动 hart 选择。应用此修复必须重新构建
ELF；`--reuse-build` 复用旧 ELF 不会应用新布局。收益应通过相同条件的小图对照确认，
不能直接外推到完整模型。

QEMU 两缓冲工作区可显式尝试实验选项
`TensorCCodegen(..., constant_storage="external", qemu_workspace_layout="two-buffer-pad-4k")`。
完整探测及检查点 CLI 对应 `--qemu-workspace-layout {none,two-buffer-pad-4k}`，
默认 `none` 完整保留原布局；该选项不改变默认单核或手动、自动 hart 选择。
仅当原工作区的所有分配可归入恰好两个固定、相邻、互不重叠的区间时，实验布局才在
两者之间插入 4096 字节，工作区需求也增加 4096 字节。其他图或容量不足会明确报错，
不会静默忽略选项；完整 Qwen 图和检查点不保证满足两缓冲条件。

这里的 QEMU 专用表示针对当前 QEMU SoftMMU 的软件地址转换行为验证过的布局实验，
不是修改 QEMU，也不是启用新的 RISC-V 指令。小图中，原间距 1 MiB 增加 4 KiB 有收益，
但原间距 1 MiB−4 KiB 增加到 1 MiB 出现退化；因此不按张量大小自动启用，也不承诺
不同 QEMU 版本、形状或真实硬件具有相同收益。此策略偏专项，数值顺序和算子语义保持
不变；默认关闭时真实 RISC-V/NPU 部署路径不变。若主动携带该布局到其他目标，额外容量
与地址变化仍存在，应独立评估，不能把它当作 NPU 内核、硬件缓存或通用内存布局优化。

以下只是参数示例，不是要求运行完整模型：
`--qemu-workspace-layout two-buffer-pad-4k --harts 4`。
先用相同输入的 1、4 hart 小图检查位级输出一致性、容量不足处理、耗时、累计 CPU 时间、
RSS 和 SoftMMU 样本，覆盖收益与退化形状。报告的 `qemu_workspace_layout_requested`
保留请求，`codegen.qemu_workspace_layout` 和构建证据记录所用布局；构建证据的
`qemu_workspace_buffers` 记录缓冲区位置。修改布局须重新生成、构建 ELF；复用时请求
必须与 ELF 的布局证据一致，旧证据缺少该字段仅按 `none` 兼容。

多核任务、就绪和完成等待使用 QEMU virt 的 MSIP 软件中断门铃与 WFI。
等待端先清门铃，再检查任务代次、完成状态和错误；发送端先发布状态再通知，
避免通知先到时丢失唤醒。空闲 worker 和等待完成的主 hart 可休眠，默认单核路径
不启用此机制。启动阶段极短的 BSS 就绪轮询保留；其他板型需适配中断控制器。

应同时比较同一候选、相同小图的耗时、累计 CPU 时间和输出位模式。
CPU 占用下降不等于等比例加速；短时结果不代表完整模型收益，也不代表真实
RISC-V 硬件的周期或缓存性能。

共享代码生成的数值问题先用现有 Host C 测试定位；需要验证真实 guest 行为时用
QEMU 小图，必要时再用 `diagnostic --checkpoint layer_0.output`。检查点的
`COMPLETE` 仅表示执行完成，还须查看 `diagnostic_threshold_passed`。完整模型与
原数组 audit 留到候选稳定、明确需要完整验收时运行，不作为每次编辑的默认步骤。

同一候选重复执行可复用已有 `--reuse-build`；改变源码、数学策略或 hart 数必须重新
构建。`auto` 也必须与当前探测后选择的数量一致，复用数量不符时明确报错；
不会修改旧 ELF 的 hart 数。旧的未记录 hart 数的构建按单核处理，不能直接以多核运行。

## 按风险阅读实现和测试

| 范围 | 实现 | 重点测试 |
|---|---|---|
| 权重身份和布局 | `scratchv/runtime/weight_bundle.py` | `tests/test_weight_bundle.py` |
| 外置 ABI 与数学策略 | `scratchv/backend/tensor_c_codegen.py` | `test_tensor_c_external.py`、`test_w4_contract.py`、`test_w4_fma_cross_backend.py` |
| 裸机运行、ELF 与进程清理 | `scratchv/runtime/riscv_external.py`、`riscv_tensor.py` | `test_riscv_external*.py`、`test_riscv_process_cleanup.py`、`test_w4_large_address.py` |
| 分片与故障证据 | `scratchv/runtime/weight_transport.py` | `tests/test_riscv_weight_transport.py` |
| 完整探测、诊断与 audit | `probes/w4_qwen3_full/` | `test_w4_full_probe*.py`、`test_w4_diagnostic.py`、`test_w4_audit*.py`、`test_w4_host_diagnostic.py` |
| CI 条件及失败关闭 | `.github/workflows/w4-*.yml` | `tests/test_w4_workflows.py` |

测试文件的完整路径均在 `tests/` 下。完整模型数值由探测入口实际执行，不是单测中人工
小图或伪造的报告 fixture。执行以下回归前，先按[复现步骤](reproduction.md)的 Linux 环境设置激活 Python 环境并设置 `SCRATCHV_CC`、`SCRATCHV_QEMU`；部分真实工具用例依赖这些变量，未设置时可能被跳过。下列较宽回归清单用于跨模块检查，不是每次编辑的默认范围；完整模型另按复现步骤运行：

```bash
python -B -X utf8 -m pytest -q \
  tests/test_weight_bundle.py tests/test_tensor_c_external.py \
  tests/test_riscv_external*.py tests/test_riscv_process_cleanup.py \
  tests/test_riscv_weight_transport.py tests/test_w4_*.py \
  tests/test_tensor_boundary_contracts.py tests/test_tensor_return_contract.py \
  tests/test_w1_qwen3_export.py tests/test_llm_inputs.py \
  tests/test_w2_runtime_metadata.py tests/test_w3_full_gate.py \
  tests/test_licm_safety.py tests/test_optimizer_numeric_semantics.py
```

真实 QEMU、FMA、跨 4 GiB 地址、跨后端及跨 FP32 元素边界的分片测试应实际执行。
工具缺失导致的 skip 不能作为这些覆盖已完成的证据；平台特有信号/session 与文件权限
测试应在适用环境补测。

## 门槛不能互相替代

W1/W2 小模型 IR 及算子门槛按各自定义保留；W3 完整 IR 为严格 `max_abs < 1e-4`，
W4 完整 QEMU 为严格 `max_abs < 1e-3`，均保持各自报告的输入范围。W4 对全部
38,895,616 个 FP32 logits 比较，`rtol=0`，不接受 NaN/Inf、shape/dtype 错误、缺失输出
或等于阈值。FMA 指令语义使用精确位/FCSR 判断，不使用模型容差。

离线 audit 同时核对用例名称对应的固定 token、mask 和有效长度，不能仅更新文件哈希
就把另一组输入计入该用例。W3 独立探测和 W4 完整入口在成功前复核执行源码指纹；
运行中源码变动或无法读取时拒绝发布成功。W4 仅构建入口也执行该复核。

## Workflow

普通 PR 的 W4 检查归入统一 `CI` 运行，不再单独触发一笔 W4 Actions。
`w4-full-numeric.yml` 仅提供 `workflow_call`，其 unit job 保留合同、小图真实目标及
跨阶段低成本回归。完整 job 依赖 unit 成功，且调用者必须明确传入 `run_full=true`
才执行单个固定 case；该参数默认关闭，调用者的 push、PR 或定时事件本身不会启用完整模型。
完整 job 依次要求 build、实际工具 FMA、QEMU 完成、全 logits 数值和原始数组 audit，
并检查报告与请求 case、提交和工具身份一致。失败、取消、跳过不能替代成功。

需要运行完整模型时，从 `LLM Deploy v1.0`
（`llm-deploy.yml`）选择候选分支，手动设置 `run_w4_validation=true`，并选择
`w4_case`（默认 `full_seed_0`）。此入口以 `run_full=true` 调用同一提交的 `w4-full-numeric.yml`；
W3 的 `run_w3_validation` 可独立选择，两者默认均关闭，不改变普通 PR 或定时任务的范围。
W4 调用未请求时显示 `NOT RUN`；明确请求后失败、取消或跳过会令 W4 汇总失败。

Actions Summary 直接显示本次单测的通过/失败/错误/跳过数。完整任务实际执行时，
另显示请求用例、MaxAbs、RMSE、全量比较及超限元素数、不含权重 ELF 大小和 QEMU
墙钟；门禁、阶段耗时及错误摘要折叠展示。详细机器报告和原始数组仍从 Artifacts 下载。
未执行的完整任务明确显示 `NOT RUN`，报告缺失显示 `MISSING`，不会引用历史运行的
数值填补。本次摘要只读取产物，不改变数值计算、门槛或 Nightly 启用策略。

`w4-nightly.yml` 通过仓库内 `uses` 调用同一提交的完整 workflow，默认仅运行
`full_seed_0`。**所有 Nightly 入口均需仓库变量 `W4_NIGHTLY_ENABLED=true`**，否则不执行
昂贵任务；上述 `LLM Deploy v1.0` 手动入口不受此变量限制。定时配置存在或本地测试
通过，不等于已经启用定时执行或取得远端成功记录。汇总成功也不代表团队已验收。

W4 不自动重跑昂贵的 W3 完整七例。发布评审应另外核对同一候选源码的 W3 证据；变更
共享解析、IR、解释器或数值策略时，应补对应层级回归，不能只用 W4 较宽门槛覆盖 W3。

完整 QEMU 截止为 10800 秒，CI 的完整 runner 外层软截止 195 分钟、30 秒强制回收宽限，
step 200 分钟、job 240 分钟。正常失败上传已有报告与日志；宿主丢失、强制取消或整体
超时可能阻止上传，产物缺失按未完成处理。

## 证据与保留

- 报告包保留源码快照、摘要、工具/依赖、构建源、布局、清单和阶段错误；原始数值包
  另外保留输入、ORT、QEMU output、UART 完成帧及 FMA 观测。
- CI 报告包保留 30 天、原始数值包 3 天。需要独立审计时，在到期前保存原始数组；
  只下载摘要不足以重算误差。
- 不上传大权重 bundle 或运输副本；独立复现按固定清单重新获取模型。
- `source-snapshot.zip` 仅覆盖报告记录的源码和固定配置，需结合其 Git 基线恢复，
  不是完整独立仓库归档。
- 原始报告、工具下载和运行产物保存在忽略的 `output/`，不随源码提交发布；验收材料
  应提供单独的产物获取位置和保留期限，公开摘要不能代替原始数组。

## 第二人回传结果

复现者可直接复制以下模板回传；各项填写实际结果，未执行写“未执行”，不要预填通过。
最终验收由团队另行确认。直接重算作者提供的数组属于离线 audit，独立复现应由复现者
在指定候选源码上重新构建并执行，不能仅复用作者的 ELF。

```text
复现者 / 日期：
源码：候选完整 SHA、工作树是否干净；若为未提交快照，附基线 SHA、源码清单和快照 SHA256：
环境：宿主 OS/架构、Python/依赖、Zig/QEMU 实际版本及二进制 SHA256（见主报告）：
执行：完整命令、case、MatMul 策略、hart 数（默认 1）；是否独立重新构建并执行 QEMU：
单测：通过/失败/跳过数，tests.xml 位置；每项相关跳过原因：
完整门槛：build:riscv-full / smoke:qemu-full / numeric:qemu-full-qwen3：
数值：elements_compared、max_abs、atol、rtol；FMA 精确检查和离线 audit 结果：
资源：ELF 字节数、guest 布局、qemu_wall_s、采样 QEMU RSS（不填成进程树峰值）：
证据：主 report.json、source-snapshot.zip、模型/权重摘要、FMA 报告、audit 报告、原始数组包位置及保留期限：
关联证据：同一候选的 W1/W2 回归与 W3 精度结果，未执行项明确列出：
失败或不足：stage/error、退出码/超时、最后日志、相关 run.json/transport.json；缺失文件及原因：
结论：本次实际覆盖范围和未完成项；不代替团队签字或 Nightly 记录：
```

字段以主报告为索引：`git`/`source_sha256`/`source_snapshot` 记录源码；`environment`、
`execution_tools`、`execution_tool_binary_sha256` 记录环境；`model_files` 和 `weight_sha256`
记录模型与 bundle 身份。数值及运行状态在 `cases[*].numeric` 和 `cases[*].qemu`。
工具版本字符串相同仍需保留二进制摘要；不同平台的二进制摘要可以不同，分别通过实际工具的
FMA 预检并满足相同门槛，不要求跨平台 ELF 或全部 logits 逐位一致。

成功时保留整个探测输出目录以及独立的 audit 目录，分享时按上述 CI 保留规则去除大
权重和运输副本，保留相对目录结构。失败时先保留已生成的 `report.json`、`progress.json`、
控制台日志及 case 下的 `qemu/run.json` 和运输清单（如存在），再使用新目录重试。
没有生成原数组或主报告时如实注明，不能用一次重试的 PASS 覆盖首次失败。
