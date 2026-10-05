# W3 Linux CI 与 Nightly

配置已随 [PR #96](https://github.com/ScratchV-Compiler/ScratchV/pull/96) 提交。2026-10-05，作者在提交 `1c49e8acbcd9174491b2d0c2a5a0072178ae5f9a` 的[第四轮 Linux 手动编排](https://github.com/yuki-328/ScratchV/actions/runs/37299235103)中完成同提交 preparation、完整七组 full-numeric 和汇总，均为 success。preparation 下载已复核；完整 reports/raw 均已下载验真，七组原始数组离线 audit 通过。前三轮失败与历史数值 profile 原样保留。下面区分实际运行记录与通用命令模板；没有设置 Nightly 远端开关，手动调用不等于 schedule，也不替代他人独立复现。团队确认内容见 [独立复现与验收清单](独立复现与验收清单.md)。

## 2026-10-05 Linux 实测记录

### 第一轮：preparation 通过，完整模型失败

运行：[37290898463 / attempt 1](https://github.com/yuki-328/ScratchV/actions/runs/37290898463)，仓库 `yuki-328/ScratchV`，提交 `2e5f9f4817812a43bd9587c8f79fab43e068b1d9`。触发方式为作者 `workflow_dispatch` → `run_w3_validation=true` → W3 reusable 编排；没有更改默认分支或开启定时任务。该记录证明作者代码在 GitHub 托管 Linux runner 上实际执行，不能登记为 E1/E2/E5 三人的独立复现。

| 检查 | 实际结果 | 用时 |
|---|---|---:|
| preparation 回归 | 1203 passed、1 skipped；跳过项仅适用于 Windows Job ownership | 46.80 s |
| 六层 medium | 7/7 输入、81 检查点、18 不变量；各项对比最大绝对误差 `2.771615982055664e-6 < 1e-5` | 97.052790 s |
| 真实权重子图 | 14/14 输入；各项对比最大绝对误差 `3.0517578125e-5 < 1e-4` | 5.894554 s |
| 小 Attention 后端 | 12/12 执行；QEMU vs ORT 最大绝对误差 `3.5762786865234375e-7 < 1e-4` | 22.863963 s |
| 完整资产预检 | 固定资产与 7847 节点图通过；不执行完整模型数值 | 2.158714 s |
| layer-diff | 81/81 检查点；最大绝对误差 `2.205371856689453e-6 < 1e-5`，无首个分歧 | 包装进程 0.385691 s |
| preparation 汇总 | 五项 PASS；12 次 QEMU 进程时间合计 `0.952405885 s` | 总计 132.844285 s |
| 完整七组 full-numeric 首跑 | FAIL；`short_17` 输出误差超标，`changed_padding` 的父进程资源观察失败；其余五组通过 | 693.543067 s |
| 完整七组原始数组离线复核 | 首跑失败产物待下载复核；不能登记为完整数值 PASS | 待完成 |

完整模型首跑报告保留 `FAIL`：`short_17` 的普通/诊断 logits 最大绝对误差均为 `0.00013637542724609375`，超过严格 `<1e-4`；`changed_padding` 的 ORT 子进程返回 0，但父进程记录 `Cannot observe worker ... resident memory`，该 case 未被接受。其余五组数值通过不构成七组全覆盖通过。原始失败报告与源码指纹保持原样，后续修复必须以新提交、新运行记录重新验证；此处仅登记验收事实，不记录数值调查过程。

medium 与子图表中的最大误差覆盖各自报告内所有比较，不能当作单一 IR/ORT 指标。子入口用时不包含统一包装器启动与核验开销，QEMU 进程时间包含启动、UART 与退出。preparation 主报告依然明确 `full_ir_executed=false`、`w3_exit_accepted=false`。

preparation runner 为 Ubuntu 24.04.5、Python 3.12.14、4 vCPU AMD EPYC 7763；RAM `16,766,414,848` bytes、开始时可用 `15,263,739,904` bytes，磁盘可用 `92,331,950,080` bytes。实际工具为 QEMU 8.2.2、Zig 0.14.1。medium/subgraphs/attention/preflight 子进程 RSS 峰值分别为 `649,732,096` / `798,867,456` / `105,816,064` / `147,361,792` bytes；这些不是整个 runner 或编译器/QEMU 子孙进程的合计峰值。环境、线程、依赖原始记录随 artifact 保存。

preparation 产物上传成功且已下载核对：

| 产物 | ZIP 大小 | ZIP SHA256 |
|---|---:|---|
| [w3-preparation-reports](https://github.com/yuki-328/ScratchV/actions/runs/37290898463/artifacts/11336224886) | 1,903,761 bytes，29 文件 | `a83ab4c192e0b3c644b0aa6334dc9d731018f789fe1be131fdc2a4fcb4a19594` |
| [w3-preparation-diagnostics](https://github.com/yuki-328/ScratchV/actions/runs/37290898463/artifacts/11337260339) | 7,378,424 bytes，65 文件 | `10c1951a3f44b770076aaf753f000e86061a768941b7b59b59251778e8631056` |

诊断清单中的 63 个证据文件逐文件哈希匹配、无遗漏；另外两份文件为收集清单本身。下载后的 medium 原始数组已在作者本机离线重算，layer-diff 再次 81/81 PASS，最大误差 `2.205371856689453e-6`。这验证了迁移后选定证据可读且自洽，没有再次执行模型，也没有验证全部 preparation 原始数组。

同提交[通用 CI](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37290861920)为 3727 passed、5 skipped，[W1/W2](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37290862687)与 [Topic06](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37290861847)均通过。同次 fork 运行的 W1 完整 ONNX 校验也通过：有效长度 5/256 两组输出均为有限 FP32 `[1,256,151936]`，总用时 26.701271 s。上述结果分别证明对应范围，不替代完整 W3 七组验收。

### 第二轮：完整模型通过，preparation 失败

运行：[37294081510](https://github.com/yuki-328/ScratchV/actions/runs/37294081510)，提交 `ca110be3654b99cb4caebdab5eff5dc2e6994a7f`，checkout clean。完整模型主报告及 worker 报告已下载并核对；本轮 `numeric:ir-full-qwen3` 为 PASS，七组输入的普通/诊断完整 logits（含 padding）最大绝对误差均为 0，四项不变量也均为 0，总用时 `456.723772664 s`。实际 profile 为 `numpy-fp32-reference-v3`、`cpu_strategy=avx512`。

该轮 preparation 为 FAIL：子图仅 12/14 通过，其余四项准备检查通过。因此整套 W3 编排不通过；首轮 preparation PASS 不能代替该提交上的失败结果。完整模型主报告 SHA256 为 `e34c20c72407ae926949a04b65eb5147534eac2ed170bd63ab6758e740a9f0c8`；报告下载核对不等于原始七组数组的离线复算。后续完整 raw 数组下载与 audit 按实际新提交和运行单独记录，不能提前沿用本轮 PASS。

前两轮原始报告、失败状态、源码和数值 profile 均保持原样。

### 第三轮：preparation 通过，完整模型失败

运行：[37295229214](https://github.com/yuki-328/ScratchV/actions/runs/37295229214)，提交 `8132f3ce5f4e4892663a6bc2ee15ad59dc891f28`，checkout clean。[preparation job](https://github.com/yuki-328/ScratchV/actions/runs/37295229214/job/111714763982) 已通过；完整模型已确认 FAIL，整套编排未通过；后续修复须另存新提交和运行记录。

| 检查 | 第三轮实际结果 | 用时 |
|---|---|---:|
| preparation 回归 | 1270 passed、1 skipped；skip 仅为 Windows Job ownership 测试 | 49.64 s |
| 六层 medium | 7/7 输入、81 检查点、18 不变量；全部比较最大绝对误差 `2.771615982055664e-6 < 1e-5` | 99.530785 s |
| 真实权重子图 | 14/14 输入、10 不变量；全部比较最大绝对误差 `3.0517578125e-5 < 1e-4` | 5.807397 s |
| 小 Attention 后端 | 12/12 执行；QEMU vs ORT 最大绝对误差 `3.5762786865234375e-7 < 1e-4` | 23.216017 s |
| 完整资产预检 | 固定资产及图契约通过；仍不是完整模型数值执行 | 2.189634 s |
| layer-diff | 81/81 检查点；最大绝对误差 `2.205371856689453e-6 < 1e-5` | 包装进程 0.387957 s |
| preparation 汇总 | 五项 PASS；12 次 QEMU 进程时间合计 `1.018794338 s` | 总计 135.615884 s |
| 完整模型和整套编排 | FAIL；preparation 通过不能替代完整模型门槛 | 后续修复与复跑待完成 |

preparation 主报告 SHA256 为 `2f12bdf97f18356829c6222bc86e2da7175983619e17c33cacdac12dd63b3ab1`。runner 为 Ubuntu 24.04.5、Python 3.12.14、4 vCPU AMD EPYC 7763、RAM `16,766,414,848` bytes；开始时可用内存 `15,431,999,488` bytes，磁盘可用 `92,331,909,120` bytes。工具仍为 QEMU 8.2.2、Zig 0.14.1。medium/subgraphs/attention/preflight 子进程 RSS 峰值分别为 `652,468,224` / `807,346,176` / `106,446,848` / `147,103,744` bytes，口径与第一轮相同，不包含整个 runner 或全部子孙进程。

| 第三轮下载产物 | ZIP 大小 | 已核对 ZIP SHA256 |
|---|---:|---|
| [w3-preparation-reports](https://github.com/yuki-328/ScratchV/actions/runs/37295229214/artifacts/11338866162) | 1,905,927 bytes，29 文件 | `1ebdc88ed7f3bb4ae7e076db90b6f990796e7ab6316282d1d4df70124f370089` |
| [w3-preparation-diagnostics](https://github.com/yuki-328/ScratchV/actions/runs/37295229214/artifacts/11338910911) | 7,378,475 bytes，65 文件 | `f1cbb7289e29a383b425871cd18c032f6fe4ddf50e5c7b587303405b94479d59` |

下载后的诊断清单 63 个证据文件共 `21,516,508` bytes，大小及 SHA256 全部匹配，`selection_complete=true`、无 omissions。另两份文件是清单本身。作者本机重新比较下载的 medium `short_17` 原始数组，81/81 PASS、最大误差 `2.205371856689453e-6`；本地复核输出为 `output/w3-linux-37295229214/layer-diff-download-audit-01/report.json`。这仍仅是选定数组迁移后的离线复算，没有再次执行模型，也不替代完整七组 audit 或他人独立复现。

### 第四轮：preparation、完整模型与汇总均通过

运行：[37299235103 / attempt 1](https://github.com/yuki-328/ScratchV/actions/runs/37299235103)，提交 `1c49e8acbcd9174491b2d0c2a5a0072178ae5f9a`，远端 checkout clean。本轮使用已提交的 `numpy-fp32-reference-v4`，通过既有手动入口调用同提交 preparation、全部七组 full-numeric 及汇总。preparation、full-numeric 和整套汇总均为 success；这是同一提交上的完整作者手动 Linux 验证。preparation 下载已复核，完整 reports/raw 产物均已下载并核对 GitHub digest，七组原始数组离线 audit 也已通过。前三轮原始失败结果与原数值 profile 继续保留，不用本轮结果覆盖。

| 检查 | 第四轮实际结果 | 用时 |
|---|---|---:|
| preparation 回归 | 1297 passed、1 skipped；skip 为 Windows Job ownership 测试；真实 Haswell BLAS 回归通过 | JUnit 50.074 s |
| 六层 medium | 7/7 输入、81 检查点、18 不变量；全部比较最大绝对误差 `2.771615982055664e-6 < 1e-5` | 97.368098 s |
| 真实权重子图 | 14/14 输入、10 不变量；全部比较最大绝对误差 `3.0517578125e-5 < 1e-4` | 5.775279 s |
| 小 Attention 后端 | 12/12 QEMU 执行；QEMU vs ORT 最大绝对误差 `3.5762786865234375e-7 < 1e-4` | 22.837814 s |
| 完整资产预检 | 固定资产及 7847 节点图契约通过；不执行完整模型数值 | 2.164485 s |
| layer-diff | 81/81 检查点；最大绝对误差 `2.205371856689453e-6 < 1e-5` | 包含于总计 |
| preparation 汇总 | 五项 PASS；12 次 QEMU 进程时间合计 `0.913058207 s` | 总计 132.923089 s |
| full worker 专项回归 | 581 passed、0 skipped | 15.90 s |
| 完整七组 full-numeric | v4/avx2-fma3 下普通及诊断完整 logits 最大误差 0；每组 30 检查点、4 项不变量的最大绝对误差均为 0 | 总计 1477.314972 s（约 24.6 分钟） |
| 整套手动编排 | preparation、full 与汇总均 success；不是团队正式验收 | 以 Actions 日志为准 |

preparation 主报告 SHA256 为 `ceeeb239ac7b88c22372d0e76f5506841e1b634a81b5b1567af35ca84d443046`。实际环境为 Ubuntu 24.04.5、Python 3.12.14、4 vCPU AMD EPYC 7763；NumPy 2.2.6、ONNX 1.18.0、ORT 1.22.1、Torch 2.7.1+cpu。NumPy 运行诊断记录 AVX2/FMA3，可选库诊断实际读取到 OpenBLAS 0.3.29 的 `core_name=Haswell`；`OPENBLAS_CORETYPE` 未设置（`null`）。这些是分别记录的 CPU 和库事实，不推断 ORT 内部内核。同提交[通用 CI](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37299215953)为 3821 passed、5 skipped，真实 Haswell 回归及 17 项 BLAS 环境证据测试在 Linux 实际通过；[W1/W2 工作流](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37299217144)和 [Topic06](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37299216093)通过。两层小模型 QEMU 为 28/28 通过、最大误差 `1.78813934326e-6`、QEMU 进程用时合计约 146.393 秒；显式产物校验为 3 passed、0 skipped。上述各项分别记录自己的验证范围。

| 第四轮下载产物 | ZIP 大小 | 已核对 ZIP SHA256 |
|---|---:|---|
| [w3-preparation-reports](https://github.com/yuki-328/ScratchV/actions/runs/37299235103/artifacts/11341510503) | 1,908,329 bytes，29 文件 | `0437e1cc3bc9c063ee3b361ed027438bb12939a037dc83a3d237b4b6046df3ec` |
| [w3-preparation-diagnostics](https://github.com/yuki-328/ScratchV/actions/runs/37299235103/artifacts/11341395639) | 7,378,481 bytes，65 文件 | `7516a668991bc97ffb444265aa03d899d3a703dbbd3d72e7081e217cc37ffd02` |

诊断清单 63 个证据文件共 `21,516,508` bytes，逐文件大小及 SHA256 全部匹配，无 omissions 或额外文件；另外两份为清单本身。核对记录位于 `output/w3-linux-37299235103/preparation-independent-audit.json`。下载后的 medium `short_17` 数组另行重算 81/81 PASS，最大误差 `2.205371856689453e-6`，报告位于 `output/w3-linux-37299235103/layer-diff-download-audit-01/report.json`。这是作者对迁移后选定证据的复核，不是第二人重新执行模型，也不等于完整七组 audit。preparation 报告仍为 `full_ir_executed=false`、`w3_exit_accepted=false`。

完整 full 主报告 SHA256 为 `0ddb97d13b6bab44442fb0b453d1e0132ef7a02c96be9c8caf70ae2862e31746`，`git.head=1c49e8acbcd9174491b2d0c2a5a0072178ae5f9a`、`dirty=false`。七组全部普通/诊断 logits（包含所有 padding query）严格 `<1e-4`，本次最大误差为 0；每组 30 检查点与四项不变量的最大绝对误差也均为 0。实际 reference profile 为 `numpy-fp32-reference-v4/avx2-fma3`，NumPy 已装载 OpenBLAS 0.3.29、`core_name=Haswell`，`OPENBLAS_CORETYPE=null`。这是该提交、资产与环境的实测结果，不承诺任意平台逐位一致。

完整 runner 耗时 `1477.314971525 s`，包括加载、普通/诊断前向、比较及证据写盘，并非单次前向性能 benchmark。七组 ORT worker 每组总用时约 15.20–15.53 秒，IR worker 每组约 193.45–195.88 秒。worker 自报全生命周期 RSS 峰值范围：ORT `4,443,017,216–4,443,987,968` bytes，IR `3,102,253,056–3,102,715,904` bytes。父进程采样的进程树 RSS 范围分别为 `4,443,328,512–4,444,389,376` 与 `2,943,205,376–3,057,094,656` bytes；采样可能错过瞬时峰值，不与自报值混用，也不是整个 runner 的总峰值。full runner 为 AMD EPYC 7763、4 vCPU，RAM `16,766,414,848` bytes，开始时可用内存 `15,577,612,288` bytes、可用磁盘 `92,333,924,352` bytes。

| 第四轮完整模型产物 | 大小 / SHA256 | 当前核对状态 |
|---|---|---|
| [w3-full-numeric-reports](https://github.com/yuki-328/ScratchV/actions/runs/37299235103/artifacts/11341033619) | 819,824 bytes；`46047d41334f3336eaf2e39385bffca37551a726891ad6c671c23df1e60f88c5` | 已下载，ZIP digest 与 GitHub 记录一致；主报告及 worker 报告已核对 |
| [w3-full-numeric-raw](https://github.com/yuki-328/ScratchV/actions/runs/37299235103/artifacts/11341924764) | 4,350,864,947 bytes，140 文件；`b99c06f3e39c7c465d31763eba98350682961de1541cab719ce3da1df2ee4171` | 已下载，ZIP digest 与 GitHub 记录一致；完整七组离线 audit 通过 |

下载后的完整 raw 已通过 `audit:w3-full-saved-evidence`，7 组 case、4 项不变量均重新核对，`coverage_complete=true`，`trusted_report_hash_checked=true`、`pinned_model_files_checked=true`，用时 `21.234900 s`。audit 报告位于 `output/w3-linux-37299235103/full-evidence-download-audit-01/report.json`，SHA256 为 `804cc0db397af3dd7fe1696635357999635f0a45cb409410d6bcd6f9b5db2bae`。该核对没有重新执行模型：`model_executed=false`、`independent_reproduction=false`。

原 audit 记录 `source_comparison.matches_current=false`、129 个字节哈希差异。随后只读核对 `1c49e8a` 的 Git blob：159 个生产源码哈希全部与该提交 blob 吻合，129 个本地差异仅为 CRLF/LF 换行，其余 30 个文件逐字节相同；记录位于 `output/w3-linux-37299235103/source-checkout-line-ending-audit.json`（`all_producer_match_committed_blobs=true`、`all_working_copies_identical_or_crlf_only=true`）。原 audit 和生产报告均未改写；这项核对解释源码身份，不将数组 audit 当成重新执行。

当前作者在同提交上的两套 Linux gate、汇总、产物下载及完整数组 audit 均已完成。作者手动远端实跑、作者下载数组 audit、E1/E2/E5 独立执行、定时 schedule 和团队签认仍是不同事项；本轮保留 `w3_exit_accepted=false`，没有启用定时 Nightly。

## 入口与验收范围

| 工作流 | 手动入口 | 可复用入口 | 通过条件 |
|---|---|---|---|
| [w3-preparation.yml](../../../.github/workflows/w3-preparation.yml) | `workflow_dispatch` | `workflow_call`，无参数 | 相关回归和五项 preparation 全部成功，证据收集完整 |
| [w3-full-numeric.yml](../../../.github/workflows/w3-full-numeric.yml) | `case=all` 为默认；`short_17` 仅诊断 | `case` 字符串，默认 `all` | 完整七组真实 28 层 IR/ORT 数值门槛及不变量全部通过 |
| [w3-nightly.yml](../../../.github/workflows/w3-nightly.yml) | 显式运行完整编排，不必先启用定时 | `workflow_call`；调用上述两套同提交工作流 | 两套调用都为 `success`；失败、取消或跳过均不能通过汇总 |

Nightly 使用仓库内相对路径调用，因此被调用工作流与调用者来自同一提交；没有从其他分支临时取最新版脚本。[GitHub reusable workflow 说明](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)

完整比较固定 `--fp32-mode reference`，当前已提交 profile 是 `numpy-fp32-reference-v4`；默认运行全部七组输入。`SCRATCHV_FP32_REFERENCE_CPU` 允许 `auto`（默认）、`avx2-fma3`、`avx512`，决定 IR 参考算术策略，不控制 ORT dispatch。报告保存解析后的 `cpu_strategy`；当前工作流未显式覆盖该变量，因此按 runner 能力自动选择。数值门槛仍为全部普通/诊断 logits 的严格 `max_abs < 1e-4`，含 padding query，`rtol=0`。选择 `short_17` 即使数值正确也返回 `PARTIAL/2`，工作流保持失败状态，不能用单例绿色代替完整验收。`native` 模式和 W4 完整 RISC-V/QEMU 前向不在这套完整 IR 门槛的通过范围内。首跑记录仍属于当时的 v2 契约，不能用当前说明覆盖其原 profile；第二轮 v3/avx512 完整数值通过仅属于该轮提交与环境。

## 先做 Linux 手动验证

若 W3 新入口尚未进入默认分支，但仓库已注册并运行过 `LLM Deploy v1.0`，可从该既有手动入口选择包含 W3 代码的分支，显式设置 `run_w3_validation=true`。它从同一提交复用 W3 完整编排，执行 preparation、七组 full-numeric 和汇总；原 W1/W2 任务也照常执行。默认值为 false，普通 PR 和原 schedule 不会因此自动增加 W3 重型任务。首次远端通过前仍须按实际日志核对，手动调用不算定时 Nightly 记录。

```bash
gh workflow run llm-deploy.yml --repo YOUR_ACCOUNT/ScratchV --ref YOUR_W3_BRANCH -f run_w3_validation=true
```

这个入口不要求更改默认分支，也不设置 Nightly 开关。若既有入口也未注册，则仍需按下列默认分支条件准备入口。[GitHub 手动触发与目标 ref](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch)

1. 选定个人 fork 或有写权限的运行仓库及包含完整 W3 改动的提交；代码现已在 PR 中，不需要重新创建一笔 W3 PR。复核目标分支的实际 SHA，先保持 `W3_NIGHTLY_ENABLED` 未设置；这一步不会启动定时重计算。
2. 确认仓库已启用 Actions，并允许本工作流使用的 actions/reusable workflows。手动入口文件必须已经存在于默认分支，才能正常通过 `workflow_dispatch` 发起；之后可以选择含 W3 修改的目标分支。仅上传到一个没有默认分支入口的功能分支，不应预期必然出现 Run workflow 按钮。[GitHub 手动运行说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
3. 在 Actions 页面运行 `W3 nightly` 的手动入口，可一次执行两套流程；或分别运行 `W3 preparation` 和 `W3 full IR numeric`，后者选择 `all`。记录 run URL、commit SHA 和 attempt，确认两套结果来自同一代码版本。
4. 检查资源日志、依赖安装、资产获取与哈希、回归、QEMU、数值比较和产物上传。需要整个流程完成，不能只看某一步为绿。
5. 在原始产物过期前下载留存，按下文核对。出现 Linux 数值差异时保留失败数组和环境信息，用已有 layer-diff/localize 入口定位；不得通过缩减 case、忽略 padding 或放宽阈值取得通过。

需要命令行时，可在确定运行仓库和分支后使用以下模板；替换占位符。实际已执行记录见本文首节，命令模板本身不是运行证据：

```bash
gh workflow run w3-nightly.yml --repo YOUR_ACCOUNT/ScratchV --ref YOUR_W3_BRANCH
gh run list --repo YOUR_ACCOUNT/ScratchV --workflow w3-nightly.yml
gh run view RUN_ID --repo YOUR_ACCOUNT/ScratchV
gh run download RUN_ID --repo YOUR_ACCOUNT/ScratchV --dir downloaded-w3-run
```

手动运行需要仓库写权限；工作流自身仅请求 `contents: read`，checkout 不持久保存凭据，不使用发布或仓库写入权限。Actions 产物由标准 `upload-artifact` 步骤保存，不是 Git/LFS 提交。

## 环境与资源记录

两套执行工作流固定为 `ubuntu-24.04`、Python 3.12、CPU Torch 2.7.1 和仓库的固定依赖清单；CI 配置测试额外固定 PyYAML 6.0.3。准备项安装 Zig 0.14.1 和 Ubuntu 的 `qemu-system-misc`，记录实际版本。Ubuntu 镜像补丁、apt 包和传递依赖仍可能更新，因此保留 `pip freeze` 及实际系统信息，不能将一个版本标签当成完整镜像指纹。

preparation 的 checkout 显式设置 `fetch-depth: 0`：其 `tests/test_onnx*.py` 回归包含真实历史版本与当前版本的算子 benchmark，需要用 `git archive` 读取固定历史基线。只拉取最新一次提交会缺少该对象，不能正常执行这项回归。完整模型 workflow 的测试不依赖该历史，因此保留默认浅克隆。源码压缩包本身不含 Git 历史，不能把源码包里的历史 benchmark 检查等同于 Git checkout 下的历史比较。

- `w3-runner-resources.txt`：commit、run ID、attempt、发行版、内核、CPU、内存、磁盘容量和工具版本。
- `w3-dependencies.txt`：实际安装依赖版本。
- 数值报告的 `environment.numeric_runtime`：线程环境、CPU/NumPy 构建与运行诊断；进程资源与各阶段耗时继续按原报告记录。
- `environment.numeric_runtime.blas_environment.OPENBLAS_CORETYPE`：启动时环境变量原值，未设置时为 `null`；设置值本身不证明 OpenBLAS 已采用该内核。
- `environment.numeric_runtime.numpy_openblas`：可选的已装载 NumPy 私有 OpenBLAS 库诊断，记录总体 `status` 和 `libraries`；可用的库项含 `path`、`status`、`core_name`、`config`，不可用时保留 `reason` 或 `error`。它不通过额外装载系统库来补造信息。

CPU 的 SIMD 支持、IR 选择的 `cpu_strategy` 和 NumPy 实际 BLAS 内核是不同信息，必须分别保留。系统 BLAS/MKL 等未被此可选诊断覆盖时，`unavailable` 或字段缺失只表示无法取得该项诊断，不能推断内核，也不能使任何数值失败变成通过；完整七组门槛不变。上述 NumPy 诊断不证明 ORT 内部采用哪个内核。

完整流程每个 worker 超时 1800 秒、进程树 RSS 上限 8 GiB，job 超时 60 分钟。这些是失败保护上限，不是资源已足够或预计用时的证明。准备和完整模型在两个不同 runner 上执行；父进程、系统和依赖也占内存。公开仓库标准 x64 Ubuntu runner 的官方配置当前为 16 GB RAM；私有仓库标准配置较小，不能默认满足这套容量要求。选择 runner 前应核对实际资源和账户设置。[GitHub runner 规格](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)

远端首跑还要核对资产下载、安装缓存及全部原始数组的磁盘占用，避免只按模型文件大小估算。OOM、磁盘不足、网络下载失败或 job 超时都属于本次 CI 未通过；应保留日志后调整资源/下载策略，不能将未完成执行计为数值通过。

## 产物与失败定位

| Artifact | 保存期限 | 用途 |
|---|---|---|
| `w3-preparation-reports` | 30 天 | 报告、测试 XML、日志、环境与收集清单 |
| `w3-preparation-diagnostics` | 7 天 | 选定 layer-diff 轨迹及失败 case 数组，受收集器上限约束 |
| `w3-full-numeric-reports` | 30 天 | 完整门槛报告、进度、哈希/schema、环境与测试 XML |
| `w3-full-numeric-raw` | 3 天 | 报告及全部已生成 inputs/logits/diagnostic logits/checkpoints，供离线核对 |

下载后先检查完整模型 `status=PASS`、七组覆盖齐全、数值模式/profile 符合约定、四项不变量通过、源码/模型指纹匹配；准备项则检查五项全部通过与收集清单没有遗漏。报告 JSON 的原有路径可能指向 runner 的绝对目录；迁移后的离线核对应显式指定下载目录，不能默认旧绝对路径仍存在。

完整 raw artifact 内的相对根路径为 `w3-full/`。用 `gh run download` 指定单个 `-n` 时，它直接解压到 `--dir`；所以传给离线工具的是 `--dir` 下的 `w3-full`，不能把下载根目录直接当作模型证据根。下面使用新目录，先分别下载 reports 与 raw，再要求两份主报告字节哈希一致：

```bash
W3_REPO=YOUR_ACCOUNT/ScratchV
W3_RUN_ID=RUN_ID
W3_DOWNLOAD=output/downloaded-w3-RUN_ID
gh run download "$W3_RUN_ID" --repo "$W3_REPO" \
  -n w3-full-numeric-reports --dir "$W3_DOWNLOAD/reports"
gh run download "$W3_RUN_ID" --repo "$W3_REPO" \
  -n w3-full-numeric-raw --dir "$W3_DOWNLOAD/raw"
W3_REPORT_SHA=$(python -c 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$W3_DOWNLOAD/reports/w3-full/report.json")
printf '%s\n' "$W3_REPORT_SHA"
python -B -X utf8 scripts/verify_w3_evidence.py \
  --evidence-dir "$W3_DOWNLOAD/raw/w3-full" \
  --output-dir output/w3-ci-evidence-audit-RUN_ID \
  --expected-report-sha256 "$W3_REPORT_SHA"
```

先核对 Actions run 的完整 SHA、attempt、结果和 artifact 身份，再单独保存 `W3_REPORT_SHA`。这里从 reports 产物取哈希，用它校验 raw 中的报告，能发现两包混用；两包本身来自同一次运行，不构成额外的发布者身份认证。若发件人已通过另一可信渠道给出报告 SHA，应直接用该值作为 `--expected-report-sha256`。Windows 可用 `(Get-FileHash -Algorithm SHA256 -LiteralPath '下载目录/reports/w3-full/report.json').Hash.ToLowerInvariant()` 获取同一值。每次 audit 使用新输出目录，检查返回码以及生成的 `report.json`。

完整原始数组可以用 [复现清单](独立复现与验收清单.md) 中的离线复核入口重新核对。仅有报告和哈希不能重新计算数组误差；选定 preparation 轨迹也不能替代完整七组数组。v4 auditor 依据报告保存的 CPU 策略核对完整规范 profile（包括 MatMul 布局字段），不根据下载者 CPU 推断生产策略，不需要为数组核对更改本机 CPU 策略；旧 v2/v3 证据必须使用对应历史源码的验证器，不能手改报告升级。`audit:w3-full-saved-evidence` PASS 表示保存证据满足其契约，`source_comparison` 明列生成时与当前审计源码的差异；它不执行 ORT/IR，也不能替代第二人在其环境重新执行模型。

上传和证据保留步骤使用 `always()`，以尽可能保存失败现场；它们不清除前面的失败状态。runner 强制终止、job 超时或上传服务失败仍可能导致产物不全，需要检查实际上传结果。仓库设定的保留期限上限和存储额度也需由维护者核对。

## 何时启用 Nightly

先取得同一提交在 Linux 手动完整运行通过的记录，完成资源和产物检查，再与维护者确认运行频率及验收口径。维护者在仓库 Settings → Secrets and variables → Actions → Variables 设置：

```text
W3_NIGHTLY_ENABLED=true
```

只使用非敏感仓库变量，不需要创建访问令牌。变量未设置、为空或为 `false` 时，定时触发的数值任务与汇总任务全部跳过；GitHub 可能仍显示一次被跳过的 schedule 记录，这不代表 Nightly 已通过。需要停用时移除变量或设为 `false`。手动 dispatch 是一次明确运行，允许在定时开关关闭时进行验证。[GitHub 变量与条件说明](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-variables)

配置为每天 `19:23 UTC`，即北京时间次日 `03:23`；未采用整点以减少排队拥堵。schedule 只运行默认分支最新提交，工作流也必须存在于默认分支。GitHub 高负载时可能延迟或漏掉排队任务，公开仓库长时间无活动也可能停用定时工作流；不能把“没有失败消息”当作每天实际运行成功。[GitHub schedule 规则](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)

启用后至少检查实际 schedule run 的 SHA、两套 gate、汇总与产物；连续运行次数及观察周期由团队确认。本地静态验证、手动成功和 Nightly 自动成功分别记录，不能相互替代。

## 本地配置验证

```bash
python -B -X utf8 -m pytest tests/test_w3_workflows.py -q -p no:cacheprovider
actionlint -shellcheck= -pyflakes= .github/workflows/w3-preparation.yml .github/workflows/w3-full-numeric.yml .github/workflows/w3-nightly.yml
```

测试读取实际 YAML，校验默认全覆盖、手动/复用入口、opt-in 条件、独立并发组、失败汇总和产物范围；通过 Bash 执行实际参数组装及汇总片段，确保 `FAIL/1`、`PARTIAL/2` 不被吞掉，并对所有 shell 块做语法检查。Windows 可使用已有 Git Bash；没有 Bash 时 shell 执行测试会明确跳过，不能据此声称该部分已经验证。

本轮使用 actionlint 1.7.12 对上述三个文件静态检查通过；工具放在本地忽略的 output 下，未安装系统服务。actionlint 不执行下载、Python 环境安装、QEMU 或数值模型；实际 Linux 执行与产物核对另见本文首节。
