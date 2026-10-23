# W5：Linux 容器部署入口

本页说明 [Dockerfile](../../../deployment/w5/Dockerfile) 和
[一键脚本](../../../scripts/deploy_w5.sh) 的使用方式。
生成合同、固定参考和阶段门槛见 [W5 总览](README.md)。

## 部署范围

镜像使用 Ubuntu 24.04 / Python 3.12，安装发行版 `qemu-system-misc`，并直接复用
`python -m pip install -e ".[llm-deploy]"` 固定的依赖，其中包括 Zig 0.14.1。
不再维护第二份 Python/Zig 版本列表。镜像中的 QEMU 由 Ubuntu 软件源提供；
正式验收须记录实际版本，不能将它的耗时与另一 QEMU 版本的历史记录直接比较。

基础镜像默认仍为 `ubuntu:24.04`。Docker Hub 不可达时，可显式设置
`SCRATCHV_W5_BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:24.04`，选择 Canonical 发布的
[Amazon ECR Public Ubuntu 镜像](https://gallery.ecr.aws/ubuntu/ubuntu)；
来源说明见 [Ubuntu 官方容器文档](https://ubuntu.com/docs/oci-registries/)。
脚本将该值作为一个完整的 `--build-arg BASE_IMAGE=...` 参数传入 Dockerfile；
未设置或为空时保留默认值，不自动切换来源、不重标记 `ubuntu` 镜像，也不修改引擎或代理配置。
例如，在下方部署命令前增加：

```bash
SCRATCHV_W5_BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:24.04 \
  bash scripts/deploy_w5.sh --help
```

此例仍会通过原脚本构建镜像，再运行 CLI 帮助；实际生成时追加下方原有参数即可。

[构建上下文白名单](../../../deployment/w5/Dockerfile.dockerignore) 仅包含源码、安装
元数据和命令行入口所需文件，模型权重、tokenizer、虚拟环境、ELF、Git 和历史输出
均不随镜像构建上下文发送。模型资产在运行时只读挂载到 `/assets`，构建和报告写入
可写挂载 `/output`。脚本按调用者的 Linux UID/GID 运行，避免产生 root 所有的报告。

脚本只做容器预检、构建与启动，所有生成参数原样传给已有 CLI，不再实现一套
采样、token 循环、模型下载或校验流程。未准备资产时明确失败，不自动下载权重。
Linux 主机需要 `setsid`（通常由 util-linux 提供）。引擎预检和镜像构建客户端在各自
进程组内运行；脚本收到 SIGINT/SIGTERM 时会向客户端和辅助进程转发终止、等待清理，
5 秒后对仍存在的进程组强制结束，并以 130/143 返回，不继续启动模型。
容器启动后保留 `exec ... run --init` 的信号转发路径。本轮已额外验证真实 Docker
构建取消时客户端进程组和引擎侧 RUN 进程均消失，实测范围和证据见下文。
Docker 是本页的部署目标；替换容器 CLI 前需确认其支持 Dockerfile 专用 ignore 文件，
本轮未验证 Podman 兼容性。

## 小图部署验证

先在正常 Linux 开发环境准备一个满足生成 ABI 的小图及对应本地 tokenizer，放在同一
资产目录下：`model.onnx`、`tokenizer/`；若 ONNX 使用外置数据，数据文件也必须位于
该挂载目录中。小图使用与正式生成相同的固定长度输入和 logits 接口，但不能当作
Qwen3 的语言能力或 Paris 门槛证据。

本轮已有真实 CLI 小图资产 `output/w5-cli-small-20261023/model.onnx` 和
`output/w5-cli-small-20261023/tokenizer/`，可在同一工作树中复用。先复制到统一的
只读资产挂载目录，再执行下面的容器命令；本轮真实容器结果见页末：

```bash
mkdir output/w5-small-assets
cp -a output/w5-cli-small-20261023/model.onnx \
  output/w5-cli-small-20261023/tokenizer output/w5-small-assets/

SCRATCHV_W5_ASSETS="$PWD/output/w5-small-assets" \
SCRATCHV_W5_OUTPUT="$PWD/output/w5-container-smoke" \
  bash scripts/deploy_w5.sh \
  --model /assets/model.onnx \
  --tokenizer-dir /assets/tokenizer \
  --prompt 1 --max-new-tokens 8 \
  --matmul-policy blocked_fma \
  --harts 1 --timeout 30 --total-timeout 120 \
  --verify-onnx /assets/model.onnx
```

2026-10-23，以上资产已通过宿主 Linux CLI 的真实 QEMU 对照：1 和 4 hart 均
**PASS**，各运行 4 次前向，生成 IDs 同为 `[2, 3, 4, 6]`，文本为 `2 3 4`，由 EOS
停止；每步进行 ORT 数值/greedy 对照。使用显式 `blocked_fma / 1x4 / original`、
padding `none`。证据位于 `output/w5-cli-final-20261023/hart1/report.json` 和
`output/w5-cli-final-20261023/hart4/report.json`，这两份是本地诊断产物，不提交 Git。
上述两份记录是**宿主 CLI 小图通过**，本轮新增的真实容器证据另列于页末。宿主完整模型的初始旧算术
配置曾使三例均在首步完整 logits 检查失败；该历史结果保留于 [实跑报告](qemu-generation.md)。
随后显式启用 Mean 与非线性参考策略的正式 4 hart QEMU 套件已完成 **3/3 用例、24/24 步通过**，
数值、冻结 IDs、最终文本与停止原因均满足合同。通过配置对应下方完整验收示例，
默认单核、Mean `sequential`、非线性 `libm` 与 padding `none` 不变。
这些宿主结果不替代容器证据；本轮容器小图和清理、完整 Paris 单 token 均已有独立实跑通过记录。

脚本默认将 CLI 输出写入 `/output/run`，该子目录必须尚不存在；重复验证时显式
增加 `--output-dir /output/run-h4` 等新目录，并传 `--harts 4`。不重复使用已有输出目录，
也不自动删除旧报告。宿主输出目录可以包含空格，挂载路径不可包含逗号。

没有指定 `--harts` 时仍为单核，手动 `1/2/4` 和 `auto` 原样交给 CLI。`auto` 在构建时
解析一次，不在 token 之间改变 ELF 的 hart 数。padding 始终关闭。运行器每个 token
可以重新启动 QEMU，但复用同一轮构建得到的 ELF 和权重 bundle，不按 token 重新编译。

`--reuse-build` 仅复用相同容器路径下的已保存构建。构建元数据可能包含绝对路径，
不要直接把宿主生成的 ELF/bundle 元数据拿到 `/output` 下宣称可跨环境复用；跨环境
从 `--model` 重新构建。在容器内先构建、后复用时，保持挂载目标和构建路径一致。

## 容器完整模型最短生成命令（本轮已通过）

固定真实模型资产沿用 W1 manifest 和 W5 三条参考；目录结构是
`onnx/model.onnx`、外置权重和 `tokenizer/`。本轮仅补 France 的一个真实生成 token，
不在容器重复已经通过的宿主三例 24 步：

```bash
SCRATCHV_W5_ASSETS="$PWD/assets/qwen3-0.6b" \
SCRATCHV_W5_OUTPUT="$PWD/output/w5-container-acceptance" \
SCRATCHV_W5_BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:24.04 \
  bash scripts/deploy_w5.sh \
  --model /assets/onnx/model.onnx \
  --tokenizer-dir /assets/tokenizer \
  --prompt 'The capital of France is' --max-new-tokens 1 \
  --verify-onnx /assets/onnx/model.onnx --harts 4 \
  --matmul-policy blocked_fma --matmul-tile 2x4 \
  --matmul-rhs-layout column-panel \
  --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
  --timeout 1800 --total-timeout 3600
```

该命令执行一遍完整 28 层模型，验证容器内实际生成的 ` Paris`，不会运行 3×8 token。
本轮实测产物位于 `output/w5-container-20261023/paris/`，已成功生成 ` Paris`，进程退出码为 `0`。
单 prompt 命令的 `full_model_gate=NOT_RUN` 表示没有再次运行三例 suite；
不能把它误读为没执行完整模型，也不能将单次成功计为容器三例通过。
宿主三例和独立单次前向记录继续保留于 [实跑报告](qemu-generation.md)。

可设置 `SCRATCHV_CONTAINER_ENGINE` 为 Docker 可执行文件，
`SCRATCHV_W5_IMAGE` 为本地镜像标签；默认分别为 `docker` 和 `scratchv-w5:local`。
一键脚本会先检查引擎存在且可用，再构建镜像；不会安装引擎、修改 WSL 或 Windows。

## 本轮真实容器验证（2026-10-23）

早期检查时环境没有容器引擎，历史静态检查不冒充容器验收。本轮已使用 **Docker
Engine 29.9.2 / Buildx 0.38.1** 真正构建镜像并运行容器。引擎仅手动启动；
验证结束后 `docker.service / docker.socket / containerd.service` 均已
**inactive + disabled**，停止前无运行容器。临时 socket 权限恢复为 `root:docker 0660`，
未新增永久组成员，WSL 配置原文保持不变；已安装引擎保留，不开启自启。

Docker Hub 的认证请求出现 TLS EOF，IPv4/TLS 1.2 复查仍失败；本轮因此显式设置
`SCRATCHV_W5_BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:24.04`，使用上方 Canonical
官方来源。默认仍为 `ubuntu:24.04`，没有重标记该镜像或修改代理配置。
基础镜像和部署镜像实际拉取/构建成功；小图镜像内 Python **3.12.3**、Zig **0.14.1**、
QEMU **8.2.2（Ubuntu 包 1:8.2.2+ds-0ubuntu1.18）**、ORT **1.22.1**、
NumPy **2.2.6**，Torch 为 **2.7.1+cpu**。

| 真实容器检查 | 结果 |
|---|---|
| 默认单 hart CLI 小图 | PASS；4 次前向，IDs `[2,3,4,6]`，文本 `2 3 4`，EOS 停止 |
| 手动 4 hart CLI 小图 | PASS；同上，完整 logits 逐位 ORT 一致 |
| `auto` 复用 4 hart 构建 | PASS；解析为 4，4 次前向，IDs/text/EOS 一致 |
| 三组小图数值与布局 | 全部 `max_abs=0`，padding `none` |
| 真实 QEMU 1/4 hart 多 token、超时、取消、复用 | 2 项测试 PASS、0 skip；每个配置 8 次 guest 启动（6 次成功、2 次预期中断），16 次进程/进程组均回收 |
| Docker 实际构建 SIGINT/SIGTERM | 2/2 PASS；退出码 130/143，客户端组与引擎侧 RUN 进程均消失，未启动模型 |
| 完整 28 层 France 单 token 生成 | **PASS**；IDs `[12095]`、文本 ` Paris`，预算停止，完整 logits `max_abs=0.000172138214 < 0.001`；28 层图审计、FMA、QEMU 回收均通过 |

QEMU 清理测试直接复用已有 `test_real_qemu_multitoken_reuse_numerics_timeout_and_cancel`，
不同上下文生成 `[4]`，超时/取消后再次生成 `[6]`，均使用同一配置内的 ELF/bundle。
真实 Docker 构建取消使用原部署脚本的逐字节相同副本和独立小型诊断上下文，
在实际引擎侧 RUN 进程已运行后分别发 SIGINT/SIGTERM；两次约 **0.117s / 0.115s**
观察到清理完成，延后复查无残留。它验证实际 Docker 构建生命周期，不运行模型或下载资产。

之前的 Bash 语法及 **17 项**隔离/打包测试继续有效，包含默认基础镜像、空值回退、
ECR 与字面量参数边界及客户端取消；本轮真实引擎结果另列，不把替身检查重复累计成真实容器测试。
完整 Paris 验收由实际 Docker 调用独立执行：容器内从只读 ONNX 资产重新构建 ELF/bundle，
显式使用 `4 hart / blocked_fma / 2x4 / column-panel / mean two-four-lane-last-axis /
nonlinear fp32-reference / padding none`，仅生成 1 token。输入 prompt IDs 为
`[785,6722,315,9625,374]`，输出 `[12095]`、` Paris`，停止为 `max_new_tokens`。
全部 **38,895,616** 个 logits 对当轮 ORT 的最大误差为 **0.000172138214**，没有跳过补齐行。

| 容器 Paris 实测范围 | 结果 |
|---|---:|
| CLI 总耗时（含独立重新构建、FMA、ORT 和核验） | 547.454 s |
| UART 前向耗时（含 QEMU 启动和装载） | 415.058 s |
| QEMU 完整进程耗时（另含 stop/dump/quit） | 415.478 s |
| 运行器记录耗时（权重传输、QEMU、后检查及清理） | 430.481 s |
| QEMU RSS 观测峰值 | 5.398 GiB |
| 模型 QEMU 前向次数 / 成功回收 | 1 / 1 |

运行器记录耗时不含计时起点之前的输入打包、输入文件写入及 ELF/bundle 校验，不能当作整个 adapter 调用耗时。

RSS 来自 0.2s 周期采样，共 2071 个有效样本；有 **1 次 RSS 读取失败**，
记录为 `partial`，上述值是观测峰值，不声称完整无缺失采样。本次没有独立采集
QEMU/Host CPU 总时间，不用 Docker 客户端时间或宿主前向记录冒充容器 CPU。
完整 **155,582,464 字节**输出已直接与既有宿主独立完整前向比较，**逐字节一致**，
没有新增哈希检查。
正式 CLI 保留 `full_model_gate=NOT_RUN` 和 `container_reproduction=NOT_ASSERTED`：
单 prompt 没有重跑三例 gate，而容器身份不由内部 CLI 自行宣称。外层真实 Docker 构建、
容器调用、退出码与完整结果共同构成 **deploy:script PASS（本地真实容器）** 的证据；
宿主三例 24 步继续单独记录，不扩张为容器三例通过。

本地证据路径相对仓库根：

- `output/w5-container-20261023/engine.json`、`image.json`：引擎和镜像内实际工具版本。
- 同目录 `default-h1.log`、`base-pull.log`、`default-h1-ecr.log`：Docker Hub 失败、官方 ECR 拉取与实际构建记录。
- 同目录 `small-summary.json` 与 `small-runs/{default-h1,manual-h4,auto-reuse}/report.json`：三组真实 CLI 小图。
- 同目录 `small-runs/real-qemu-tests.xml`、`real-qemu-tests.log`，以及 `small-runs/real-qemu-tests/` 下两份 `generation-check.json`：真实 QEMU 数值、中断及复用。
- 同目录 `build-cancel/report.json`、`build-cancel/delayed-recheck.json`：Docker 实际构建取消与延后清理检查。
- 同目录 `paris.log`、`paris-summary.json`、`paris/run/report.json` 和 `paris/run/steps/step-0000/run.json`：本轮唯一完整模型容器生成、图审计/FMA/数值与回收证据。
- 同目录 `services-final.json`：容器清空、服务 inactive/disabled、临时 socket 权限恢复与 WSL 配置未变。

W5 CI 已复用既有流程接入；容器收尾时 753 项本地轻量检查通过，新增必跑节点仅收集核对。
后续提交前检查补齐 W5-only 手动入口和 job 余量，811 项相关检查通过，最终 11 项重叠
边界复查通过，未重新执行完整模型，见 [提交前收尾](README.md#提交前检查与收尾2026-10-23)。
具体模式、资源要求与证据边界见 [CI 接入说明](README.md#ci-接入与远端执行边界)。
远端 **NOT_RUN**，未提交、未推送，不能将本地结果写成远端 CI 已通过。
W5 本地功能出口与本轮容器收尾已完成；默认配置及 WSL 配置不变。
