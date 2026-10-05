# W3 Linux CI 与 Nightly

当前完成的是本地配置和静态/脚本验证。本次没有提交、上传、触发 GitHub Actions 或设置仓库变量；没有远端 Linux 或 Nightly 通过记录。下面的远端操作是后续执行说明，不能作为已经执行的证明。独立复现和团队确认内容见 [独立复现与验收清单](独立复现与验收清单.md)。

## 入口与验收范围

| 工作流 | 手动入口 | 可复用入口 | 通过条件 |
|---|---|---|---|
| [w3-preparation.yml](../../../.github/workflows/w3-preparation.yml) | `workflow_dispatch` | `workflow_call`，无参数 | 相关回归和五项 preparation 全部成功，证据收集完整 |
| [w3-full-numeric.yml](../../../.github/workflows/w3-full-numeric.yml) | `case=all` 为默认；`short_17` 仅诊断 | `case` 字符串，默认 `all` | 完整七组真实 28 层 IR/ORT 数值门槛及不变量全部通过 |
| [w3-nightly.yml](../../../.github/workflows/w3-nightly.yml) | 显式运行完整编排，不必先启用定时 | 调用上述两套同提交工作流 | 两套调用都为 `success`；失败、取消或跳过均不能通过汇总 |

Nightly 使用仓库内相对路径调用，因此被调用工作流与调用者来自同一提交；没有从其他分支临时取最新版脚本。[GitHub reusable workflow 说明](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)

完整比较固定 `--fp32-mode reference`，当前 profile 是 `numpy-fp32-reference-v2`；默认运行全部七组输入。数值门槛仍为全部普通/诊断 logits 的严格 `max_abs < 1e-4`，含 padding query，`rtol=0`。选择 `short_17` 即使数值正确也返回 `PARTIAL/2`，工作流保持失败状态，不能用单例绿色代替完整验收。`native` 模式和 W4 完整 RISC-V/QEMU 前向不在这套完整 IR 门槛的通过范围内。

## 先做 Linux 手动验证

1. 后续获得上传授权后，将完整 W3 源码提交放入个人 fork 或有写权限的仓库。先保持 `W3_NIGHTLY_ENABLED` 未设置；这一步不会启动定时重计算。
2. 确认仓库已启用 Actions，并允许本工作流使用的 actions/reusable workflows。手动入口文件必须已经存在于默认分支，才能正常通过 `workflow_dispatch` 发起；之后可以选择含 W3 修改的目标分支。仅上传到一个没有默认分支入口的功能分支，不应预期必然出现 Run workflow 按钮。[GitHub 手动运行说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
3. 在 Actions 页面运行 `W3 nightly` 的手动入口，可一次执行两套流程；或分别运行 `W3 preparation` 和 `W3 full IR numeric`，后者选择 `all`。记录 run URL、commit SHA 和 attempt，确认两套结果来自同一代码版本。
4. 检查资源日志、依赖安装、资产获取与哈希、回归、QEMU、数值比较和产物上传。需要整个流程完成，不能只看某一步为绿。
5. 在原始产物过期前下载留存，按下文核对。出现 Linux 数值差异时保留失败数组和环境信息，用已有 layer-diff/localize 入口定位；不得通过缩减 case、忽略 padding 或放宽阈值取得通过。

需要命令行时，可以在未来已授权的远端阶段使用以下示例；替换仓库与分支占位符，本次未执行：

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
- 数值报告的 `environment.numeric_runtime`：线程环境、CPU/NumPy/BLAS 诊断；进程资源与各阶段耗时继续按原报告记录。

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

完整原始数组可以用 [复现清单](独立复现与验收清单.md) 中的离线复核入口重新核对。仅有报告和哈希不能重新计算数组误差；选定 preparation 轨迹也不能替代完整七组数组。作者产物的离线复核不能替代第二人在其环境重新执行模型。

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

本轮使用 actionlint 1.7.12 对上述三个文件静态检查通过；工具放在本地忽略的 output 下，未安装系统服务。actionlint 不执行下载、Python 环境安装、QEMU 或数值模型，仍需后续 Linux 实跑。
