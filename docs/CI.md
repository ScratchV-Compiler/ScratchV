# PR 验收与 Actions 报告入口

配置入口是 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)，所有工作流位于 [`.github/workflows/`](../.github/workflows/)。本次改动见[简化前后对比](#simplification-report)。

在 PR 的 **Checks → CI → CI overview** 查看本次需要执行的任务和实际结果。点击相应任务查看失败步骤或数值摘要，运行页底部的 **Artifacts** 提供本次运行的原始报告。

| 想看什么 | 去哪里看 |
| --- | --- |
| 单测通过、失败、错误、跳过明细 | 代码 PR 的 `test-reports` 制品内有 `test_results.xml`。主干测试还生成 `tests.html`，通过总验收后发布到 Pages |
| 性能数据 | `benchmark` 和 `Topic 06 benchmark` 的摘要及制品。窥孔优化有独立的 `peephole-benchmark-report` |
| 模型部署结果 | 部署任务提供小模型数值、完整模型前端解析报告。`W3/W4/W5 full validation summary` 表示各阶段是否被请求、任务结果和报告名 |
| 完整模型验证 | 在手动运行页查看实际执行项目的报告和原始数据。`NOT RUN` 表示这次没有此项数据 |

测试网页只读取本次 `test` 任务的 JUnit 报告。通过率的分母包括通过、失败和错误的用例，跳过单列。全跳过时显示“无已执行用例”。缺失或无法解析报告会使页面生成失败。

## 哪些改动会运行哪些检查

- 普通代码 PR 运行通用测试、编译器 benchmark、Topic06 和 Linux 文档检查。`scripts/ci_scopes.json` 指定哪些路径还需要模型部署和 W4/W5 小图验证。
- 只有 PR 中的**全部改动**均在 `scripts/ci_scope.py` 的六个普通说明文件内，且没有选中部署专项时，才仅检查公开命令、Bash 语法和文档链接。名单包括根目录的 `README.md`、`CONTRIBUTING.md`、`docs/INDEX.md`，以及 `docs/guide` 下的编译器概念、指标解读和故障排除指南。
- 改动部署文档、加入名单外文件、混合代码与文档，或从名单外路径移动文件，均使用完整的常规 PR 检查。主干推送也执行常规测试和 benchmark。路径判断使用 PR 的 merge-base diff，同时检查改名前后的路径。
- `CI acceptance` 核对所选任务成功，未选择的任务显示 `NOT RUN`。无法取得范围、必要任务缺失、失败、意外跳过或取消都会使验收失败。新 PR 提交会取消该 PR 的旧运行。
- `AI Code Review` 独立运行，从可信的基础分支读取配置并使用评论权限。普通测试执行 PR 代码。仓库若通过任务名称设置分支保护，维护者应在发布改名后的任务时核对所需检查。

## 工作流入口

| 文件 | 用途 |
| --- | --- |
| [ci.yml](../.github/workflows/ci.yml) | 普通 PR 总入口、benchmark、总验收和主干 Pages 发布 |
| [topic06-benchmark.yml](../.github/workflows/topic06-benchmark.yml) | DSL 案例的编译、执行和报告 |
| [llm-deploy.yml](../.github/workflows/llm-deploy.yml) | W1/W2 及相关模型检查、W3/W4/W5 手动入口和完整验证摘要 |
| [w3-preparation.yml](../.github/workflows/w3-preparation.yml) | W3 准备检查、子图和相关单测 |
| [w3-full-numeric.yml](../.github/workflows/w3-full-numeric.yml) | 完整模型 IR 与 ONNX Runtime 的数值对比 |
| [w3-nightly.yml](../.github/workflows/w3-nightly.yml) | 组合 W3 两个任务，要求都成功。定时执行受仓库开关控制 |
| [w4-full-numeric.yml](../.github/workflows/w4-full-numeric.yml) | 默认运行小图，可显式选择 W4 单次完整前向或 W5 容器三例生成 |
| [w4-nightly.yml](../.github/workflows/w4-nightly.yml) | 按仓库开关运行 W4 完整前向 |
| [ai-code-review.yml](../.github/workflows/ai-code-review.yml) | 获取 PR 文件、分批审查、汇总审查覆盖情况；本轮未修改 |

工作流调用的主要脚本：

| 路径 | 用途 |
| --- | --- |
| [scripts/ci_scope.py](../scripts/ci_scope.py)、[ci_scopes.json](../scripts/ci_scopes.json) | 根据改动文件选择检查范围 |
| [scripts/summarize_ci.py](../scripts/summarize_ci.py) | 汇总必跑任务，缺失、失败或意外跳过时返回失败 |
| [scripts/check_linux_repro_docs.py](../scripts/check_linux_repro_docs.py)、[check_docs_links.py](../scripts/check_docs_links.py) | 检查公开文档中的命令、Bash 语法和链接 |
| [scripts/run_topic06_benchmarks.py](../scripts/run_topic06_benchmarks.py) | 运行 DSL 案例，并检查必需类别的结果 |
| [scratchv/ci/test_page.py](../scratchv/ci/test_page.py) | 将实际 JUnit 测试结果转换为网页 |
| [tests/test_ci_workflows.py](../tests/test_ci_workflows.py)、[test_ci_routing.py](../tests/test_ci_routing.py) | 检查工作流调用关系、路径选择与失败处理 |

<a id="避免重复执行单测"></a>

## 单测、环境和 benchmark 的分工

| 部分 | 执行方式 |
| --- | --- |
| 部署与 W4/W5 小图 | 普通 PR 由必跑的通用测试负责输入处理、词表大小、特殊 token 编号、前端算子模式和工作流配置单测。独立调用、手动及定时入口仍执行自己的测试。真实 QEMU、数值、超时清理和模型集成检查在相应环境中运行 |
| 进程清理 | 普通 PR 覆盖正常清理。重复 20 次的检查留在手动和定时运行中 |
| W3 | 组合入口由准备任务执行重叠单测，完整数值任务执行七例 IR 验证。独立启动完整数值任务仍先执行相应单测 |
| Linux 文档 | 普通委托 PR 在总入口检查一次，部署任务依赖该结果。独立调用、手动及定时运行部署任务时仍自行检查 |
| 依赖 | 通用测试一次安装 `.[all,llm-deploy]`。W3 和 W4 完整前向一次安装 `.[llm-deploy]`，保留固定 CPU Torch 和 `pip check`。pip 缓存下载文件，部署探测使用独立虚拟环境 |
| C 与命令行单测 | CNN 异常测试复用本轮编译和分析产物。外置权重测试按图和编译配置复用真实 C 库，每例独立准备运行数据。W5 模拟命令行用例检查清理调用，另有实际回收用例 |
| 窥孔优化 | `CI.test` 执行单测及命令行检查，`make bench-peephole` 负责实测并生成报告。本地 `make ci-peephole` 先测试再测量 |
| Topic06 | 23 个案例各执行一次。`activation`、`elementwise`、`loop` 三个必需类别中，有类别缺失或有用例失败就判任务失败。其他类别的结果写入诊断报告 |

配置单测覆盖实际不同的判断条件，包括缺失结果、跳过、失败、错误及取消。单测模拟的失败摘要写入 pytest 临时目录。正式 Actions 页面展示本次任务的结果。

<a id="simplification-report"></a>

## 减少 CI 重复测试，合并 W3/W4/W5 验证报告

普通代码 PR 中，通用测试任务统一检查输入处理、词表大小和结束、填充等特殊 token 的编号，每项只跑一次。W3 组合验证中，17 个共用的测试文件只跑一次。W4/W5 合并了重复的状态检查，仍会逐项检查必跑用例是否缺失。主测试任务收集的用例从 8,603 条减到 8,356 条。

PR 的全部改动都在指定的 6 份说明文档内时，只运行文档检查。Topic06 的 23 个案例各跑一次，执行次数从 35 次降到 23 次。通用测试合并了依赖安装，CNN 和外置权重 C 测试复用相同配置的编译结果。

W3/W4/W5 的三个摘要合成一张表，列出各阶段的执行结果和报告位置。测试网页读取本次运行生成的 JUnit 报告，文件缺失或损坏时会报错。W5 单测模拟生成失败时，失败摘要写入测试临时目录。

本地 230 条 CI 相关测试通过。前后各测一次，15 条 CNN 调度测试从 32.77 秒缩短到 8.84 秒，9 条外置权重 C 接口测试从 1.91 秒缩短到 1.04 秒。整条 CI 的耗时待 GitHub Actions 实跑确认。

### 简化前后明细（2026-11-06）

本节的基线为 `b7192c3`，与 W5 合入主干时的 `a99d035` 文件内容相同。对比覆盖 9 个工作流中的重复执行、测试准备和报告生成。

| 部分 | 修改前 | 修改后 | 为什么这样改 |
| --- | --- | --- | --- |
| Linux 文档检查（本次） | 普通委托 PR 的总入口和部署任务各检查一次 | 总入口检查一次；独立部署入口仍检查 | 同一命令、同一提交，部署任务已依赖总入口成功 |
| 普通说明文档 PR | 通用测试和两组 benchmark 都运行 | 仅当改动全在 6 个明确列出的说明文件中，运行文档检查 | 这些文件不参与编译或推理；未知文件和混合代码改动仍执行常规检查 |
| 部署和 W4 的通用单测 | 部分输入处理、词表大小、特殊 token 编号、前端和配置测试在通用任务后又跑 | 普通 PR 交给必跑的通用任务；独立入口保留 | 同一提交的相同逻辑只需检查一次；实际 QEMU 和环境相关检查仍各自执行 |
| W3 组合入口 | 准备任务和完整数值任务重复执行 17 个测试文件 | 准备任务负责单测，完整数值任务负责 7 例数值对比 | 准备任务已经覆盖这 17 个文件，组合结果仍要求两个任务都成功 |
| 窥孔优化 benchmark | `make ci-peephole` 再跑一遍通用任务已执行的单测 | `make bench-peephole` 只做性能测量与报告 | 不重复做功能检查，保留性能测量次数；本地原命令仍先测功能 |
| Topic06 案例 | 必需类别 12 例，再跑全量 23 例，共 35 次 | 全量 23 例各跑一次，从同一结果检查必需类别 | 少执行 12 次；类别缺失或其用例失败仍使任务失败 |
| 参数化状态检查 | 3 组测试共 299 种组合 | 保留 72 种能区分行为的组合 | 去掉程序不读取的事件维度和相同提前返回；每个必跑项缺失仍逐个检查 |
| W3/W4/W5 状态报告 | 3 个摘要任务、61 条相关状态测试 | 1 张汇总表、17 条针对不同分支的测试 | 避免打开多张内容相近的卡片；请求过但未运行仍判失败 |
| 依赖安装 | 通用测试先装一般依赖，再安装固定模型依赖；部分任务无 pip 缓存 | 合并依赖解析，按环境复用下载缓存 | 避免重复下载、版本替换；保留固定 CPU Torch 检查和 `pip check` |
| CNN 测试准备 | 多个异常测试重复编译和分析相同 CNN | 共享一次真实构建，每例复制要修改的数据 | 保留各项断言，省去重复准备 |
| 外置权重 C 测试 | 9 条相关测试进行 13 次编译 | 相同配置共用 C 库，编译 6 次 | 每例仍独立创建输入、权重和工作区，真实 C 调用保留 |
| W5 命令行模拟测试 | 无模型资源的测试也反复进行 Python 全量垃圾回收 | 检查回收调用，另保留实际回收用例 | 去掉模拟测试中无意义的全堆扫描，不改正式程序清理逻辑 |
| 测试网页 | 测试与 benchmark 各生成一次，后者没有本次测试 XML | 只从测试任务的 XML 生成一次，总验收成功后发布 | 避免缺失数据被显示为 100%；错误和跳过分别统计 |

20 次进程清理检查已在基线中限制到手动和定时运行，本轮继续保留。普通 PR 合并入口和自动取消同一 PR 的旧运行也已存在，不计作本轮新增收益。

### 本地耗时对照

11 月 6 日在同一环境中，对基线和修改后的测试各运行一次。只替换测试源码，正式实现和 pytest 配置相同；使用 Python 3.12.3、Clang 18.1.3、LLVM 18.1.3。

| 测试范围 | 修改前实际耗时 | 修改后实际耗时 | 耗时减少 | 结果 |
| --- | ---: | ---: | ---: | --- |
| CNN 调度与异常检查，15 条 | 32.766 秒 | 8.838 秒 | 73.0% | 前后均 15 条通过 |
| 外置权重 C 接口，9 条 | 1.907 秒 | 1.038 秒 | 45.6% | 前后均 9 条通过；实际共享库构建 13→6 次 |

这里记录的是整个 pytest 进程的实际经过时间，包含启动和用例收集。四次运行都没有失败或跳过。前后各一次只能说明本机观察到的变化，不代表稳定平均值，也不是 GitHub Actions 总耗时的降幅。

本地原始记录集中在 `output/ci-simplification-20261106/`：`local-measurement.json` 包含完整命令、前后耗时、CPU 时间和内存记录，同目录保留日志、JUnit 和参与比较的测试源码。这些临时测量文件不纳入 Git，本文保留可阅读的结论。

### 本次验证

| 检查 | 结果 |
| --- | --- |
| CI 工作流关系、路径选择、结果汇总和测试网页 | 230 条通过 |
| Topic06 报告和失败处理 | 20 条通过；没有重新执行完整 benchmark |
| 指定模拟器用例 | 7 条通过 |
| CNN 和外置 C 前后对照 | 前后各 24 条通过，见上表 |
| 工作流配置与脚本语法 | 9 个 YAML 可解析，任务依赖无环，128 个 Bash 步骤语法通过 |
| 文档命令和链接 | 129 份文档、249 个 shell 代码块通过；394 个文档链接无失效目标 |

<details>
<summary>相关验证命令</summary>

以下检查不运行完整模型。在已安装项目测试依赖的环境中，从仓库根目录执行：

```bash
python -m pytest tests/test_ci_workflows.py tests/test_ci_routing.py tests/test_ci_test_page.py -q
```

```bash
python -m pytest tests/topic06/test_integration.py tests/test_simulator.py::TestStubProfiledMachine tests/test_simulator.py::TestRealProfiledMachine::test_lw_preserves_all_four_bytes -q
```

```bash
python scripts/check_linux_repro_docs.py --check-bash
```

```bash
python scripts/check_docs_links.py
```

</details>

### 仍然保留哪些检查

- 模拟器的指定用例、真实 RV64/QEMU 小图、数值比较、超时、取消和清理检查保留。
- Topic06 在 Python 3.11 下的测试保留；通用任务使用 Python 3.12，不能把不同环境的检查直接当成重复删除。
- benchmark 的预热、重复采样和优化前后测量保留；减少采样次数会影响性能结论。
- 独立、手动和定时入口能自行完成必要检查；未获委托时不跳过通用单测。
- AI 审查按仓库现有规则在 PR 更新时自动触发。

<a id="follow-up"></a>

### 11 月 7 日核对结果

| 核对项 | 结果与决定 |
| --- | --- |
| 两次 pytest 能否合并 | 保留两次。主任务同口径收集从 8,603 条降到 8,356 条，其中包括 7 条单独选出的模拟器用例。直接合并参数会漏掉这 7 条。加 `--keep-duplicates` 会收集 8,381 次，多选 18 条模拟器用例，另有 7 条重复。`-k` 按名称筛选在当前恰好选中 8,356 条，但指定方法消失时也可能成功退出。两次显式调用能发现这种漏测。 |
| 仅说明文档的 PR | 用只改根 README 的真实提交 `a06a0f8` 在本地运行当前工作流的范围判断、文档命令和链接检查。结果只要求文档检查。 |
| 普通代码 PR | 用 W5 代码提交 `e8c94a7` 作同样回放。结果要求通用测试、两组 benchmark 和两个部署任务。两次回放都向验收脚本传入模拟的任务状态，检查失败或任务意外跳过时，脚本均返回失败。 |
| 旧摘要名称是否为合并条件 | 11 月 7 日查询时，主干 `a99d035` 受保护，公开的必需检查名称列表为空，生效规则列表也为空。旧三个摘要名称没有出现在可见的必需检查中。管理员审批设置未取得。 [主干保护摘要](https://api.github.com/repos/ScratchV-Compiler/ScratchV/branches/main) · [生效规则](https://api.github.com/repos/ScratchV-Compiler/ScratchV/rules/branches/main) |

**真实 Actions 耗时与缓存。** 下面两次运行都在 11 月 2 日，使用简化前的配置。总耗时取运行创建至结束的实际经过时间。

| 运行 | 总耗时 | 通用 pytest 步骤 | Topic 9 DSL benchmark 步骤 | test 和 benchmark 的 pip 缓存 | Pages |
| --- | ---: | ---: | ---: | --- | --- |
| [PR `b7192c3`](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/41456537745) | 9 分 49 秒 | 486 秒 | 118 秒 | 精确命中旧缓存键 | 按 PR 条件跳过 |
| [主干 `a99d035`](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/41461300819) | 8 分 30 秒 | 407 秒 | 90 秒 | 按前缀恢复旧缓存 | 发布成功 |

两次运行中，通用测试任务都比并行的 benchmark 更晚完成。主干 JUnit 按测试文件累计，最慢的五组是 CNN 调度（42.51 秒）、W4 工作流（29.38 秒）、Topic28 编码与段布局（24.56 秒）、W5 命令行（20.53 秒）和外置权重 C（19.93 秒）。其中四组已经调整了重复执行或测试准备。Topic28 涵盖不同输入编码和目标平台，目前保留这些用例。后续观察应优先对比通用 pytest 步骤，再看 benchmark 中的 DSL 测量步骤。 [主干测试任务](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/41461300819/job/125330731082)

**Pages 本地验证。** 使用上述主干运行的三个原始制品，在临时目录执行当前工作流中的生成和复制步骤。新测试页读出 8,501 条通过、95 条跳过。站点中的测试页、JUnit XML 和历史数据逐字节等于对应的源文件，测试页的四个本地链接均可访问。共生成 25 个课程页面，文档检查的 394 个链接全部有效。

新配置的线上总耗时、缓存命中和 Pages 发布结果仍待观察。这批修改本身包含工作流 YAML，作为 PR 时会走普通代码检查，可查看实际任务、耗时和缓存日志。等目标主干包含新规则后，再观察只改名单内说明文档的 PR，核对实际跳过项。Pages 发布在主干成功运行后确认。本地核对数据位于 `output/ci-simplification-20261107/`。
