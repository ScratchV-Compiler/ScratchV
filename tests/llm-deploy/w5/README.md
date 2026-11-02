# W5 报告保存与容器取消测试

本目录验证生成 CLI 的报告保存边界，以及真实 Docker 容器中的外部取消和进程清理。只使用小图，不下载或运行完整 Qwen3 模型。功能、实现和实跑说明见 [W5 开发说明](../../../docs/llm-deploy-v1.0/W5/README.md)。

## 本轮 core 兼容与 main 整合验证（2026-10-30）

本轮在整合 main 的 `013b3c7` 后，验证新旧核数接口、配置读写及相关真实小图。
四组测试互不重叠，合计 **631 passed，0 failed / error / skipped**；
核数测试文件中另有 **18 项未被本轮筛选选中**，没有将它们计入通过数。

| 范围 | 本轮结果 | 本地日志与 JUnit 文件名前缀 |
|---|---:|---|
| W4/W5 CLI、报告检查、诊断、旧构建复用和假引擎部署编排 | 480 passed | `interfaces` |
| 默认/手动/auto、新旧参数和元数据、`asdict/replace`、C/汇编宏兼容 | 82 passed，18 deselected | `cores` |
| QEMU 适配器、真实 1/4 core 多 token、不同上下文、超时/取消后恢复 | 29 passed | `qemu-generation` |
| main 带入的指令计数模块 | 40 passed | `main-integration` |

真实 1/4 core 小图均生成 `[2,3,4,6]`，不同上下文生成 `[4]`，超时和取消后的恢复生成 `[6]`。
小图完整输出与 ORT 逐元素相同，最大绝对误差为 0；复用同一 ELF/权重包，padding 关闭，进程均已回收。
本轮没有运行完整 Qwen3 模型，也没有启动真实 Docker 验证；部署编排使用假引擎。

日志、JUnit 和两份小图结果保存在本机 `output/w5-cores-publish-20261030/`，不随 Git 提交。
本节与 PR 说明提供可随仓库查看的范围和结论。旧 CI
[#40173064525](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/40173064525)
对应 `e8c94a7`，不能代表本轮修改；远端检查应查看本轮提交对应的运行。

<details>
<summary>展开本轮实际执行的相关验证命令</summary>

在仓库根目录、已满足下方环境要求时执行。使用新的输出目录保留本轮 JUnit。

```bash
python -m pytest tests/test_w4_full_probe_contract.py tests/test_w5_inference_cli.py tests/test_w5_ci.py tests/test_w4_diagnostic.py tests/test_w5_deployment.py -q --junitxml=output/w5-cores-publish-20261030/interfaces.xml
```

```bash
python -m pytest tests/test_riscv_external_harts.py -q -k 'core or auto or manual or each_hart or invalid_hart or macro' --junitxml=output/w5-cores-publish-20261030/cores.xml
```

```bash
python -m pytest tests/test_qemu_generation.py -q --junitxml=output/w5-cores-publish-20261030/qemu-generation.xml
```

```bash
python -B -m pytest tests/test_inst_counter.py -q -p no:cacheprovider --junitxml=output/w5-cores-publish-20261030/main-integration.xml
```

</details>

## 环境与约定

本文的 core 指 QEMU 模拟的 CPU 核。当前 CLI 使用 `--cores`，旧 `--harts` 继续兼容。

在仓库根目录执行命令，使用已安装 `.[llm-deploy]` 的 Ubuntu 24.04 / Python 3.12 虚拟环境。`conftest.py` 复用既有 CLI fixtures；Docker 测试复用 `tests/test_qemu_generation.py` 的小图生成器和原 `scripts/deploy_w5.sh`。

真实容器测试要求本机 Linux Docker Engine 正常运行、当前用户有访问权限，且能够构建 `deployment/w5/Dockerfile`；模型和输出通过临时目录挂载。宿主不需要额外安装 QEMU，镜像提供它及编译工具链。可通过 `SCRATCHV_W5_BASE_IMAGE` 指定可访问的 Ubuntu 24.04 基础镜像，默认使用 `public.ecr.aws/ubuntu/ubuntu:24.04`。首次构建需要下载依赖，后续复用 Docker 缓存。

测试不会安装或启动 Docker，也不修改 WSL 配置。每次使用独立镜像标签，仅清理自己启动的客户端、容器及标签。中断诊断使用临时 QEMU 启动包装器的 `-S` 暂停真实 guest，不修改生产 harness。默认单核、手动 4 core 和 padding 关闭均保持；4 core 的恢复阶段在容器可用 CPU 自动选择为 4 时仍使用旧别名 `--harts auto`，同时覆盖兼容调用。

## 运行全部本目录测试

显式启用真实容器检查；缺少引擎或权限时失败，不会将缺失条件当作通过：

```bash
SCRATCHV_W5_DOCKER_TESTS=1 python -B -m pytest tests/llm-deploy/w5 -q -p no:cacheprovider
```

未设置该变量时，报告保存测试照常执行，真实 Docker 两项会跳过；这不能作为容器清理验收结果。若要保留小图、日志与 JSON，可追加 `--basetemp=output/<新的空目录>`，注意 pytest 会清理指定的 basetemp，不要指向已有证据目录。

## 分项运行

| 文件 | 覆盖内容 | 通过标准 |
| --- | --- | --- |
| `test_report_persistence.py` | 权限、磁盘满、I/O/序列化失败；生成、超时、取消、数值/参考文本错误；中间保存恢复与下一次调用 | 原错误和退出码保留；保存失败不误报 PASS；信号处理恢复、清理执行；后续可正常保存 |
| `test_docker_cancellation.py` | 默认 1 / 手动 4 core 的真实多 token、Docker 客户端 SIGINT/SIGTERM、同 ELF/bundle 恢复 | 正常输出 `[2,3,4,6]` / `2 3 4`，四步 ORT 误差为 0；SIGINT/SIGTERM 分别返回 130/143、报告 CANCELLED；容器、宿主 PID 和 QEMU 进程组消失；恢复结果一致 |

仅报告保存边界：

```bash
python -B -m pytest tests/llm-deploy/w5/test_report_persistence.py -q -p no:cacheprovider
```

仅真实容器取消与恢复：

```bash
SCRATCHV_W5_DOCKER_TESTS=1 python -B -m pytest tests/llm-deploy/w5/test_docker_cancellation.py -q -p no:cacheprovider
```

Docker 测试在每项临时目录写入 `baseline.log`、`sigint.log`、`sigterm.log`、相应恢复日志和 `docker-cancellation-summary.json`；`deployment/<运行名>/report.json` 保留逐步数值及 QEMU 清理状态。清理断言来自独立的宿主进程和容器检查，失败后的兜底清理不能使测试变为通过。

CLI 测试还检查 `--cores` 的默认、手动和自动选择，旧 `--harts` 别名，以及新旧参数冲突；
新旧核数字段的读取与构建复用保持一致；还检查 `parse_args().harts` 的读取、双向赋值及旧包装脚本继续执行。
CLI 与部署编排测试仍位于原位置，可按影响范围单独执行：

```bash
python -B -m pytest tests/test_w5_inference_cli.py -q -p no:cacheprovider
```

```bash
python -B -m pytest tests/test_w5_deployment.py -q -p no:cacheprovider
```

后者覆盖引擎不可用、构建失败、参数透传，以及 info/build 阶段的取消清理；使用假引擎，不替代本目录的真实 Docker run 阶段检查。

执行对象的构造、`asdict()` 和 `replace()` 兼容检查沿用已有运行时测试：确认新旧核数字段等值、
没有私有字段进入结果、复制相互独立，替换核数时同步且拒绝无效值。下面只运行内存中的对象检查，不启动 QEMU：

```bash
python -B -m pytest tests/test_riscv_external_harts.py -k 'executable_constructor or executable_asdict or executable_replace' -q -p no:cacheprovider
```


W5 专项的日志退出状态沿用现有工作流测试，直接执行 YAML 中的 Bash，并以短小假生成器
替代模型。需要 Bash、GNU tee、`/dev/full` 和 PyYAML，不需要 Docker 或模型资产：
生成或日志任一失败必须失败，分别保留退出码；状态文件本身写失败也不能成功。

```bash
python -B -m pytest tests/test_w4_workflows.py -k 'w5_deployment_pipeline or w5_manual_job' -q -p no:cacheprovider
```

报告检查器的无模型测试覆盖有效、非零、缺失、不可读和无效的日志状态；即使生成报告
为 PASS，也不能覆盖日志失败。还检查报告、构建记录和每次前向中的 `cores` / `harts` 字段：新格式、旧格式及混合格式均可读取，缺失、无效或冲突值必须失败。旧版未传日志检查参数的调用仍可用：

```bash
python -B -m pytest tests/test_w5_ci.py -q -p no:cacheprovider
```

原多轮文档中的模板对照入口保留如下，使用本地官方 tokenizer，不运行完整模型：

```bash
SCRATCHV_QWEN3_TOKENIZER_DIR=assets/qwen3-0.6b/tokenizer python -B -m pytest tests/test_conversation.py -q -p no:cacheprovider
```

它对照有/无 system 的三轮官方模板渲染和 token 序列；回复由测试预设，不能当作完整模型三轮聊天结果。依赖本地官方 tokenizer 及 Transformers。显式指定 tokenizer 目录时，缺资产会失败；未指定且默认位置缺资产时，对应官方资产测试会跳过。
