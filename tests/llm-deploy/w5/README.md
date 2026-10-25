# W5 报告保存与容器取消测试

本目录验证生成 CLI 的报告保存边界，以及真实 Docker 容器中的外部取消和进程清理。只使用小图，不下载或运行完整 Qwen3 模型。功能、实现和实跑说明见 [W5 开发说明](../../../docs/llm-deploy-v1.0/W5/README.md)。

## 环境与约定

在仓库根目录执行命令，使用已安装 `.[llm-deploy]` 的 Ubuntu 24.04 / Python 3.12 虚拟环境。`conftest.py` 复用既有 CLI fixtures；Docker 测试复用 `tests/test_qemu_generation.py` 的小图生成器和原 `scripts/deploy_w5.sh`。

真实容器测试要求本机 Linux Docker Engine 正常运行、当前用户有访问权限，且能够构建 `deployment/w5/Dockerfile`；模型和输出通过临时目录挂载。宿主不需要额外安装 QEMU，镜像提供它及编译工具链。可通过 `SCRATCHV_W5_BASE_IMAGE` 指定可访问的 Ubuntu 24.04 基础镜像，默认使用 `public.ecr.aws/ubuntu/ubuntu:24.04`。首次构建需要下载依赖，后续复用 Docker 缓存。

测试不会安装或启动 Docker，也不修改 WSL 配置。每次使用独立镜像标签，仅清理自己启动的客户端、容器及标签。中断诊断使用临时 QEMU 启动包装器的 `-S` 暂停真实 guest，不修改生产 harness。默认单核、手动 4 hart 和 padding 关闭均保持；4 hart 的恢复阶段在容器可用 CPU 自动选择为 4 时使用 `--harts auto`。

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
| `test_docker_cancellation.py` | 默认 1 / 手动 4 hart 的真实多 token、Docker 客户端 SIGINT/SIGTERM、同 ELF/bundle 恢复 | 正常输出 `[2,3,4,6]` / `2 3 4`，四步 ORT 误差为 0；SIGINT/SIGTERM 分别返回 130/143、报告 CANCELLED；容器、宿主 PID 和 QEMU 进程组消失；恢复结果一致 |

仅报告保存边界：

```bash
python -B -m pytest tests/llm-deploy/w5/test_report_persistence.py -q -p no:cacheprovider
```

仅真实容器取消与恢复：

```bash
SCRATCHV_W5_DOCKER_TESTS=1 python -B -m pytest tests/llm-deploy/w5/test_docker_cancellation.py -q -p no:cacheprovider
```

Docker 测试在每项临时目录写入 `baseline.log`、`sigint.log`、`sigterm.log`、相应恢复日志和 `docker-cancellation-summary.json`；`deployment/<运行名>/report.json` 保留逐步数值及 QEMU 清理状态。清理断言来自独立的宿主进程和容器检查，失败后的兜底清理不能使测试变为通过。

既有 CLI 与部署编排回归仍位于原位置，可按影响范围单独执行：

```bash
python -B -m pytest tests/test_w5_inference_cli.py -q -p no:cacheprovider
```

```bash
python -B -m pytest tests/test_w5_deployment.py -q -p no:cacheprovider
```

后者覆盖引擎不可用、构建失败、参数透传，以及 info/build 阶段的取消清理；使用假引擎，不替代本目录的真实 Docker run 阶段检查。

W5 专项的日志退出状态沿用现有工作流测试，直接执行 YAML 中的 Bash，并以短小假生成器
替代模型。需要 Bash、GNU tee、`/dev/full` 和 PyYAML，不需要 Docker 或模型资产：
生成或日志任一失败必须失败，分别保留退出码；状态文件本身写失败也不能成功。

```bash
python -B -m pytest tests/test_w4_workflows.py -k 'w5_deployment_pipeline or w5_manual_job' -q -p no:cacheprovider
```

报告检查器的无模型测试覆盖有效、非零、缺失、不可读和无效的日志状态；即使生成报告
为 PASS，也不能覆盖日志失败。旧版未传日志检查参数的调用仍可用：

```bash
python -B -m pytest tests/test_w5_ci.py -q -p no:cacheprovider
```

原多轮文档中的模板对照入口保留如下，使用本地官方 tokenizer，不运行完整模型：

```bash
SCRATCHV_QWEN3_TOKENIZER_DIR=assets/qwen3-0.6b/tokenizer python -B -m pytest tests/test_conversation.py -q -p no:cacheprovider
```

它对照有/无 system 的三轮官方模板渲染和 token 序列；回复由测试预设，不能当作完整模型三轮聊天结果。依赖本地官方 tokenizer 及 Transformers。显式指定 tokenizer 目录时，缺资产会失败；未指定且默认位置缺资产时，对应官方资产测试会跳过。
