# W5：PR115 第二人独立复现交接

本说明供同事在独立的 Ubuntu 环境中执行，针对
[PR115](https://github.com/ScratchV-Compiler/ScratchV/pull/115) 的固定提交
`8ad347477f7f5ba15b2da7c91178b3c6668456cc`，不跟随分支最新提交。
该 SHA 是已有完整数值验收的功能基线；本次后续提交仅修改审查超时处理、相关测试和文档，
生成实现未变，因此复现仍固定此版本，不随分支移动。
本文整理完成不表示第二人已复现；执行人回传下面的记录后才能签收。
作者再次执行或 CI 通过不能代替第二人独立复现，见
[Linux 交付约定](../LINUX_REPRODUCTION.md#4-执行与留存证据)。

## 复现范围与固定环境

使用新 clone、新容器构建和独立输出目录，只从固定源码及既有资产来源准备环境。
不要复制作者的虚拟环境、`output/`、ELF、bundle 或报告。
复用 [deploy_w5.sh](../../../scripts/deploy_w5.sh)，容器内独立构建一次 ELF/bundle；
三条用例的 **24 次前向共用同一构建**，每个 token 仍启动一次 QEMU。
不先单独运行 Paris、不重录 ORT 参考，也不额外跑 W4 完整前向。

| 项目 | 本次约定 |
|---|---|
| 源码 | 上述完整 SHA；全程工作树干净，不修改文件以消除失败 |
| 主机 | Ubuntu 24.04 x86_64；Docker Engine/Buildx 已安装且当前账号可用 |
| 主机工具 | Bash、Git/Git LFS、Python 3.12、GNU time、coreutils（timeout）、util-linux（setsid）、tar |
| 基础镜像 | `public.ecr.aws/ubuntu/ubuntu@sha256:5ce7043d3beb84e91bfeecacb5fafc80602bf247a1dd6016bc0fe7d69d707eca` |
| 容器 Python/依赖 | Python 3.12；依赖以固定提交的 `pyproject.toml[project.optional-dependencies].llm-deploy` 为准，不另安装一套推理环境 |
| 已有实跑版本 | Python 3.12.3、QEMU 8.2.2（Ubuntu 包 `1:8.2.2+ds-0ubuntu1.18`）、Zig 0.14.1、ORT 1.22.1、NumPy 2.2.6、Torch 2.7.1+cpu |
| 模型 | `Qwen/Qwen3-0.6B`；revision `c1899de289a04d12100db370d81485cdf75e47ca`；固定 FP32 ONNX、batch=1、容量 256、无 KV cache |
| 生成配置 | 4 hart；`blocked_fma / 2x4 / column-panel`；Mean `two-four-lane-last-axis`；非线性 `fp32-reference`；padding `none` |
| 资源 | 镜像和资产准备后，至少 4 个可用逻辑 CPU、10 GiB 可用内存、10 GiB 剩余磁盘；初始建议预留至少 25 GiB 磁盘 |
| 时间预算 | 预留数小时；单步 1800 s，CLI 总预算 16500 s，外层 280 分钟，超时后最多再等 30 s |

基础镜像 digest 来自原容器实际拉取和构建记录，用于固定已有制品身份，不新增哈希流程。
**它只固定基础镜像**：原 Dockerfile 的 apt 软件源和未逐项锁定的间接依赖仍可能更新。
下面记录实际镜像 ID、系统包、依赖和工具版本，并确认 suite 前后镜像 ID 一致。
若 QEMU 不再是 8.2.2、固定 Python 依赖不匹配，或系统包版本与上表不同，
先回传环境差异，由负责人确认后再运行完整套件；不要自行降级、改 Dockerfile 或宣称完全相同环境。
原有 Docker 实跑使用 Engine 29.9.2 / Buildx 0.38.1；同事记录自己的版本，版本不同须注明。
这些说明不要求修改 WSL/Windows 或安装宿主 QEMU。

**默认值不变**：默认单 hart，仍可手动选择 `1/2/4` 或 `auto`。
此交接只显式验收上述 4 hart 数值策略；不把它写成默认算术配置已通过完整模型。
`auto` 在构建时解析一次，不在 token 间调整。4 hart 也不意味着容器只获配 4 个宿主 CPU。
W5 CLI 固定 padding 关闭，**没有** `--qemu-workspace-layout` 参数。

## 1. 新 clone 并固定提交

以下命令在同一个 Bash 会话按顺序执行，从有足够空间的父目录开始。
目录名 `scratchv-w5-repro` 必须不存在；重试时改用新的目录名，保留失败记录。

```bash
set -euo pipefail
export W5_COMMIT=8ad347477f7f5ba15b2da7c91178b3c6668456cc
test ! -e scratchv-w5-repro
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout --filter=blob:none \
  https://github.com/ScratchV-Compiler/ScratchV.git scratchv-w5-repro
cd scratchv-w5-repro
git fetch --no-tags origin refs/pull/115/head
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach "$W5_COMMIT"
test "$(git rev-parse HEAD)" = "$W5_COMMIT"
test -z "$(git status --porcelain)"

export W5_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SCRATCHV_CONTAINER_ENGINE=docker
export SCRATCHV_W5_IMAGE="scratchv-w5:independent-$W5_RUN_ID"
export SCRATCHV_W5_ASSETS="$PWD/output/w5-independent-assets"
export SCRATCHV_W5_OUTPUT="$PWD/output/w5-independent-$W5_RUN_ID"
export SCRATCHV_W5_BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu@sha256:5ce7043d3beb84e91bfeecacb5fafc80602bf247a1dd6016bc0fe7d69d707eca
test ! -e "$SCRATCHV_W5_ASSETS"
test ! -e "$SCRATCHV_W5_OUTPUT"
mkdir -p "$SCRATCHV_W5_OUTPUT"
date -u +%FT%TZ > "$SCRATCHV_W5_OUTPUT/start-utc.txt"
{
  git rev-parse HEAD
  git status --porcelain
  git remote -v
} > "$SCRATCHV_W5_OUTPUT/source.txt"
{
  cat /etc/os-release
  uname -a
  lscpu
  free -b
  df -B1 .
  python3 --version
  git --version
  git lfs version
  docker version
  docker buildx version
  docker info --format 'OS={{.OSType}} Arch={{.Architecture}} CPUs={{.NCPU}} Memory={{.MemTotal}}'
  command -v setsid timeout tar
  /usr/bin/time --version
} > "$SCRATCHV_W5_OUTPUT/runner-resources.txt" 2>&1
docker info > /dev/null
test "$(uname -m)" = x86_64
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
printf 'Run output: %s\n' "$SCRATCHV_W5_OUTPUT"
```

检查 `runner-resources.txt`：宿主应为 Ubuntu 24.04，Docker daemon 也是本机 Linux；
若使用远端 daemon、容器资源配额不足或没有引擎，先停止并回报，不用修改 WSL 来凑条件。
挂载目录不能包含逗号。不要把私人凭据写入报告。

## 2. 构建同一镜像并记录实际环境

复用现有 Dockerfile；随后部署脚本会再检查构建并复用 Docker 缓存。
基础镜像和构建参数不变，不使用作者构建好的镜像。

```bash
docker build --file deployment/w5/Dockerfile \
  --build-arg "BASE_IMAGE=$SCRATCHV_W5_BASE_IMAGE" \
  --tag "$SCRATCHV_W5_IMAGE" . \
  2>&1 | tee "$SCRATCHV_W5_OUTPUT/image-build.log"
docker image inspect "$SCRATCHV_W5_IMAGE" \
  > "$SCRATCHV_W5_OUTPUT/container-image.json"
docker image inspect --format '{{.Id}}' "$SCRATCHV_W5_IMAGE" \
  > "$SCRATCHV_W5_OUTPUT/image-id-before.txt"
docker run --rm --entrypoint python "$SCRATCHV_W5_IMAGE" -m pip freeze \
  > "$SCRATCHV_W5_OUTPUT/dependencies.txt"
docker run --rm --entrypoint python "$SCRATCHV_W5_IMAGE" \
  -c 'import json; from probes.w1_qwen3_export.run import environment; print(json.dumps(environment(export=True), indent=2))' \
  > "$SCRATCHV_W5_OUTPUT/image-environment.json"
docker run --rm --entrypoint sh "$SCRATCHV_W5_IMAGE" -c \
  'python --version; python -m ziglang version; qemu-system-riscv64 --version; dpkg-query -W python3.12 qemu-system-misc' \
  > "$SCRATCHV_W5_OUTPUT/image-tools.txt"
cat "$SCRATCHV_W5_OUTPUT/image-tools.txt"
```

人工核对 `image-tools.txt` 与环境表后再继续。版本有偏差时，保留日志并联系负责人；
部署脚本本身不锁 apt 补丁，也不会替你判定偏差是否可接受。

## 3. 独立获取固定资产

下载只调用现有 `acquire_model`，沿用 W1 manifest 中已有的下载/分片完整性检查。
它只用 Python 标准库，不执行模型、不重新导出、不安装宿主推理依赖。
Tokenizer 从同一提交的 Git LFS 获取。

```bash
git lfs install --local
git lfs pull --include="assets/qwen3-0.6b/tokenizer/*" --exclude="" \
  2>&1 | tee "$SCRATCHV_W5_OUTPUT/tokenizer-acquisition.log"
python3 - <<'PY' 2>&1 | tee "$SCRATCHV_W5_OUTPUT/model-acquisition.log"
import json, os, shutil
from pathlib import Path
from probes.w1_qwen3_export.run import acquire_model
assets = Path(os.environ["SCRATCHV_W5_ASSETS"])
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
result = acquire_model(assets / "onnx")
shutil.copytree("assets/qwen3-0.6b/tokenizer", assets / "tokenizer")
(out / "acquisition.json").write_text(json.dumps(result, indent=2) + "\n")
PY
test -z "$(git status --porcelain)"
```

不要另跑参考采集脚本或添加 `--record`：suite 内部已经逐步对照当轮 ORT 与冻结参考，
无需再运行一遍完整 ORT 生成作为“预热”。

## 4. 先过三轮模板及资源预检

直接复用 CI 的两条官方模板测试，确认有/无 system prompt 的三轮历史拼接。
这两条不运行完整模型，也不代表三轮完整 QEMU 聊天已验收。

```bash
docker run --rm --init --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --env SCRATCHV_QWEN3_TOKENIZER_DIR=/assets/tokenizer \
  --mount "type=bind,src=$SCRATCHV_W5_ASSETS,dst=/assets,readonly" \
  --mount "type=bind,src=$SCRATCHV_W5_OUTPUT,dst=/output" \
  --mount "type=bind,src=$PWD/tests/test_conversation.py,dst=/tmp/test_conversation.py,readonly" \
  --entrypoint python "$SCRATCHV_W5_IMAGE" \
  -m pytest -q -c /dev/null -p no:cacheprovider \
  /tmp/test_conversation.py::test_official_three_round_rendering_matches_transformers \
  --junitxml /output/multiturn-tests.xml \
  2>&1 | tee "$SCRATCHV_W5_OUTPUT/multiturn-tests.log"
python3 - <<'PY'
import json, os, shutil, xml.etree.ElementTree as ET
from pathlib import Path
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
root = ET.parse(out / "multiturn-tests.xml").getroot()
for suffix in ("None", "Give brief answers."):
    name = f"test_official_three_round_rendering_matches_transformers[{suffix}]"
    rows = [node for node in root.iter("testcase") if node.get("name") == name]
    if len(rows) != 1 or any(rows[0].find(tag) is not None for tag in ("skipped", "error", "failure")):
        raise SystemExit(f"Official three-round template must execute and pass: {name}")
mem = {line.split(":")[0]: int(line.split()[1]) * 1024
       for line in Path("/proc/meminfo").read_text().splitlines()
       if line.startswith(("MemTotal:", "MemAvailable:"))}
evidence = {"stage": "after_image_and_assets",
            "available_cpus": len(os.sched_getaffinity(0)), "memory": mem,
            "disk_free_bytes": shutil.disk_usage(out).free,
            "minimum_available_cpus": 4,
            "minimum_available_memory_bytes": 10 * 1024**3,
            "minimum_disk_free_bytes": 10 * 1024**3}
evidence["passed"] = (evidence["available_cpus"] >= 4
                      and mem.get("MemAvailable", 0) >= 10 * 1024**3
                      and evidence["disk_free_bytes"] >= 10 * 1024**3)
(out / "resource-preflight.json").write_text(json.dumps(evidence, indent=2) + "\n")
if not evidence["passed"]:
    raise SystemExit("Insufficient W5 CPU/RAM/disk; generation NOT RUN.")
PY
```

预检检查主机可用资源；还需确认 Docker 没有更小的 CPU/内存配额。
任一预检失败即停止，不运行 suite；回传已有日志，标记 `NOT_RUN`，不自动放宽门槛。

## 5. 一次运行三条完整用例，并始终留证

整块复制执行。`--reference-suite` 自带三条用例及每例 8-token 预算，
不要同时传 `--max-new-tokens`，不要传 `--reuse-build`。
代码从固定模型重新构建一次，跨用例自动复用同一 ELF/bundle。

```bash
test "$(git rev-parse HEAD)" = "$W5_COMMIT"
test -z "$(git status --porcelain)"
test ! -e "$SCRATCHV_W5_OUTPUT/run"
date -u +%FT%TZ > "$SCRATCHV_W5_OUTPUT/generation-start-utc.txt"

set +e
timeout --signal=INT --kill-after=30s 280m \
  /usr/bin/time -v -o "$SCRATCHV_W5_OUTPUT/process-time.txt" \
  bash scripts/deploy_w5.sh \
    --model /assets/onnx/model.onnx --tokenizer-dir /assets/tokenizer \
    --reference-suite --verify-onnx /assets/onnx/model.onnx \
    --harts 4 --matmul-policy blocked_fma --matmul-tile 2x4 \
    --matmul-rhs-layout column-panel \
    --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
    --timeout 1800 --total-timeout 16500 \
    2>&1 | tee "$SCRATCHV_W5_OUTPUT/runner.log"
pipeline_status=("${PIPESTATUS[@]}")
printf '%s\n' "${pipeline_status[0]}" > "$SCRATCHV_W5_OUTPUT/runner-exit-code.txt"
printf '%s\n' "${pipeline_status[1]}" > "$SCRATCHV_W5_OUTPUT/log-exit-code.txt"
date -u +%FT%TZ > "$SCRATCHV_W5_OUTPUT/generation-end-utc.txt"

# 此唯一 tag 只供本次使用；不停止其他人的容器。
{
  containers=$(docker ps --quiet --filter "ancestor=$SCRATCHV_W5_IMAGE")
  list_status=$?
  printf 'list_exit_code=%s\n' "$list_status"
  for container in $containers; do
    docker stop --time 30 "$container"
    printf 'stop_exit_code=%s\n' "$?"
  done
  docker ps --filter "ancestor=$SCRATCHV_W5_IMAGE" \
    --format '{{.ID}} {{.Status}}'
  printf 'final_list_exit_code=%s\n' "$?"
} > "$SCRATCHV_W5_OUTPUT/container-cleanup.log" 2>&1

docker image inspect --format '{{.Id}}' "$SCRATCHV_W5_IMAGE" \
  > "$SCRATCHV_W5_OUTPUT/image-id-after.txt"
cmp "$SCRATCHV_W5_OUTPUT/image-id-before.txt" \
    "$SCRATCHV_W5_OUTPUT/image-id-after.txt"
printf '%s\n' "$?" > "$SCRATCHV_W5_OUTPUT/image-match-exit-code.txt"

python3 -B -X utf8 scripts/check_w5_ci.py \
  --report "$SCRATCHV_W5_OUTPUT/run/report.json" \
  --reference probes/w5_generation/references.json \
  --exit-code-file "$SCRATCHV_W5_OUTPUT/runner-exit-code.txt" \
  --output "$SCRATCHV_W5_OUTPUT/ci-gate.json" \
  > "$SCRATCHV_W5_OUTPUT/gate-check.log" 2>&1
printf '%s\n' "$?" > "$SCRATCHV_W5_OUTPUT/gate-exit-code.txt"
cat "$SCRATCHV_W5_OUTPUT/gate-check.log"
{
  date -u +%FT%TZ
  git rev-parse HEAD
  git status --porcelain
  free -b
  df -B1 .
} > "$SCRATCHV_W5_OUTPUT/final-resources.txt"
set -e
printf 'Preserved output: %s\n' "$SCRATCHV_W5_OUTPUT"
```

这里为失败留证临时关闭 `errexit`，**脚本走到最后不等于 PASS**。
即使 suite 非零退出，仍运行容器清理和已有报告检查器；报告缺失时 checker 应失败，
不能补造成功字段。`ci-gate.json` 在本地的 `commit` 可为 null；实际 SHA 由
`source.txt` 和最终 Git 记录确认，不冒充 GitHub Actions 执行。

若手动 Ctrl-C 使整段 shell 中断，重新进入同一 clone，恢复上面输出目录和镜像变量，
执行本节从“此唯一 tag”注释之后的收尾段，再打包已有报告。不要直接重新跑模型。
主机掉电、daemon 失联或强制终止可能阻止清理；如清理日志有非零码或最后仍列出容器，
标为未完成并回报，不视为通过。

## 6. 三条用例及共同验收标准

全部为 **raw completion**：按 prompt 原文 tokenize，不加 chat 模板，greedy，
每例新生成 8 token，停止原因均为 `max_new_tokens`。字符串包含**前导空格**。

| 用例 | 原始 prompt | 必须匹配的新生成文本 |
|---|---|---|
| France | `The capital of France is` | ` Paris. The capital of Italy is Rome` |
| Germany | `The capital of Germany is` | ` Berlin. The capital of France is Paris` |
| Japan | `The capital of Japan is` | ` Tokyo. The capital of the United States` |

新生成 token IDs 也必须与[冻结参考](../../../probes/w5_generation/references.json)完全一致：

- France：`[12095,13,576,6722,315,15344,374,21718]`。
- Germany：`[19846,13,576,6722,315,9625,374,12095]`。
- Japan：`[26194,13,576,6722,315,279,3639,4180]`。

只看见 Paris/Berlin/Tokyo 不够。`check_w5_ci.py` 必须退出 0，
`ci-gate.json` 和 `run/report.json` 都为 PASS，且具备：

- **3/3 用例、24/24 次 QEMU 前向和数值对照**；prompt IDs、生成 IDs、文本和停止原因均匹配。
- 每步比较完整 `256×151936=38,895,616` 个 logits，包括补齐行；均有限，严格 `max_abs < 1e-3`、`rtol=0`；QEMU、当轮 ORT、冻结参考逐 token 一致。
- 28 层完整图审计通过；FMA 的 12 行 binary32 位模式和 FCSR 检查全部通过。
- 所有步骤使用同一 ELF/bundle 的既有身份；所选策略完全匹配，padding `none`。
- QEMU 正常完成、退出 0、已回收，无清理错误；套件退出码为 0。

此外人工确认：官方三轮模板 **2/2 PASS、0 skip**，环境/资源预检通过，
`log-exit-code.txt` 和 `image-match-exit-code.txt` 为 0，源码未改，
`container-cleanup.log` 中命令全部成功且最后没有容器行。
任何失败、跳过或缺失证据都不能签为完整 PASS，不以放宽数值阈值或重录参考处理。

`container_reproduction=NOT_ASSERTED` 是内部 CLI 的既有字段，不要改成 PASS。
容器构建/执行日志、真实退出码和完整 suite 的通过记录共同证明本次独立容器复现。

## 7. 回传材料与签收

在 `$SCRATCHV_W5_OUTPUT/handoff.txt` 填写以下内容（可用编辑器，不改项目源码）：

| 字段 | 要填写的内容 |
|---|---|
| 执行人/日期 | 姓名或团队账号，UTC 开始/结束时间 |
| 独立性 | 独立 clone、容器构建和输出目录；是否复制作者产物（应为否） |
| 版本/环境 | 完整提交 SHA、主机/CPU/内存、Docker/Buildx、实际镜像 ID、所有已确认的环境偏差 |
| 结果 | PASS / FAIL / NOT_RUN；三例各自 token 数、文本/IDs 是否匹配、数值最大误差、模板 2/2 结果 |
| 资源与计时 | CLI 总耗时、各步 UART/进程耗时、可用 RSS 观测峰值和采样缺失情况；未采集字段写“未独立采集” |
| 清理/异常 | 容器是否清空、工作树是否干净，失败阶段、日志位置及是否有人为中断 |

**计时口径**：`process-time.txt` 的 CPU/RSS 是 Docker 客户端，不是容器或 QEMU 的总 CPU/RSS。
逐步 QEMU 耗时/RSS 使用 `run/steps/*/run.json` 中已有数据；若采样状态为 partial，
注明为观测值。现有链路没有独立采集容器总 CPU，就写“未独立采集”；
不要把作者的历史 CPU 数据填到同事报告，不据此宣称跨机器加速比。

在同一 Bash 会话执行以下命令，复用 CI 的报告扩展名白名单，并额外保留小型 UART 完成帧
`uart.bin`；排除缓存和其他二进制大制品：

```bash
python3 - <<'PY'
import os, tarfile
from pathlib import Path
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
allowed = {".json", ".xml", ".txt", ".log", ".stderr", ".stdout"}
archive = out.parent / (out.name + "-reports.tar.gz")
if archive.exists():
    raise SystemExit(f"Keep earlier evidence; archive already exists: {archive}")
with tarfile.open(archive, "w:gz") as bundle:
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if (path.is_file() and not path.is_symlink()
                and ".zig-cache" not in relative.parts
                and (path.suffix in allowed or path.name == "uart.bin")):
            bundle.add(path, arcname=str(Path(out.name) / relative), recursive=False)
print(archive)
PY
```

回传 `output/w5-independent-<run-id>-reports.tar.gz`，至少包含：

- `handoff.txt`、`source.txt`、UTC 时间、资源预检、实际镜像/工具/依赖记录、下载记录。
- `image-build.log`、`multiturn-tests.xml/.log`、`runner.log`、所有退出码、清理日志和 `ci-gate.json`。
- `run/report.json`、`run/executable.json`、`run/build/` 中的 JSON，
  `run/steps/step-0000…0023/run.json` 和 UART 完成帧 `uart.bin`/错误日志，以及 `run/fma-conformance/` 中的报告。

失败时也按同一方式回传已有文件，并说明缺失项。准备阶段失败则标记 suite `NOT_RUN`。
无需传权重、ELF、原始 logits、Zig 缓存；保留本地原始产物直到签收，若需重试则新建目录，
不覆盖旧报告。本任务不要求执行人提交代码、发起 AI 审查、操作 Actions 开关或合并 PR。
