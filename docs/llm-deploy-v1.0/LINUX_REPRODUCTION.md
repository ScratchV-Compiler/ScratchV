# Linux 交付与独立复现约定

本文集中说明 LLM-Deploy 的 Linux 环境、固定模型资产和复现准备。W1/W2 的既有流程见第 1～4 节，W5 的本机运行与容器准备见第 5～6 节。

使用 **Ubuntu 24.04 x86_64 / Python 3.12**。环境检查通过只代表准备完成，不等于模型数值或生成结果已经通过。
Linux 是编译器、ORT、IR 解释器和 QEMU 的运行环境；目标程序仍为 RV64GC/LP64D 裸机程序。

## 1. 安装与运行前提

在仓库根目录、已选定的 Python 3.12 环境中安装一次：

```bash
python -m pip install -v -e ".[llm-deploy]"
```

所有 Python 依赖统一声明在 [pyproject.toml](../../pyproject.toml) 的 `llm-deploy` 中，
包括固定版本的 CPU PyTorch、ONNX、ORT、Tokenizer、测试工具和 Zig。
CPU PyTorch 使用官方 Linux x86_64 / CPython 3.12 wheel，安装时无需另选索引或分步安装。
旧 `requirements/qwen3-small-probe.txt` 和 `requirements/qwen3-export.txt` 仅保留兼容入口。

宿主机还需具备 Git、Git LFS 和 `qemu-system-riscv64`（Ubuntu 软件包 `qemu-system-misc`）。
这些系统工具不由 pip 安装。QEMU 应在 PATH 中；运行入口会自动发现当前 Python 环境中的 Zig。
后续文档中的 `python` 均指安装上述 extra 的解释器，命令均在仓库根目录执行。

## 2. 核对版本与环境

检出待复现 PR 的准确提交，获取 Git LFS 对象，并在记录中保存完整 SHA 和工作区状态。
有本地修改时同时保存 diff 和新增文件；仅记录 HEAD 无法确定实际执行内容。
历史提交应使用其自带指南，不能将旧提交的通过记录用于当前版本。

需要预检 Git checkout、固定依赖、CPU PyTorch、Zig/QEMU 和 LFS 模型时运行：

```bash
python scripts/check_w1_repro_env.py --require-clean --output-dir output/linux-repro-env-01
```

预检输出目录必须不存在或为空。预检通过只证明前置条件，不代表数值验收通过。
源码快照没有 Git 历史，应核对随包 manifest；不能声称通过依赖 Git 的预检或历史 benchmark。

## 3. 获取并核验固定资产

Qwen3 完整 ONNX、三个权重分片和 Tokenizer 大文件由 Git LFS 管理，固定在
`assets/qwen3-0.6b/`；配置和许可证由普通 Git 管理。来源与 SHA256 以
[固定 manifest](../../probes/w1_qwen3_export/manifest.json) 为准。

首次配置仓库时执行 `git lfs install --local`，然后执行
`git lfs pull --include="assets/qwen3-0.6b/**,probes/w1_tiny_transformer/out/tiny_transformer_2l.onnx" --exclude=""`。
切换到资产版本不同的提交后重新拉取。必须得到真实文件，只有 LFS pointer 不可运行。
完整 ONNX 约 2.24 GiB，Tokenizer 约 15.2 MiB；LFS 本地对象库与工作区各保存一份，
需为模型、解析和测试输出预留磁盘空间及数 GiB 内存。

离线验证完整 ONNX 并执行 ORT：

```bash
python probes/w1_qwen3_export/run.py --output-dir output/linux-onnx-verify-01
```

离线验证 Tokenizer：

```bash
python probes/w2_runtime/run.py --output-dir output/linux-tokenizer-01
```

W2 入口只读取上述本地资产并执行测试，不提供资产下载功能或模式参数。
小算子和两层随机模型仍在 `output/` 生成。资产缺失、未展开或哈希不匹配时验收失败，
不会联网获取或替换资产。W1 导出工具仍保留独立的 `download` / `export` 模式；W2 复现通过 Git LFS 准备资产。

## 4. 执行与留存证据

| 阶段 | 入口 | 判据 |
|---|---|---|
| W1 | [W1 README](W1/README.md) | 各项探测分别记录结果 |
| W2 | [W2 README](W2/README.md) | 七项全部 PASS；部分执行不算通过 |
| W3（含 W3 的版本） | [W3 PR #96](https://github.com/ScratchV-Compiler/ScratchV/pull/96) | 按对应版本的复现清单执行 |

每次执行使用新的输出目录，保存命令、退出码、报告、源码与资产身份，以及实际环境和计时条件。
命令失败后先检查报告与日志，不用后续成功覆盖失败，也不将跳过项记为通过。
计时比较需使用一致的线程数和系统负载；历史个人环境数据保留原始标签，不能改写为 Linux 实测。
作者再次执行或 CI 自动运行不能替代第二人独立复现。

文档检查：

```bash
python scripts/check_linux_repro_docs.py --check-bash
```

<a id="w5-local-environment"></a>

## 5. W5 本机运行环境与模型准备

本节供 W5 的三条固定生成用例和单条提示词运行使用。已有可用的 W4/W5 环境与同版本模型时，可直接回到 [W5 运行命令](W5/README.md#如何复现)，不重复创建环境、下载资产或运行模型。

| 准备项 | 要求与入口 |
|---|---|
| Python 与依赖 | 使用 Python 3.12 虚拟环境；从仓库根目录按第 1 节安装 `.[llm-deploy]`，始终使用该环境的 `python` |
| 系统工具 | Git、Git LFS、`qemu-system-riscv64`；Zig 由项目 Python 依赖提供，运行入口自动发现 |
| 固定模型 | `Qwen/Qwen3-0.6B` 的 FP32 ONNX、外置权重和同版本 tokenizer；本机准备方式见第 3 节，独立容器准备见第 6 节 |
| 资产位置 | 本机使用 `assets/qwen3-0.6b/onnx/model.onnx`、同目录权重文件及 `assets/qwen3-0.6b/tokenizer/` |
| 资源 | 复现历史 4 core 配置需至少 4 个可用逻辑 CPU；准备后至少 10 GiB 可用内存、10 GiB 剩余磁盘，初始建议预留 25 GiB 磁盘 |
| 输出 | 使用新的 `output/` 子目录；已有 ELF/bundle 可在原环境复用，不能把包含本机绝对路径的构建描述直接搬进容器 |

只检查工具是否可用时，从已激活的虚拟环境执行：

```bash
python --version
python -m pip --version
python -m ziglang version
qemu-system-riscv64 --version
git lfs version
```

这些命令不执行模型。第 3 节的 ORT 验证命令会运行模型，仅在需要该项验证时执行，不能把它当成每次生成前都要重复的环境准备。

<a id="w5-measured-environment"></a>

<details>
<summary>历史实跑的模型快照与工具版本</summary>

- 模型快照：`c1899de289a04d12100db370d81485cdf75e47ca`，见
  [模型清单](../../probes/w1_qwen3_export/manifest.json)及[冻结参考](../../probes/w5_generation/references.json)。
- QEMU 8.2.2（Ubuntu 包 `1:8.2.2+ds-0ubuntu1.18`），Zig 0.14.1；
  目标为 `riscv64 / RV64GC / LP64D / QEMU virt` 裸机，数学库为 Zig 随附 musl。
- Python 3.12.3、NumPy 2.2.6、ONNX Runtime 1.22.1、tokenizers 0.21.4。
- 以上为历史记录中的版本；完整字段见
  [数据摘要](W5/reports/validation-summary.json)的 `configuration`。不据此推断当前机器或最新依赖版本。


</details>

环境准备完成后，按 [W5 报告中的生成命令与结果判定](W5/README.md#如何复现)执行；本节不把环境检查或镜像构建当作生成成功。

<a id="w5-container-reproduction"></a>

## 6. W5 容器准备与独立复现

入口为 [部署脚本](../../scripts/deploy_w5.sh)和 [Dockerfile](../../deployment/w5/Dockerfile)。需要已安装并可访问的 Docker Engine/Buildx；部署脚本会检查引擎和资产、构建镜像，再启动生成，不负责安装引擎或下载模型，也不更改 WSL 配置。

**镜像、挂载与路径。** 容器只打包运行所需源码和依赖，资产只读挂载到 `/assets`，构建及报告
写入 `/output`，以调用者 UID/GID 运行。默认镜像 `scratchv-w5:local`、基础镜像
`ubuntu:24.04`；可显式设置 `SCRATCHV_CONTAINER_ENGINE`、`SCRATCHV_W5_IMAGE`、
`SCRATCHV_W5_BASE_IMAGE`、`SCRATCHV_W5_ASSETS` 和 `SCRATCHV_W5_OUTPUT`。
路径可含空格，挂载路径不能含逗号。脚本检查引擎和资产后构建、运行，不安装引擎，
不下载模型，不更改 WSL 或 Windows。依赖来自
[pyproject.toml](../../pyproject.toml) 的 `llm-deploy`，不另维护一份版本锁。

容器中的 Python 与 QEMU 仍承担相同职责；容器不会替代 QEMU 执行模型。生成结果和计时见 [W5 实验结果](W5/README.md#实验结果)。

<details>
<summary>独立容器复现：固定版本、环境准备、三例生成与报告回传</summary>

以下是一套需要在同一 Bash 会话中顺序执行的完整流程。准备阶段设置的目录、镜像与输出变量会被后续运行和清理步骤继续使用。

**固定版本的独立容器复现。** 以下步骤在同一 Bash 会话依次执行，使用独立的
Ubuntu 24.04 x86_64 环境，不复制作者的虚拟环境、`output/`、ELF、bundle 或报告。
三例共用一次新构建、执行 24 次前向，不先另跑 Paris，也不重新录制 ORT 参考。
固定提交 `8ad347477f7f5ba15b2da7c91178b3c6668456cc` 是已完成完整数值实跑的功能基线；
该提交用于复现既有完整数值结果，不含后续报告保存、SIGTERM 分类及 core 名称兼容修复；
这些行为另用相关小图或 Python 测试验证，不将旧完整结果标为修改后的重跑。
该固定提交也尚不支持 `--cores`，所以下面的历史复现命令保留 `--harts`；
运行当前分支时优先使用 `--cores`。这次名称兼容修改没有重跑完整模型。

| 条件 | 固定内容或要求 |
|---|---|
| 主机工具 | Bash、Git/Git LFS、Python 3.12、Docker Engine/Buildx、GNU time、timeout、setsid、tar |
| 模型 | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，FP32 ONNX 及同版本 tokenizer |
| 基础镜像 | `public.ecr.aws/ubuntu/ubuntu@sha256:5ce7043d3beb84e91bfeecacb5fafc80602bf247a1dd6016bc0fe7d69d707eca` |
| 原实跑工具 | Python 3.12.3、QEMU 8.2.2（Ubuntu `1:8.2.2+ds-0ubuntu1.18`）、Zig 0.14.1、ORT 1.22.1、NumPy 2.2.6、Torch 2.7.1+cpu |
| 原容器引擎 | Docker Engine 29.9.2、Buildx 0.38.1 |
| 准备后的资源 | 至少 4 个可用逻辑 CPU、10 GiB 可用内存、10 GiB 剩余磁盘；初始建议预留 25 GiB 磁盘 |
| 时间预算 | 单步 1800 s，总预算 16500 s，外层 280 分钟，强制清理再留 30 s |

镜像 digest 只固定基础镜像；apt 与未逐项锁定的间接依赖仍可能变化。记录实际镜像、
工具和依赖版本，先核对差异是否影响结果比较，不默认为相同环境。这里沿用原资产
身份检查，不增加源码、输出或报告哈希。保留部署前后镜像 ID 作为运行记录，
不要求两者必须相等：Buildx 的缓存构建仍可能更新 provenance/attestation 元数据，
使 manifest ID 改变。ID 变化本身不足以说明运行内容改变，也不代表版本差异可忽略；
继续记录固定源码、模型和实际工具/依赖版本，复用既有 ELF/bundle 检查。

**1. 新 clone 并固定提交。**

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

检查 `runner-resources.txt`：容器外的 Linux 环境应为 Ubuntu 24.04，Docker daemon 也运行在该 Linux 环境中；
若使用远端 daemon、容器资源配额不足或没有引擎，先停止并回报，不用修改 WSL 来凑条件。
挂载目录不能包含逗号。不要把私人凭据写入报告。

**2. 构建镜像并记录实际环境。**

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

核对 `image-tools.txt` 与上表；环境差异应保留在报告中，部署脚本本身不锁 apt 补丁。

**3. 独立获取固定资产。**

下载只调用现有 `acquire_model`，沿用 W1 manifest 中已有的下载/分片完整性检查。
它只用 Python 标准库，不执行模型、不重新导出、不在容器外的 Linux 环境安装推理依赖。
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

**4. 资源检查。** 镜像和资产准备后核对可用资源，Docker 配额不能低于要求。

```bash
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
out = Path(os.environ["SCRATCHV_W5_OUTPUT"])
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

资源不足时保留记录，不运行完整生成。

**5. 一次运行三条完整用例并保存报告。**

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
python3 -B -X utf8 scripts/check_w5_ci.py \
  --report "$SCRATCHV_W5_OUTPUT/run/report.json" \
  --reference probes/w5_generation/references.json \
  --exit-code-file "$SCRATCHV_W5_OUTPUT/runner-exit-code.txt" \
  --output "$SCRATCHV_W5_OUTPUT/result-check.json" \
  > "$SCRATCHV_W5_OUTPUT/result-check.log" 2>&1
printf '%s\n' "$?" > "$SCRATCHV_W5_OUTPUT/check-exit-code.txt"
cat "$SCRATCHV_W5_OUTPUT/result-check.log"
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
不能补造成功字段。`result-check.json` 在本地的 `commit` 可为 null；实际 SHA 由
`source.txt` 和最终 Git 记录确认。

若手动 Ctrl-C 使整段 shell 中断，重新进入同一 clone，恢复上面输出目录和镜像变量，
执行本节从“此唯一 tag”注释之后的收尾段，再打包已有报告。不要直接重新跑模型。
主机掉电、daemon 失联或强制终止可能阻止清理；如清理日志有非零码或最后仍列出容器，
标为未完成并回报，不视为通过。

**三条用例与结果判定。** 输入、应得到的文字、token 编号及停止原因统一见
[三条实际输出与冻结参考对照](W5/README.md#fixed-case-results)。复现必须逐项匹配，不能只检查是否出现城市名。

成功要求：三例、24 次前向全部完成，prompt IDs/新 IDs/文本/停止原因匹配；每步全部
`256×151936=38,895,616` 个 logits 有限且严格 `max_abs < 1e-3`；28 层完整图检查、
FMA 12 行 binary32 位模式与 FCSR 检查通过；同一 ELF/bundle 与所选策略匹配、
padding `none`；每个 QEMU 正常退出并回收，没有清理错误。套件、日志管道、报告
检查器退出码均为 0，最终无本次容器残留，源码没有改动。
`container_reproduction=NOT_ASSERTED` 是内部 CLI 不会自行宣称容器复现的字段，
不能手改；容器身份由外层构建、运行和退出记录说明。

**回传报告。** 附 `handoff.txt` 说明执行人、UTC 时间、完整 SHA、独立 clone/构建、
环境差异、三例 IDs/文本/误差、计时与缺失采样、清理结果以及失败阶段，保留：

- 来源、主机资源、镜像/依赖/工具、下载记录、所有退出码与构建/运行/清理日志。
- `run/report.json`、`result-check.json`、`run/executable.json`、`run/build/` 的 JSON。
- `run/steps/step-0000…0023/run.json`、UART 完成帧 `uart.bin`、错误日志及
  `run/fma-conformance/` 的报告。

不传权重、ELF、原始 logits 或 Zig 缓存；失败时也回传已有报告并标明缺失项。
原始产物保留至复现结果确认；重试使用新目录。沿用原小文件筛选方式打包：

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

`process-time.txt` 的 CPU/RSS 属于 Docker 客户端，不是容器或 QEMU 总量。逐步耗时、
RSS 使用 `run/steps/*/run.json`，partial 采样明确注明；未采集的 CPU 写“未独立采集”，
不填作者历史数据，也不据不同机器的记录计算加速比。

</details>
