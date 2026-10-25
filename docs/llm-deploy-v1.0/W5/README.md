# W5：端到端文本生成

## 本分支增加了什么

W5 将 W4 的单次 QEMU 前向接成文本生成：Host 负责 tokenizer、逐 token 输入更新、
greedy 选择与停止，QEMU 执行前向。命令行支持原始文本补全、三条固定参考用例和
多轮文本对话，Linux 一键部署脚本复用同一个入口。

- [Host 生成循环](../../../scratchv/runtime/generation.py)规定输入、输出和停止行为，
  每步只接收一个新 token，保留异常、超时和取消信息。
- [QEMU 适配器](../../../scratchv/runtime/qemu_generation.py)跨 token、跨 prompt
  复用同一 ELF 和权重 bundle，也支持已有构建的显式复用。
- [多轮对话](../../../scratchv/runtime/conversation.py)使用本地官方聊天模板，
  成功后一起保存 user/assistant 消息，失败时保持原历史。
- [命令行](../../../scripts/run_inference.py)输出逐步数值与进程记录；
  [部署脚本](../../../scripts/deploy_w5.sh)和
  [Dockerfile](../../../deployment/w5/Dockerfile)提供干净 Linux 容器运行方式。

**默认配置保持不变**：单 hart、MatMul `sequential / 1x4 / original`、Mean `sequential`、
非线性 `libm`、padding `none`。可手动指定 `--harts 1/2/4` 或 `--harts auto`；
auto 按进程可用 CPU 在构建或启动时选择 1/2/4，上限 4，不在 token 间动态调整。
完整模型通过的记录使用下文显式参考策略，不代表默认算术策略也通过同一数值要求。
W5 没有 `--qemu-workspace-layout` 参数，padding 始终关闭。

## 整体怎么实现

**生成接口与数值行为。** `generate(prompt, *, tokenizer, forward, max_new_tokens,
cancelled=None, timeout_seconds=None)` 返回不可变的 `GenerationResult`，包含 prompt
IDs、新生成 IDs、文本和 `stop_reason`。固定模型是 FP32 `Qwen/Qwen3-0.6B`，
batch=1、序列容量 256，无 KV cache。

| 项目 | 行为 |
|---|---|
| 原始文本模式 | 直接编码 prompt，不自动插入 BOS、EOS 或聊天模板；沿用 tokenizer 的 NFC 归一化 |
| 输入 | INT64 `input_ids[1,256]`；FP32 `attention_mask[1,1,256,256]`，包含 causal 与 padding mask |
| 输出 | FP32 `logits[1,256,151936]`；全部元素必须有限，包括补齐行 |
| 选词 | 取 `valid_length-1` 行的 greedy argmax，同值选较小 ID；不额外做 Softmax |
| 有效词表 | tokenizer 为 151669，模型输出宽度为 151936；选中不可解码 ID 时失败，不过滤后重选 |
| 停止 | 对新 token 按 EOS、容量、预算顺序判断；EOS IDs 为 `151645`、`151643` |
| 输入与预算 | prompt 非空且不超过 256，不隐式截断；零预算或初始上下文已满时不调用 backend |
| 异常与中断 | 不返回看似成功的部分生成；QEMU 适配器负责运行中的超时、取消和进程回收 |

每步准备当前 token 输入和 mask、运行前向并取下一个 token。复用的是构建产物；
每 token 仍启动一个 QEMU 进程并装载权重。编译、FMA 预检和 ORT session 初始化
不按 token 重做；完成后沿用 UART 完成帧及 QMP stop、dump、quit。已有构建元数据
检查继续约束 ELF/bundle 身份与所选配置。

`--verify-onnx` 为相同输入运行 ORT 并比较完整输出，不以 ORT 替代 QEMU 生成。
要求所有值有限、`max_abs < 1e-3`、`rtol=0`、greedy ID 相同，并记录 top-2 margin。
`--reference-suite` 进一步比较冻结 prompt IDs、逐 token IDs、文本与停止原因；
它不会自动选择算术策略，也不接受另传 `--max-new-tokens`。完整三例显式配置为：

```text
--harts 4 --matmul-policy blocked_fma --matmul-tile 2x4
--matmul-rhs-layout column-panel
--mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference
```

**多轮文本。** `Conversation.from_directory(tokenizer_directory, system_prompt=None)`
加载同一快照的 `tokenizer_config.json` 官方 `chat_template`，通过 Jinja2 不可变沙箱
渲染，固定 `add_generation_prompt=True`、`enable_thinking=False`。生产路径使用
tokenizers/Jinja2，不依赖 Transformers/Torch。仅支持文本 system/user/assistant，
system 只在构造时提供；每轮重新渲染完整历史并编码，不截断旧消息，不缓存 KV。

`turn(...)` 只有成功且至少生成一个 token，才同时保存 user 和 assistant；异常、取消、
超时、零预算或没有生成空间时历史不变。EOS-only 可以保存合法的空 assistant 回复。
`history` 是不可变快照，`reset()` 清除已完成轮次但保留初始 system。历史保存文本，
不承诺下一轮 token IDs 直接拼接；超过 256-token 容量会在前向前失败。单个会话拒绝
并发或重入 turn/reset。原始三例不使用聊天模板。

**部署与报告。** 容器只打包运行所需源码和依赖，资产只读挂载到 `/assets`，构建及报告
写入 `/output`，以调用者 UID/GID 运行。默认镜像 `scratchv-w5:local`、基础镜像
`ubuntu:24.04`；可显式设置 `SCRATCHV_CONTAINER_ENGINE`、`SCRATCHV_W5_IMAGE`、
`SCRATCHV_W5_BASE_IMAGE`、`SCRATCHV_W5_ASSETS` 和 `SCRATCHV_W5_OUTPUT`。
路径可含空格，挂载路径不能含逗号。脚本检查引擎和资产后构建、运行，不安装引擎，
不下载模型，不更改 WSL 或 Windows。依赖来自
[pyproject.toml](../../../pyproject.toml) 的 `llm-deploy`，不另维护一份版本锁。

部署预检和构建客户端各有独立进程组，SIGINT/SIGTERM 时转发终止并等待清理，必要时
5 秒后强制结束，返回 130/143。容器运行阶段保留 `exec ... run --init` 信号转发，
QEMU 适配器负责其子进程回收。CLI 最终保存报告失败时向 stderr 说明保存错误，
保留已有生成失败或取消的退出码 1/130/143；原本成功或仅构建成功时改为退出 1。
数值、参考文本等原始错误不会被保存错误覆盖，也不打印误导性的成功报告路径。
运行中的 SIGINT/SIGTERM 均记录为取消，分别保留退出码 130/143。

本分支的报告检查器支持 `--log-exit-code-file`：指定后要求生成与日志退出码均为 0，
日志状态缺失、不可读或内容无效也返回 FAIL。外层 Bash 分别保存
`runner-exit-code.txt` 和 `log-exit-code.txt`；生成失败优先保留原退出码，
生成成功而 `tee` 失败时返回日志退出码。下方固定 `8ad3474` 的历史复现命令保留原版
参数，仍须按其成功要求单独核对日志退出码，不向旧检查器传入新参数。

## 如何复现

**本地已有环境。** 从仓库根执行，激活项目 Python 3.12 环境并按
[Linux 开发说明](../LINUX_REPRODUCTION.md)安装依赖。已有固定资产时，原始补全命令为：

```bash
python scripts/run_inference.py \
  --model assets/qwen3-0.6b/onnx/model.onnx \
  --tokenizer-dir assets/qwen3-0.6b/tokenizer \
  --prompt 'The capital of France is' --max-new-tokens 1 \
  --verify-onnx assets/qwen3-0.6b/onnx/model.onnx \
  --harts 4 --matmul-policy blocked_fma --matmul-tile 2x4 \
  --matmul-rhs-layout column-panel \
  --mean-policy two-four-lane-last-axis --nonlinear-policy fp32-reference \
  --timeout 1800 --total-timeout 3600 --output-dir output/w5-paris-new
```

这会执行一次完整 28 层模型。只构建可加 `--build-only`，其 `BUILD_ONLY` 不表示
完成生成。复用时以 `--reuse-build output/w5-paris-new/executable.json` 替换
`--model ...`，仍传相同 tokenizer 和所有原有算术/hart 配置，并使用新的输出目录。
元数据可能含绝对路径，不直接把宿主构建搬进容器复用；跨环境从模型重新构建。

聊天模式将上例 `--prompt ...` 换为下列参数，其他模型、参考策略及输出目录参数保留：

```text
--mode chat --chat-turn 'Hello.' --chat-turn 'What is 1+1?'
--chat-turn 'Reply with one word.' --system-prompt 'Give brief answers.'
```

这会按顺序运行三轮，前两轮回答也计入后续 256-token 容量。模板一致性检查不等于
完整模型三轮生成已实跑。相关测试命令集中在
[W5 测试说明](../../../tests/llm-deploy/w5/README.md)。

**固定版本的独立容器复现。** 以下步骤在同一 Bash 会话依次执行，使用独立的
Ubuntu 24.04 x86_64 环境，不复制作者的虚拟环境、`output/`、ELF、bundle 或报告。
三例共用一次新构建、执行 24 次前向，不先另跑 Paris，也不重新录制 ORT 参考。
固定提交 `8ad347477f7f5ba15b2da7c91178b3c6668456cc` 是已完成完整数值实跑的功能基线；
该提交用于复现既有完整数值结果，不含本轮报告保存及 SIGTERM 分类修复；这些行为
使用当前分支的相关小图验证，不将旧完整结果标为本轮重跑。

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

检查 `runner-resources.txt`：宿主应为 Ubuntu 24.04，Docker daemon 也是本机 Linux；
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

**三条用例与结果判定。** 全部为 raw completion、greedy、8 个新 token，
停止原因均为 `max_new_tokens`。文本含前导空格；只出现城市名不足以判定一致，
需同时核对[冻结参考](../../../probes/w5_generation/references.json)的 IDs 和文本：

| 原始 prompt | 新生成 IDs | 新生成文本 |
|---|---|---|
| `The capital of France is` | `[12095,13,576,6722,315,15344,374,21718]` | ` Paris. The capital of Italy is Rome` |
| `The capital of Germany is` | `[19846,13,576,6722,315,9625,374,12095]` | ` Berlin. The capital of France is Paris` |
| `The capital of Japan is` | `[26194,13,576,6722,315,279,3639,4180]` | ` Tokyo. The capital of the United States` |

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

## 实际复现结果

以下完整模型数字来自 **2026-10-23 的历史实跑**，其功能基线为上述 `8ad3474`。
本轮只补异常处理与文档相关验证，没有重新执行完整模型。完整模型均为
**28 层 Qwen3-0.6B**，使用显式 `4 hart / blocked_fma / 2x4 / column-panel /
two-four-lane-last-axis / fp32-reference / padding none`。

**宿主 QEMU 三条固定生成。** 三例各 8 步，24 步的文本、IDs 与停止原因全部符合上表，
每步完整 logits 均满足严格误差要求，全部 QEMU 进程和进程组已回收。

| 用例 | 生成调用耗时（s） | QEMU CPU（s） | Host CPU（s） | 最大绝对误差 |
|---|---:|---:|---:|---:|
| France | 3375.024 | 11517.789 | 167.657 | 0.000303983688 |
| Germany | 3668.852 | 12574.174 | 190.673 | 0.000262737274 |
| Japan | 3489.655 | 11958.881 | 178.759 | 0.000197887421 |

整次测量 10540.190 s（约 2 小时 55 分 40 秒），包括准备、ORT 和数值比较。
24 步 UART 耗时合计 9990.355 s，平均 **416.265 s/token**；QEMU 完整进程合计
9998.091 s，CPU 合计 36050.844 s，平均约 360.58%（100% 表示一个逻辑 CPU）。
Host 总 CPU 540.717 s；表中生成阶段合计 537.089 s。
RSS 观测峰值 QEMU 5.406 GiB、Host 4.254 GiB、同时进程合计 7.644 GiB。
17/24 步的采样各有一次读取失败，属于 partial，不能当作无缺失峰值。

**独立单次完整前向与干净容器 Paris。** 两次都只生成 `[12095]`、文本 ` Paris`，
完整 logits 最大误差同为 `0.000172138214`。容器内重新构建一次，没有重复三例套件；
155,582,464 字节完整输出与宿主单次前向逐字节相同。

| 测量项 | 宿主单次前向 | 容器单次前向 |
|---|---:|---:|
| 全过程耗时（s） | 448.263 | 547.454 |
| UART 耗时（s） | 418.954 | 415.058 |
| QEMU 完整进程耗时（s） | 419.383 | 415.478 |
| QEMU CPU（s） | 1498.851 | 未独立采集 |
| QEMU RSS 观测峰值（GiB） | 5.398 | 5.398 |

宿主生成调用耗时 441.033 s，Host 总 CPU 25.332 s（生成期间 21.604 s），
Host/同时进程 RSS 观测峰值分别 4.119/7.210 GiB。容器总耗时含新构建、FMA 和 ORT；
运行器记录 430.481 s 包括权重传输、QEMU、后检查及清理，不含更早的输入打包、写入和
ELF/bundle 检查。容器 RSS 有 2071 个有效样本、1 次读取失败，同样为 partial。

UART 耗时包含 QEMU 启动和权重装载，不是纯算子时间；完整进程另含 stop/dump/quit。
宿主单次前向启动了新 QEMU，但未清空宿主页缓存，不宣称冷缓存性能。
容器单 prompt 的 `full_model_gate=NOT_RUN` 表示未运行三例，不表示未运行完整模型。

**小图与多轮行为。** 历史真实容器默认单 hart、手动 4 hart、auto 解析为 4 的小图均
生成 IDs `[2,3,4,6]`、文本 `2 3 4`，EOS 停止，逐步完整 logits 对 ORT 的最大误差为 0，
padding `none`。不同上下文和中断后的再次生成可复用同一配置 ELF/bundle；
历史 QEMU 超时/取消及 Docker 构建 SIGINT/SIGTERM 后未遗留目标进程。
官方聊天模板的有/无 system 三轮渲染和 token 序列与 Transformers 一致；这些回复由
模板验证预设，未表示完整模型三轮聊天实跑。

**2026-10-25 局部修复验证。** 基于 `e93233143d2323de7de6888ff7d73bdbf50784c3`
加入本轮报告保存与取消处理修复后，注入权限、磁盘满、I/O 和序列化失败，CLI 均保留原始生成错误
或取消退出码；原本成功的运行在最终报告无法保存时返回失败，不误报成功。后续调用
仍可正常保存。本轮没有重新执行完整模型，上方完整数值与性能数据仍属于历史基线。

真实 Docker 小图在默认 1 hart 和手动 4 hart 下，正常生成均得到 `[2,3,4,6]`、
文本 `2 3 4`，逐步 ORT 最大误差为 0。向 Docker 客户端发送 SIGINT/SIGTERM 后，
报告均为 `CANCELLED`，退出码分别为 130/143；宿主 PID 被回收、QEMU 进程组为空、
`--rm` 容器消失，0.3 秒后复查仍无残留。中断检查以临时 `-S` 包装器控制真实 guest
的暂停时机，不修改生产 harness；正常生成和恢复使用实际小图前向。

两种信号之后均复用原 ELF/bundle 恢复生成，IDs、文本及全部逐步 logits 仍与基线一致；
4 hart 的恢复还实测 `auto` 解析为 4，两组 padding 均为 `none`。这些结果说明小图
取消、清理和复用正常，不扩大为完整模型重跑或完整多轮聊天结果。

原始产物为本地 `output/` 内容，不提交权重、ELF 和逐步 logits：

- `output/w5-followup-20261023/qemu-reference-4hart/run/report.json`：
  完整三例；同目录 `steps.json`、`numeric-steps.json`、`resources.json` 为计时与资源细节。
- `output/w5-followup-20261023/qemu-final-single-forward-4hart/`：宿主独立单次前向。
- `output/w5-container-20261023/paris/run/report.json`：容器完整 Paris 单 token。
- `output/w5-container-20261023/small-summary.json`、`small-runs/` 与 `build-cancel/`：
  容器小图、构建取消和进程清理记录。
- `output/w5-hardening-20261025/docker-final/test_external_signals_reap_gue{0,1}/docker-cancellation-summary.json`：
  本轮 1/4 hart 小图外部取消、进程清理和恢复结果。

完整三例的独立容器运行以及其他执行人的独立复现记录仍未取得；不能把宿主三例、
容器单 token 或本轮局部验证合计成这些结果。现有实现仍是无 KV cache、逐 token
重启 QEMU 与装载权重；常驻运行、多轮完整模型语义和其他算术策略的完整结果未在
上述实跑范围内。
