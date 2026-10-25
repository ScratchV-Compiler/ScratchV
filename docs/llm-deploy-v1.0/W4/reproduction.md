# W4 复现步骤

从仓库根目录执行，每次使用全新输出目录，保存源码版本、工具环境与原始产物。
完整模型执行需要较长时间和较多资源；CI 在开始时要求至少 8 GiB 可用内存、15 GiB
空闲磁盘，安装/下载后再次要求 8 GiB 可用内存和 8 GiB 磁盘。这是拒绝明显不足环境的
预检，不保证任意机器峰值都在阈值内。当前记录 guest 地址规划与 QEMU 单进程采样 RSS，
没有提供整个进程树的并发内存峰值；CI 的 GNU time 数据也不能代替该口径。

## 先确定复现的源码

独立复现优先使用维护者指定的已发布仓库地址和完整候选提交 SHA。分支名、PR 编号或
作者工作树的基线提交都不能唯一代表未提交修改。以下命令在一个新的父目录中执行，
`ScratchV-w4-repro` 必须尚不存在；两项变量由维护者提供，不在正在开发的工作树切换版本。

```bash
set -euo pipefail
: "${W4_CANDIDATE_REMOTE:?请填写维护者给出的候选仓库地址}"
: "${W4_CANDIDATE_SHA:?请填写维护者给出的完整候选提交 SHA}"
[[ "$W4_CANDIDATE_SHA" =~ ^[0-9a-f]{40}$ ]]
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$W4_CANDIDATE_REMOTE" ScratchV-w4-repro
cd ScratchV-w4-repro
git fetch origin "$W4_CANDIDATE_SHA"
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach "$W4_CANDIDATE_SHA"
mkdir -p output/w4-reproduction-identity
git rev-parse HEAD | tee output/w4-reproduction-identity/commit.txt
git status --porcelain=v1 --untracked-files=normal | tee output/w4-reproduction-identity/status.txt
test "$(git rev-parse HEAD)" = "$W4_CANDIDATE_SHA"
test ! -s output/w4-reproduction-identity/status.txt
```

这里只取源码；模型在下一节按固定清单另外获取，不依赖 checkout 时的 LFS 下载。
尚未提交的本地结果必须保留 `git.dirty`、`source_sha256`、`source_snapshot.sha256` 和
基线提交，明确标为工作树快照。`source-snapshot.zip` 是执行来源记录，须叠加在对应
Git 基线上，不能单独当作包含全部测试、CI 和依赖配置的仓库。
现有 `scripts/package_w3_repro.py` 的契约仍为 W3，并不包含 W4 workflow；不要将其
标为完整 W4 交付包。完成运行后按[回传模板](validation.md#第二人回传结果)提供结果。

## Linux 环境与固定资产

以下为 Ubuntu 24.04 / Bash 命令。系统 QEMU 包并未冻结为不可变镜像，必须保存实际
版本及每次 FMA 预检，不能仅凭版本字符串跳过验证。

```bash
set -euo pipefail
sudo apt-get update
sudo apt-get install --no-install-recommends -y python3.12-venv qemu-system-misc time
python3.12 -m venv .venv-w4
source .venv-w4/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/qwen3-small-probe.txt PyYAML==6.0.3 ziglang==0.14.1
python -m pip install -e .
python -m pip check
zig_dir=$(python -c 'from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent)')
export PATH="$zig_dir:$PATH"
export SCRATCHV_CC="$zig_dir/zig" SCRATCHV_ZIG="$zig_dir/zig"
export SCRATCHV_QEMU=qemu-system-riscv64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
mkdir -p output
python - <<'PY'
from pathlib import Path
from probes.w1_qwen3_export.run import acquire_model
acquire_model(Path("output/assets/onnx"))
PY
```

模型下载复用 W1 的固定清单，不重新导出或替换成小模型。测试命令与分层验收见
[验证说明](validation.md)。完整复现使用本节准备的 Linux 工具链，并记录实际工具版本与摘要。

## 完整运行与复核

```bash
python -B -X utf8 -m probes.w4_qwen3_full.run \
  --model-dir output/assets/onnx --output-dir output/w4-full-run1 \
  --case full_seed_0 --cc "$SCRATCHV_CC" --qemu "$SCRATCHV_QEMU" \
  --matmul-policy blocked_fma --timeout 10800
python -B -X utf8 -m probes.w4_qwen3_full.audit \
  --report-dir output/w4-full-run1 --output-dir output/w4-full-run1-audit
```

运行顺序为固定资产校验、28 层结构审计、IR 解析、外置代码生成/权重包、RV64 构建、
实际工具 FMA 预检、ORT 参考、QEMU 前向、完整数值比较。主报告必须同时满足三个完整
gate，且覆盖的 case 与请求一致。`--build-only` 的 BUILD_ONLY 不等于数值 PASS。

运行期间，`progress.json` 记录最近开始的当前阶段；`report.json` 是最后一次写入的
阶段快照，内容可能落后于正在执行的阶段。报告初始化为保守的 `status: FAIL`、
`passed: false`，因此仅看到运行中快照的 FAIL 不能判定进程已经失败。等待 runner 退出，
再结合退出码、最终 `error`/`stage`、三个 gate 和请求的 case 判断结果；完整通过须为
退出码 0、`stage: complete`、`status: PASS`、`passed: true` 且三个 gate 全为 true。
构建或 FMA 单独通过不能代表全程通过；进程已退出而最终报告缺失也不能作为成功。

`--case all` 在本地依次执行七组固定输入，需要预留七次前向与原始数组空间；CI 的
完整单例入口只接受一个固定 case。不得把一个输入通过记成七组通过。
`--reuse-build` 是受控本地诊断选项，检查旧 ELF、bundle、元数据、源码和模型身份；
显式更换 QEMU 后重新预检，复用时不能通过 `--cc` 更换编译器。干净独立复现从构建开始。

## 定位失败

```bash
python -B -X utf8 -m probes.w4_qwen3_full.fma_conformance \
  --out output/w4-fma-run1 --cc "$SCRATCHV_CC" --qemu "$SCRATCHV_QEMU"
python -B -X utf8 -m probes.w4_qwen3_full.diagnostic \
  --model-dir output/assets/onnx --output-dir output/w4-layer0-run1 \
  --checkpoint layer_0.output --case full_seed_0 \
  --cc "$SCRATCHV_CC" --qemu "$SCRATCHV_QEMU" --matmul-policy blocked_fma
```

前缀诊断按实际图依赖选择 embedding、层输出或 final norm；它不是一次完整执行中
所有中间张量的插桩记录。Host C 的 `probes.w4_qwen3_full.host_diagnostic` 可帮助区分
生成计算和目标运行问题，但不能替代 RV64 结果。FMA 预检失败时先修正工具环境，不跳过
精确检查、不清浮点标志规避、不放宽容差。全模型失败时保留原始报告与数组，再定位首个
出现明显偏差的检查点。

源码、工具、模型、输入或数值策略改变后，应记录新实验身份。保留旧失败与旧通过报告，
不要覆盖或混合阶段结果；没有重新执行的部分应明确说明。离线 audit 只核对保留数组、
输入绑定、完成状态及数字一致性，不认证报告来源，也不替代另一人独立编译执行。
