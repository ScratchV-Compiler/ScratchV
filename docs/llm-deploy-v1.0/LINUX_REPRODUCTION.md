# Linux 交付与独立复现约定

从 W1/W2/W3 到后续阶段，正式交付、第二人复现和 CI 验收统一使用 **Linux**。
推荐 Ubuntu 24.04 x86_64、Bash、Python 3.12、CPU PyTorch 和 Zig 0.14.1。
作者可使用其他个人开发环境，但对外操作指南只提供 Linux 命令；个人路径、
工具安装位置和本机通过记录不能作为 Linux 已通过的证明。旧记录保留实际环境、
源码身份和历史链接，不能将旧平台名称改为 Linux。目标程序仍是 RISC-V 裸机程序：
这里的 Linux 是运行编译器、ORT、IR 解释器和 QEMU 的 **Host 环境**。

## 1. 获取要复现的确切版本

在独立目录克隆项目，获取待复现 PR 的准确版本。以下以 PR #94 为例；
复现 W3 时使用 96。将期望的完整提交 SHA 与 PR 页面核对，保留 Git 历史以运行旧版本 benchmark。

~~~bash
set -euo pipefail
sudo apt-get update
sudo apt-get install --no-install-recommends -y \
  git git-lfs python3.12 python3.12-venv clang lld qemu-user qemu-system-misc
git lfs install
git clone https://github.com/ScratchV-Compiler/ScratchV.git ScratchV-linux
cd ScratchV-linux
PR_NUMBER=94
git fetch origin "pull/$PR_NUMBER/head"
git switch --detach FETCH_HEAD
git lfs pull
git rev-parse HEAD
git status --short
~~~

PR 分支会更新，报告必须记录实际完整 SHA。已解压的源码快照用随包校验入口验证；
它没有 Git 历史，不能宣称执行了依赖历史的 benchmark。

## 2. 安装固定环境

以下命令均在仓库根目录执行。虚拟环境和下载资产不提交 Git。
首次安装需要网络；先安装 CPU Torch，避免解析到带 CUDA 的依赖。

~~~bash
set -euo pipefail
python3.12 -m venv .venv-linux
export SCRATCHV_PYTHON="$PWD/.venv-linux/bin/python"
"$SCRATCHV_PYTHON" -m pip install --upgrade pip
"$SCRATCHV_PYTHON" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
"$SCRATCHV_PYTHON" -m pip install -r requirements/qwen3-small-probe.txt
"$SCRATCHV_PYTHON" -m pip install -e . ziglang==0.14.1 PyYAML==6.0.3
"$SCRATCHV_PYTHON" -m pip check
"$SCRATCHV_PYTHON" -c 'import torch; assert torch.__version__ == "2.7.1+cpu" and torch.version.cuda is None'
export SCRATCHV_CC="$("$SCRATCHV_PYTHON" -c 'from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent / "zig")')"
export SCRATCHV_QEMU="$(command -v qemu-system-riscv64)"
export SCRATCHV_ZIG="$SCRATCHV_CC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
"$SCRATCHV_CC" version
"$SCRATCHV_QEMU" --version
if [[ -e .git ]]; then
  "$SCRATCHV_PYTHON" -B -X utf8 scripts/check_w1_repro_env.py \
    --require-clean --output-dir output/linux-repro-env-01
else
  printf '%s\n' 'Source archive: verify its snapshot manifest; Git/LFS preflight does not apply.'
fi
~~~

Python 命令显式使用该解释器。重新打开终端后重新设置上述环境变量，不依赖全局 Python。
环境预检输出目录必须是新目录；预检通过只证明前置条件，不代表数值验收。

## 3. 准备固定资产并选择阶段入口

W1 两层随机小模型和 W2 小算子不需要完整预训练权重。完整前端审计使用已发布 ONNX；
运行时使用固定官方 Tokenizer。已有资产默认离线校验；需要下载时显式选择 download：

~~~bash
set -euo pipefail
"$SCRATCHV_PYTHON" -B -X utf8 probes/w1_qwen3_export/run.py \
  --mode download --model-dir output/assets/onnx --output-dir output/linux-onnx-verify-01
"$SCRATCHV_PYTHON" -B -X utf8 probes/w2_runtime/run.py \
  --mode download --tokenizer-dir output/assets/tokenizer --output-dir output/linux-tokenizer-01
~~~

完整 ONNX 下载约 1.24 GB，并会解压权重、实际执行 ORT 校验。模型身份由仓库 manifest
固定，不能换成任意同名 checkpoint。W3 真实权重子图另需固定源 checkpoint；
按 [W3 PR #96 中的独立复现清单](https://github.com/ScratchV-Compiler/ScratchV/pull/96)准备，仅含 W2 的分支还没有 W3 入口。

| 阶段 | Linux 复现入口 | 判据 |
|---|---|---|
| W1 | [W1 README](W1/README.md) | 环境、MatMul、小模型、固定 ONNX 和文档分别验证 |
| W2 | [W2 README](W2/README.md) | 统一入口七项全部 PASS；部分执行不算通过 |
| W3（含 W3 的版本） | [W3 PR #96](https://github.com/ScratchV-Compiler/ScratchV/pull/96) | 准备五项与完整七组在同一版本全部通过 |

每次执行使用新的输出目录，记录源码 SHA、资产哈希、依赖和系统资源。
完整模型需要足够内存及原始数组磁盘空间；不要将小模型通过外推为完整模型容量可行。
W3 已有 Linux 实测的资源、CPU 策略和耗时见其 Linux CI 文档。

## 4. 后续提交必须遵守

- 对外复现命令使用 Bash、POSIX 路径、Linux 工具及虚拟环境 bin/python。
- 修改可执行代码后，以该提交的实际 Linux 测试为证据；本机验证与 Linux CI 分开记录。
- 不改写历史数据的平台，不把旧提交的 PASS 冒充新提交复跑，也不把离线数组审计当成重新执行模型。
- 文档检查命令如下；CI 必须执行 Bash 语法检查，缺少 Bash 不能算检查通过。

~~~bash
set -euo pipefail
"$SCRATCHV_PYTHON" scripts/check_linux_repro_docs.py --check-bash
~~~
