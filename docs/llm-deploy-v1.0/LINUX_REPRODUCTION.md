# Linux 交付与独立复现约定

W1/W2 的正式交付、第二人复现和 CI 使用 **Ubuntu 24.04 x86_64 / Python 3.12**。
Linux 是编译器、ORT、IR 解释器和 QEMU 的宿主环境；目标程序仍为 RV64GC/LP64D 裸机程序。

## 1. 安装与运行前提

在仓库根目录、已选定的 Python 3.12 环境中安装一次：

```bash
pip install -v -e ".[llm-deploy]"
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
