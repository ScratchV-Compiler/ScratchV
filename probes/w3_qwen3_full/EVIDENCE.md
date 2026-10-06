# 完整 W3 结果的离线复核

复现统一采用 **Ubuntu 24.04 x86_64、Bash、Python 3.12**。先完成 [Linux 环境与资产准备](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md)，再在同一个 Bash 会话、仓库根目录执行命令；该指南设置 `SCRATCHV_PYTHON`、`SCRATCHV_CC` 和 `SCRATCHV_QEMU`。

`scripts/verify_w3_evidence.py` 对**已有七案例原始产物**重新计算误差，不执行模型、不下载资产。
它复用完整模型 runner 的 `validate_worker`、`compare_case`、`invariants`，以及安全 NPZ 读取器。

在仓库根目录，使用已有 Python 3.12 和项目锁定依赖运行：

```bash
set -euo pipefail
"$SCRATCHV_PYTHON" -B scripts/verify_w3_evidence.py \
  --evidence-dir output/downloaded-w3-run/raw/w3-full \
  --output-dir output/w3-evidence-audit-01
```

输出目录必须不存在；不会覆盖被审核的原报告。
输出为 `report.json`、`report.md`、`report.html`，退出码 `0` 表示复核通过，`1` 表示证据无效或不足。
非法参数、已有输出目录等启动错误同样返回非零。

## 所需内容与检查

需要 full runner 的完整输出目录：顶层报告，以及每个案例的 `inputs.npz`、两个后端的报告、
普通/诊断 `logits.npy`、`diagnostic_logits.npy`、`checkpoints.npz` 和 `checkpoint_schema.json`。
仅下载 reports artifact 不足以复核，必须同时保留 raw artifact 并保持目录结构。CI raw artifact 内含 `w3-full/` 前缀：指定 `gh run download ... -n w3-full-numeric-raw --dir output/downloaded-w3-run/raw` 后，证据根是 `output/downloaded-w3-run/raw/w3-full`。完整下载和报告 SHA256 获取方法见 [Linux CI 产物说明](../../docs/llm-deploy-v1.0/W3/Linux-CI与Nightly.md#产物与失败定位)。

- 七组案例完整、顺序固定，输入 dtype/shape/有限值以及每个数值均符合确定性输入生成规则。
- 模型文件描述符与固定 manifest 一致。可选 `--model-dir PATH` 进一步核对本地模型文件哈希，仍不执行模型。
- worker 报告哈希、产物大小/哈希、checkpoint schema、执行步数、耗时和 ORT 配置符合 runner 的验证规则。
- 每组普通/诊断 logits 一致，IR/ORT 对全部 256 个位置重新比较，严格 `max_abs < 1e-4`，`rtol=0`。
- 30 个检查点的 dtype、shape、有限值和误差重新计算；其误差用于诊断定位，不额外替代或改变 logits 验收门槛。
- 两个后端各自的未来 token 隔离、padding 隔离，共四项 invariant 重新计算。
- 重新计算结果必须与原报告一致；仅有 `passed: true` 不能通过。
- 解释器数值模式和策略一致，并受当前验证器支持。当前已提交 reference 契约为 `numpy-fp32-reference-v4`，所有 IR worker 的 `cpu_strategy` 必须一致，且完整 profile 的字段和值须与 `avx2-fma3` 或 `avx512` 的规范契约匹配；不是仅检查一个策略名字。

v4 复核读取生产报告保存的 `cpu_strategy`，不从审计机器 CPU 或 `SCRATCHV_FP32_REFERENCE_CPU` 环境变量推断生产策略。完整 profile 校验包括 `matmul_k_block`、`matmul_row_tile`、`matmul_row_tail`、`matmul_column_tail`、`matmul_layout` 和 `matmul_vector_policy`，不能缺字段或仅靠版本名字通过。因此可以在另一种 CPU 上核对已保存的数组，无需控制或执行本机 ORT。这里的跨机器数组核对不证明该机器实际执行模型也能通过。

当前 v4 验证器不接受历史 v2/v3 profile，须使用报告对应源码版本中的验证器，或重新生成 v4 完整证据。不能手动改旧报告的 profile、CPU 策略或源码哈希。`SCRATCHV_FP32_REFERENCE_CPU=auto|avx2-fma3|avx512` 是实际 reference 执行的选项，默认 `auto`；它不用于把旧证据转换到新契约，也不控制 ORT dispatch。

## 结论边界

`audit:w3-full-saved-evidence` 的 PASS 只说明**保存的文件和数值结果自洽**。
它不证明生产者身份，也不代表第二人独立执行模型、当前源码已运行、Linux 通过、Nightly 通过或团队验收。
本工具无法通过保存文件独立证明报告中的历史耗时与执行声明确实发生。

默认 SHA-256 检查发现文件与报告不一致；若报告和数据同时被替换，它不能充当数字签名。
可从可信渠道预先独立保留原始顶层报告哈希，然后传入：

```bash
set -euo pipefail
"$SCRATCHV_PYTHON" -B scripts/verify_w3_evidence.py --evidence-dir output/downloaded-w3-run/raw/w3-full --output-dir output/w3-evidence-audit-02 --expected-report-sha256 ORIGINAL_REPORT_SHA256
```

`source_comparison` 明确列出原报告中生产源码与当前审计源码的更改、新增、缺失。
源码不一致时，只要保存的数值契约仍受当前验证器支持，就可复核旧数组；但**不能将旧报告升级成当前源码执行的证明**。不受支持的旧 profile 仍会失败。
报告同时保存 producer 环境、审计器环境和版本化数值策略。生产环境中的 `environment.numeric_runtime.blas_environment.OPENBLAS_CORETYPE` 是环境变量原值；`numpy_openblas` 的 `status/libraries` 及可用库项的 `core_name/config` 是可选运行诊断。审计不得以本机 CPU SIMD 或 BLAS 内核替填生产环境，也不能从该环境变量推断生产库已采用某个内核。诊断不可用或历史字段缺失不豁免任何数组数值门槛。

当前已提交 v4 的[第四轮 Linux 执行](https://github.com/yuki-328/ScratchV/actions/runs/37299235103)（`1c49e8acbcd9174491b2d0c2a5a0072178ae5f9a`）已在七组完整执行、preparation 与手动汇总通过；完整 raw 数组已下载验真、离线 audit 通过。实际执行 PASS 和保存证据 audit PASS 分别记录，实际 audit 报告及 SHA256、原样保留的源码换行差异说明见 [第四轮实测记录](../../docs/llm-deploy-v1.0/W3/Linux-CI与Nightly.md)。本说明本身不替代该原始记录。
