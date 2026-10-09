# W4 分层验证与 CI

## 按风险阅读实现和测试

| 范围 | 实现 | 重点测试 |
|---|---|---|
| 权重身份和布局 | `scratchv/runtime/weight_bundle.py` | `tests/test_weight_bundle.py` |
| 外置 ABI 与数学策略 | `scratchv/backend/tensor_c_codegen.py` | `test_tensor_c_external.py`、`test_w4_contract.py`、`test_w4_fma_cross_backend.py` |
| 裸机运行、ELF 与进程清理 | `scratchv/runtime/riscv_external.py`、`riscv_tensor.py` | `test_riscv_external*.py`、`test_riscv_process_cleanup.py`、`test_w4_large_address.py` |
| 分片与故障证据 | `scratchv/runtime/weight_transport.py` | `tests/test_riscv_weight_transport.py` |
| 完整探测、诊断与 audit | `probes/w4_qwen3_full/` | `test_w4_full_probe*.py`、`test_w4_diagnostic.py`、`test_w4_audit*.py`、`test_w4_host_diagnostic.py` |
| CI 条件及失败关闭 | `.github/workflows/w4-*.yml` | `tests/test_w4_workflows.py` |

测试文件的完整路径均在 `tests/` 下。完整模型数值由探测入口实际执行，不是单测中人工
小图或伪造的报告 fixture。可先执行以下回归，再按[复现步骤](reproduction.md)运行完整模型：

```bash
python -B -X utf8 -m pytest -q \
  tests/test_weight_bundle.py tests/test_tensor_c_external.py \
  tests/test_riscv_external*.py tests/test_riscv_process_cleanup.py \
  tests/test_riscv_weight_transport.py tests/test_w4_*.py \
  tests/test_tensor_boundary_contracts.py tests/test_tensor_return_contract.py \
  tests/test_w1_qwen3_export.py tests/test_llm_inputs.py \
  tests/test_w2_runtime_metadata.py tests/test_w3_full_gate.py \
  tests/test_licm_safety.py tests/test_optimizer_numeric_semantics.py
```

真实 QEMU、FMA、跨 4 GiB 地址、跨后端及跨 FP32 元素边界的分片测试应实际执行。
工具缺失导致的 skip 不能作为这些覆盖已完成的证据；平台特有信号/session 与文件权限
测试应在适用环境补测。

## 门槛不能互相替代

W1/W2 小模型 IR 及算子门槛按各自定义保留；W3 完整 IR 为严格 `max_abs < 1e-4`，
W4 完整 QEMU 为严格 `max_abs < 1e-3`，均保持各自报告的输入范围。W4 对全部
38,895,616 个 FP32 logits 比较，`rtol=0`，不接受 NaN/Inf、shape/dtype 错误、缺失输出
或等于阈值。FMA 指令语义使用精确位/FCSR 判断，不使用模型容差。

离线 audit 同时核对用例名称对应的固定 token、mask 和有效长度，不能仅更新文件哈希
就把另一组输入计入该用例。W3 独立探测和 W4 完整入口在成功前复核执行源码指纹；
运行中源码变动或无法读取时拒绝发布成功。W4 仅构建入口也执行该复核。

## Workflow

`w4-full-numeric.yml` 的 unit job 运行合同、小图真实目标及跨阶段低成本回归。完整 job
依赖 unit 成功，仅手动或被调用时执行单个固定 case；普通 PR 不执行完整模型。
完整 job 依次要求 build、实际工具 FMA、QEMU 完成、全 logits 数值和原始数组 audit，
并检查报告与请求 case、提交和工具身份一致。失败、取消、跳过不能替代成功。

`w4-nightly.yml` 通过仓库内 `uses` 调用同一提交的完整 workflow，默认仅运行
`full_seed_0`。**所有 Nightly 入口均需仓库变量 `W4_NIGHTLY_ENABLED=true`**，否则不执行
昂贵任务；直接手动运行 `w4-full-numeric.yml` 不受此变量限制。定时配置存在或本地测试
通过，不等于已经启用定时执行或取得远端成功记录。汇总成功也不代表团队已验收。

W4 不自动重跑昂贵的 W3 完整七例。发布评审应另外核对同一候选源码的 W3 证据；变更
共享解析、IR、解释器或数值策略时，应补对应层级回归，不能只用 W4 较宽门槛覆盖 W3。

完整 QEMU 截止为 10800 秒，CI 的完整 runner 外层软截止 195 分钟、30 秒强制回收宽限，
step 200 分钟、job 240 分钟。正常失败上传已有报告与日志；宿主丢失、强制取消或整体
超时可能阻止上传，产物缺失按未完成处理。

## 证据与保留

- 报告包保留源码快照、摘要、工具/依赖、构建源、布局、清单和阶段错误；原始数值包
  另外保留输入、ORT、QEMU output、UART 完成帧及 FMA 观测。
- CI 报告包保留 30 天、原始数值包 3 天。需要独立审计时，在到期前保存原始数组；
  只下载摘要不足以重算误差。
- 不上传大权重 bundle 或运输副本；独立复现按固定清单重新获取模型。
- `source-snapshot.zip` 仅覆盖报告记录的源码和固定配置，需结合其 Git 基线恢复，
  不是完整独立仓库归档。
- 原始报告、工具下载和运行产物保存在忽略的 `output/`，不随源码提交发布；验收材料
  应提供单独的产物获取位置和保留期限，公开摘要不能代替原始数组。

## 第二人回传结果

复现者可直接复制以下模板回传；各项填写实际结果，未执行写“未执行”，不要预填通过。
最终验收由团队另行确认。直接重算作者提供的数组属于离线 audit，独立复现应由复现者
在指定候选源码上重新构建并执行，不能仅复用作者的 ELF。

```text
复现者 / 日期：
源码：候选完整 SHA、工作树是否干净；若为未提交快照，附基线 SHA、源码清单和快照 SHA256：
环境：宿主 OS/架构、Python/依赖、Zig/QEMU 实际版本及二进制 SHA256（见主报告）：
执行：完整命令、case、MatMul 策略；是否独立重新构建并执行 QEMU：
单测：通过/失败/跳过数，tests.xml 位置；每项相关跳过原因：
完整门槛：build:riscv-full / smoke:qemu-full / numeric:qemu-full-qwen3：
数值：elements_compared、max_abs、atol、rtol；FMA 精确检查和离线 audit 结果：
资源：ELF 字节数、guest 布局、qemu_wall_s、采样 QEMU RSS（不填成进程树峰值）：
证据：主 report.json、source-snapshot.zip、模型/权重摘要、FMA 报告、audit 报告、原始数组包位置及保留期限：
关联证据：同一候选的 W1/W2 回归与 W3 精度结果，未执行项明确列出：
失败或不足：stage/error、退出码/超时、最后日志、相关 run.json/transport.json；缺失文件及原因：
结论：本次实际覆盖范围和未完成项；不代替团队签字或 Nightly 记录：
```

字段以主报告为索引：`git`/`source_sha256`/`source_snapshot` 记录源码；`environment`、
`execution_tools`、`execution_tool_binary_sha256` 记录环境；`model_files` 和 `weight_sha256`
记录模型与 bundle 身份。数值及运行状态在 `cases[*].numeric` 和 `cases[*].qemu`。
工具版本字符串相同仍需保留二进制摘要；不同平台的二进制摘要可以不同，分别通过实际工具的
FMA 预检并满足相同门槛，不要求跨平台 ELF 或全部 logits 逐位一致。

成功时保留整个探测输出目录以及独立的 audit 目录，分享时按上述 CI 保留规则去除大
权重和运输副本，保留相对目录结构。失败时先保留已生成的 `report.json`、`progress.json`、
控制台日志及 case 下的 `qemu/run.json` 和运输清单（如存在），再使用新目录重试。
没有生成原数组或主报告时如实注明，不能用一次重试的 PASS 覆盖首次失败。
