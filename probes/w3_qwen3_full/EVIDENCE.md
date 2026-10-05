# 完整 W3 结果的离线复核

`scripts/verify_w3_evidence.py` 对**已有七案例原始产物**重新计算误差，不执行模型、不下载资产。
它复用完整模型 runner 的 `validate_worker`、`compare_case`、`invariants`，以及安全 NPZ 读取器。

在仓库根目录，使用已有 Python 3.12 和项目锁定依赖运行：

```sh
python -B scripts/verify_w3_evidence.py \
  --evidence-dir output/w3-full-review-v2-01 \
  --output-dir output/w3-evidence-audit-01
```

PowerShell 可将命令写成一行。输出目录必须不存在；不会覆盖被审核的原报告。
输出为 `report.json`、`report.md`、`report.html`，退出码 `0` 表示复核通过，`1` 表示证据无效或不足。
非法参数、已有输出目录等启动错误同样返回非零。

## 所需内容与检查

需要 full runner 的完整输出目录：顶层报告，以及每个案例的 `inputs.npz`、两个后端的报告、
普通/诊断 `logits.npy`、`diagnostic_logits.npy`、`checkpoints.npz` 和 `checkpoint_schema.json`。
仅下载 reports artifact 不足以复核，必须同时保留 raw artifact 并保持目录结构。

- 七组案例完整、顺序固定，输入 dtype/shape/有限值以及每个数值均符合确定性输入生成规则。
- 模型文件描述符与固定 manifest 一致。可选 `--model-dir PATH` 进一步核对本地模型文件哈希，仍不执行模型。
- worker 报告哈希、产物大小/哈希、checkpoint schema、执行步数、耗时和 ORT 配置符合 runner 的验证规则。
- 每组普通/诊断 logits 一致，IR/ORT 对全部 256 个位置重新比较，严格 `max_abs < 1e-4`，`rtol=0`。
- 30 个检查点的 dtype、shape、有限值和误差重新计算；其误差用于诊断定位，不额外替代或改变 logits 验收门槛。
- 两个后端各自的未来 token 隔离、padding 隔离，共四项 invariant 重新计算。
- 重新计算结果必须与原报告一致；仅有 `passed: true` 不能通过。
- 解释器数值模式和策略一致，并受当前验证器支持。旧策略若已不受支持会失败，需要匹配的旧验证器或重新运行。

## 结论边界

`audit:w3-full-saved-evidence` 的 PASS 只说明**保存的文件和数值结果自洽**。
它不证明生产者身份，也不代表第二人独立执行模型、当前源码已运行、Linux 通过、Nightly 通过或团队验收。
本工具无法通过保存文件独立证明报告中的历史耗时与执行声明确实发生。

默认 SHA-256 检查发现文件与报告不一致；若报告和数据同时被替换，它不能充当数字签名。
可从可信渠道预先独立保留原始顶层报告哈希，然后传入：

```sh
python -B scripts/verify_w3_evidence.py --evidence-dir output/w3-full-review-v2-01 --output-dir output/w3-evidence-audit-02 --expected-report-sha256 ORIGINAL_REPORT_SHA256
```

`source_comparison` 明确列出原报告中生产源码与当前审计源码的更改、新增、缺失。
源码不一致时仍可复核旧数组，但**不能将旧报告升级成当前源码执行的证明**。
报告同时保存 producer 环境、审计器环境和版本化数值策略。
