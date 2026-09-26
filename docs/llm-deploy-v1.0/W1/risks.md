# 风险清单 v0.1

> 上游文档：[开发计划.md](../开发计划.md) §5
> 状态：**W1 D1 风险工作坊的输入**，会议后升为 v1.0
> 本文档是 W1 的交付物之一，对应 CI Job `docs:risks`（要求 ≥15 条）

---

## 0. 怎么读这份清单

- **状态**列区分两类：
  - `✅ 已确认` —— 撰写本文档时通过**静态代码勘察**确认存在，附证据
  - `待验证` —— 假设，由 W1 的四项探测证实或排除
- **Plan B** 列是硬性要求：W1 出口标准规定"高风险项均有 Plan B"。空白项必须在 D4 前补齐。

**当前计数：20 条**（要求 ≥15）

---

## 1. 高影响风险（会改变 v1 方案）

| # | 风险 | 概率 | 影响 | 状态 | 触发信号 | Plan B | 责任 |
|---|---|---|---|---|---|---|---|
| R1 | **MATMUL 是占位实现**——`_select_matmul` 只做一次标量整数乘，无 m/n/k 循环，而 `ir/builder.py` 已传入 m/n/k 属性 | **高** | **高** | ✅ 已确认<br>`backend/instruction_select.py` | `probe:qemu-matmul` 4×4 数值错误 | 在 W2 重写 MATMUL lowering（三重循环 + 累加）。**这是 Transformer 最核心算子，必须第一个解决** | E3 |
| R2 | **FP32 无指令选择器**——`asm_emit` 有 `FADD_S/FMUL_S/FLW/FSW` 映射，但全后端 0 处 emit | **高** | **高** | ✅ 已确认<br>全后端 grep 无 emit 点 | `probe:qemu-matmul` 出整数结果或报错 | 二选一：<br>**(a)** 补 FP32 选择器（汇编层已就绪，成本低）<br>**(b)** 改用 FP64（`inst_select_ext.py` 已有选择器）<br>**(c)** 退到 Q16.16 定点（CNN 路径已有先例） | E3 |
| R3 | **目标 ISA 未统一**——`asm_emit`/LLVM 写 riscv64，CI 与 standalone 用 riscv32；`MachineOp` 只有 `LW/SW` 无 `LD/SD` | 高 | 高 | ✅ 已确认<br>`asm_emit.py:3` vs `ci.yml:94` | 工具链链接失败 / 指针宽度错乱 | W1 D4 拍板。取 RV64 需补 `LD/SD`；取 RV32 需修订计划 §1.3/§4.1。**不可两边都留** | E2 |
| R4 | **IR 缺 `TRANSPOSE` / `CONCAT` handler**，遇到直接 `raise ValueError` | 高 | 高 | ✅ 已确认<br>28 个 opcode 中 2 个无 handler | W2 解析 Qwen3 时抛异常 | W2 补两个 handler。若不补，attention 的 Q/K/V 转置与 GQA 的头拼接都无法表达 | E3 |
| R5 | **自托管 runner 基础设施不可用**——RISC-V 工具链不全且无免密 sudo；到 GitHub 的 HTTPS 不通 | **已发生** | **高** | ✅ 已确认<br>CI 实测报错 | CI job 起不来 | 见 [W1-执行计划.md](W1-执行计划.md) §2.2 / §7。**W1 全部 CI 验收依赖它，D1 上午必须闭环** | E5 |
| R6 | IR 无法表达注意力 | 中 | 高 | 待验证（`probe:small-transformer`） | 探测 2 无法构造 Attention 节点 | 方案见计划 §5：改用 host 端循环 + 静态图 | E2 |
| R7 | **数值误差累积**（28 层） | 中 | 高 | 待验证（W3 硬门槛） | 逐层误差工具显示误差随层数放大 | 三层验证（小/中/完整）+ 逐层对比定位误差源；必要时中间层用 FP64 | E2+E5 |
| R8 | **注意力掩码错误** | 中 | 高 | 待验证 | 输出乱码 / logits 全等 | 用 PyTorch 小模型对比验证；E5 逐层误差定位 | E5 |

---

## 2. 常规风险

| # | 风险 | 概率 | 影响 | 状态 | 触发信号 | Plan B | 责任 |
|---|---|---|---|---|---|---|---|
| R9 | ONNX 导出失败（Qwen3 → ONNX） | 中 | 高 | 待验证（`probe:qwen3-onnx`） | 导出报错或形状不符 | `torch.onnx.export` 手动导出，`dynamic_axes=None`；退而手动构造等价子图 | E1 |
| R10 | QEMU 内存不足（logits 约 148 MB + 权重） | 中 | 中 | 待验证 | QEMU OOM / 被 kill | `-m 8G` + 权重 mmap + 分块写 logits | E4 |
| R11 | **固定 L=256 的 O(N²) 计算量**——每次前向都跑满 256 长度（含 pad） | **高** | 中 | 计划已述，未入风险表 | 单次前向耗时过长，W5 无法出结果 | 缩短 L（如 64）做功能验证；或只对最后 N 个位置算 attention | E3 |
| R12 | **LM Head 规模**（151936 词表 × 1024 hidden） | 高 | 中 | ✅ 已确认（配置） | W4 单次前向超时 | 只算最后一个位置的 logits（计划 §2.4 本就如此）；必要时分块 | E3 |
| R13 | **RoPE partial 实现错误**（Qwen3 用 partial RoPE） | 中 | 高 | 待验证 | 长序列输出退化 | 小模型 vs PyTorch 逐层对比锁定 | E2 |
| R14 | **GQA 头映射错误**（16 Q 头 : 8 KV 头） | 中 | 高 | 待验证 | attention 输出形状/数值错误 | KV 头复制策略单测；对比 PyTorch | E2 |
| R15 | 初级工程师卡住 | 高 | 中 | — | 站会持续无进展 | 结对编程 + 外部导师每周 review；E2/E5 介入 | E2 |
| R16 | 接口不统一 | 中 | 高 | — | 联调时签名不符 | W1 冻结接口 + 每日站会 + PR review | E2 |
| R17 | 集成冲突 | 中 | 中 | — | 合并冲突频繁 | 每周集成日 + 小步提交 + 分支保护 | 全员 |
| R18 | W3 硬门槛（`numeric:ir-full-qwen3`）不通过 | 中 | 高 | — | W3 Nightly 红 | 顺延 1 周，W8 缓冲吸收；集中 E2+E5+E1 攻关 | E2+E5 |
| R19 | **权重获取与许可**——Qwen3-0.6B 权重下载、版本一致性 | 中 | 中 | 未评估 | 下载失败 / 版本漂移导致数值不可复现 | 固定 revision + 校验和；权重不入库，用脚本拉取 | E1 |
| R20 | **数值基准不可复现**——ORT 版本/线程数/算子实现差异导致参考值漂移 | 中 | 中 | 未评估 | 同一 ONNX 两次 ORT 结果不同 | 固定 ORT 版本 + 单线程 + 固定随机种子；参考 logits 入库并做校验和 | E5 |

---

## 3. 已确认风险的证据（便于复核）

| 风险 | 证据 |
|---|---|
| R1 | `backend/instruction_select.py` 的 `_select_matmul`：`self._emit(MachineOp.MUL, dst, a_reg, b_reg, comment="matmul: a * b")` |
| R2 | `backend/asm_emit.py:60,62,70,71` 有 `FADD_S→"fadd.s"` / `FMUL_S→"fmul.s"` / `FLW` / `FSW` 映射；但 `grep -rn "MachineOp.FMUL_S" scratchv/backend/` 仅命中 `asm_emit` 与 `machine_semantics`，**选择器里 0 处** |
| R3 | `backend/asm_emit.py:3` 写 `riscv64-unknown-elf-gcc`；`.github/workflows/ci.yml:94` 装的是 `qemu-riscv32`；`machine_types.py` 的 `MachineOp` 只有 `LW/SW` |
| R4 | IR `OpCode` 28 个取值中，`TRANSPOSE`、`CONCAT` 在 `instruction_select.py` 里无 `_select_*` 方法 |
| R5 | CI 实测：`sudo: a password is required`；`git ls-remote https://github.com/...` 挂死超时（SSH 正常） |
| R12 | 计划 §2.3：`vocab_size = 151936`，`hidden_size = 1024` |

---

## 4. W1 探测与风险的对应关系

| 探测 | 证实/排除的风险 |
|---|---|
| `probe:qemu-matmul` | R1、R2、R3 |
| `probe:small-transformer` | R6、R13、R14 |
| `probe:qwen3-onnx` | R9、R19、R12（形状） |
| 环境闭环 | R5 |

**探测结果回填方式**：探测完成后，把上表对应行的 `状态` 从 `待验证` 改为 `已排除` 或 `已确认`，并补 `触发信号` 的实际观测值。

---

## 5. 复盘检查点

W1 D5 出口评审时逐条核对：

- [ ] 20 条风险中，所有 `影响=高` 的项**都有 Plan B 且写明触发条件**
- [ ] 五项 `✅ 已确认` 风险（R1–R5）的处置方案已定，且指定了责任人与周次
- [ ] R1/R2/R3 的处置结论**已回写** [interfaces.md](interfaces.md) §4.3 与 §7
- [ ] R5 已闭环（否则 W1 CI 验收无意义）

---

## 6. 变更记录

| 日期 | 版本 | 变更 | 作者 |
|---|---|---|---|
| — | v0.1 | 初稿：计划 §5 的 10 条 + 静态勘察新增 10 条 | — |
| — | v1.0 | 待 W1 D1 工作坊确认 | — |
