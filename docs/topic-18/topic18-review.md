# 分支评审报告 — pr-59「指令调度器启用与安全修复」（Mastttttter）× 对照 topic_18

> 评审日期：2026-09-14 第一轮；2026-09-15 第二轮修订  
> 评审对象：`pr-59`（第一轮 HEAD `c51200e`；第二轮 HEAD `035d180`，含新增 `dcf5f09`「feat(benchmark): add instruction scheduling A/B reports」与 upstream/main 合流；特性提交 `2c71bd8`「feat(backend): implement safe local instruction scheduling」，2026-09-11；文档提交 `ad9c806`「upd: rename doc」，2026-09-14；作者 Mastttttter）  
> 对照分支：`topic_18`（`742a561`，2026-09-13，作者 Seven Gao，「rewrite instruction scheduler with intra-block list scheduling」）  
> 共同基线：`main` `73c3926`（第一轮）；第二轮合流点 `main` `99538fe`；改动规模：pr-59 vs 第一轮 main = 16 files, +3180/−690（其中特性提交 12 files, +1670/−690）；topic_18 vs merge-base = 3 files, +1508/−559；第二轮新增基准提交 6 files, +731/−5  
> 评审人：opencode（审查角色：架构师 + 工程师双视角）  
> **第二轮修订说明**：本轮把验证目标从「调度是否正确」扩展为「调度是否真的有效」。新增 §4.5（llvm-mca 独立微架构验证、全量真实负载覆盖审计、独立解释器差分执行、发射形式静态对照、新增 Benchmark 文档逐项复现、topic_18 断言复核）；相应更新 §0、§5（新增 F11–F16）、§7、§8、§9，并在 §10 给出「值得合入、但需先修 4 项 P1」的最终结论。第一轮正文保留。  
> 本轮证据产物：`/tmp/opencode/t18/{mca,coverage,semantics,topic18}/`，全部命令与原始 JSON 可复现。

---

## 0. 结论速览

| 对象 | 结论 |
|------|------|
| **pr-59（主评审对象，正确性与工程）** | **有条件合入**。未发现正确性阻断级缺陷；安全设计（语义精确分类 → 区域切分 → DAG → 独立验证器 → 收益门控 → 严格模式 → 不写部分输出）是本仓库同类 pass 中最完整的一次。第二轮独立验证：140 对差分执行 0 语义失败、全量 848 测试通过。合入前必须处理 4 项 P1：删除误提交文件 `:memory:.ses`（F1）、同步 topic-18 文档状态（F2）、修正 div/rem 成本模型（F11）、修 linear 分支目标发射（F12）。 |
| **pr-59（有效性，第二轮新增）** | **仅模型内有效，真实收益远低于声称且可能为负**。llvm-mca 在合成语料上只兑现模型声称的 **8.1%**（rocket-rv32；40/72 region 反而变慢），sifive-e76 整体 **-236 拍**；真实负载 24 个输入只有 3 个产生换序，动态指令数 0 变化。根因：div 模型非阻塞（16 拍）与真实（rocket 33 拍且阻塞发射、e76 66 拍）不符，mul 3 拍偏低。`saved_cycles` 只能当方向上界，不能当硬件收益。 |
| **topic_18（对照）** | **不建议作为合入基线**。成本模型存在实质偏差（非流水除法阻塞整条流水、周期数不计末条指令延迟），并且不识别本仓库后端的 `base(offset)` 存储语法，在真实 `cnn.onnx` 产物上 12/59 条指令被迫钉死；验证器与集成强度也弱于 pr-59。第二轮复核 T1/T2/T3/T6 全部成立（单条 mul 报 1 拍、`div+独立 add` 17 拍/15 stall、12 条 spill store 钉死、benchmark 用非物理寄存器且 3 链 0% 改善）。其操作数宽容策略与 `.label` 处理可作为参考。 |
| **Benchmark 文档（`dcf5f09`，第二轮新增）** | **可信**。`docs/topic-18/18-指令调度器Benchmark.md` 的固定用例、CNN 静态 A/B（`not_modeled`）、合成规模、CI 集成的全部数字与声称逐项复现一致；口径明确不声称硬件加速、不把 CNN 解释为加速、不用 TinyFive 证明周期下降，属于少见的诚实报告。仅 `.venv/bin/python` 与 standalone 生成命令两处本地运行说明有误（F15）。 |
| **共性结论** | 两者都是「文本层 V1」，均未兑现 topic-18 设计文档要求的 post-RA 结构化调度；`--schedule` 均保持默认关闭，主收益为局部静态模型估算（非硬件周期），不减少动态指令数。**本 topic 对「动态指令数 vs LLVM 4.2x」这一项目核心指标贡献为 0，不应计入性能达标项。** |

---

## 1. 特性概述

### 1.1 "enable" 与 "fix" 的具体含义

pr-59 的改动可以拆成两件事：

**（a）enable：把调度器真正接进生产编译路径并加装控制面**

- `scratchv/compiler.py:490-524`：`_run_asm_passes` 改为调用 `schedule_assembly(asm_text, ScheduleConfig(strict=...))`，把 `stats["schedule"]`（含逐区域明细与 diagnostics）写进 `CompileResult.stats`，warning 级诊断进入 `CompileResult.warnings`；`scratchv/compiler.py:244-247` 增加配置校验（`--schedule` 仅限 riscv 后端；`--schedule-strict/--schedule-report` 必须伴随 `--schedule`）；`scratchv/compiler.py:336-340` 捕获 `ScheduleError` 并以编译失败返回，**不写部分输出**。
- `scratchv/main.py:95-99`：新增 `--schedule-strict`（验证失败即失败）与 `--schedule-report`（打印模型估算报告）。
- 旧生产接线（`main`）是 `parse_instructions → build_dag → schedule → "\n".join(f"  {op} " + ...)` 的文本重排，会**破坏汇编**。在 `main` 实测 `--schedule`（`benchmarks/cases/017_while_sum.dsl`）产出：`.text/.globl` 变 `text/globl`、`main:`/`.entry:` 等标签全部丢失、`ret` 行丢失换行、分支顺序错乱——不可汇编。pr-59 用「保留原始行 + 按槽位换行」的原地重写替代了它（`inst_scheduler.py:437-439`）。

**（b）fix：按实现计划重写调度核心（`docs/topic-18/18-指令调度器实现计划.md`）**

- 新增 3 个模块，语义/模型/验证与算法分离：`schedule_semantics.py`（一型一解析、未知形式一律 barrier）、`schedule_model.py`（单发射资源占用模型 + 顺序敏感估算）、`schedule_verify.py`（独立于建图的验证器）。
- 依赖图从「仅 RAW/WAW、无内存/控制边界」（设计文档 §16.2 缺陷 4）补齐为 RAW/WAR/WAW + memory 全保序 + FP flags + control/terminator 固定（`inst_scheduler.py:68-106`）。
- 优先级计算修正（反向拓扑 + 边距离，`inst_scheduler.py:114-120`），测试给出精确高度 `[6,4,1]`。
- `machine_instrs_from_scheduled` 不再把未知 opcode 回退为 `MV`，改为显式抛错（`inst_scheduler.py:462-493`，修复设计 §16.2 缺陷 8）。
- 收益判定从「与顺序无关的延迟求和」改为共享模型的顺序敏感估算（`schedule_model.py:78-101`），并要求**严格更低**才应用（`inst_scheduler.py:416-419`）。
- 重写 `benchmarks/bench_inst_scheduler.py`：改用合法物理寄存器、走生产入口 `schedule_assembly`、明确声明「本地静态模型估算，非硬件周期」。

### 1.2 模块清单与公共接口

| 文件 | 行数 | 职责 | 关键接口 |
|------|------|------|----------|
| `scratchv/backend/schedule_semantics.py` | 294 | 指令语义：`describe()` 精确匹配 30+ 种 RISC-V 形式；`SchedInst` 不可变（frozen），def/use 由 opcode 重算而非调用方提示 | `describe`、`SchedInst`、`ScheduleError` |
| `scratchv/backend/schedule_model.py` | 101 | 单发射教学模型：ALU/load/mul/div/FP 延迟与资源占用；顺序敏感 `estimate_order` | `ScheduleModel`、`estimate_order` |
| `scratchv/backend/schedule_verify.py` | 51 | 独立验证：身份/置换/边界/固定指令/读取来源与最终写值/访存·FP 副作用顺序/DAG 边序 | `verify_schedule` |
| `scratchv/backend/inst_scheduler.py` | 534 | 区域切分、DAG、列表调度、收益门控、原地重写、统计与诊断、CLI | `InstructionScheduler`、`parse_instructions`、`ScheduleConfig`、`ScheduleResult`、`schedule_assembly` |

集成面：`scratchv/backend/__init__.py` 导出 3 个新符号；`compiler.py` / `main.py` 见 §1.1；测试新增 `tests/test_inst_scheduler_integration.py`（279 行）与 `tests/test_inst_scheduler_safety.py`（314 行），并重写 `tests/test_inst_scheduler.py`。

### 1.3 topic_18 对照实现概览

单文件 1065 行自包含实现：`OpcodeInfo` 表 → `classify_operands` → `split_blocks`（含 `.label` 伪标签）→ RAW/WAR/WAW + 保守内存 chain 构建 → 列表调度 → 顺序敏感 scoreboard 估算 → `verify`（置换 + 边序 + 末条 pinned）→ `schedule_asm` 原地重建。集成方式为 `compiler.py` 的 `_asm_pass_stats` 属性 + `ScheduleConfig()`（strict 未接线），保留 legacy API。

---

## 2. 评审材料与方法

### 2.1 阅读清单

- pr-59：`inst_scheduler.py`、`schedule_semantics.py`、`schedule_model.py`、`schedule_verify.py`、`compiler.py`（diff 与现场代码）、`main.py`、`benchmarks/bench_inst_scheduler.py`、3 个测试文件、`docs/topic-18/` 4 份文档、`.github/workflows/ci.yml`、`git diff main...pr-59` 全量。
- topic_18：worktree `/tmp/opencode/topic18-wt`，`inst_scheduler.py` 1065 行、`compiler.py` diff、`tests/test_inst_scheduler.py` 617 行、原 benchmark。
- 共同依赖：`scratchv/backend/_asm_parser.py`、`riscv_encoder.py`（确认 `max`/大立即数 `li`/分支立即数为多指令展开伪操作；确认 `base(offset)` 是后端合法语法）。

### 2.2 执行的命令与结果

| 命令 | 结果 |
|------|------|
| `python3.11 -m pytest tests/test_inst_scheduler*.py -q`（pr-59） | **105 passed, 19 skipped**（skip 全部为 `clang` + `qemu-riscv32` 缺失） |
| `python3.11 -m pytest tests/ -q`（pr-59） | **760 passed, 19 skipped**（全量无回归） |
| topic_18 worktree：`pytest tests/test_inst_scheduler.py -q` / `pytest tests/ -q` | **45 passed** / **594 passed** |
| `python3.11 -m benchmarks.bench_inst_scheduler --repeats 5`（pr-59） | 10→1000 条：saved 5→430 cycles（约 14%→19% 模型节省）；5000 条区域超限整体跳过 |
| topic_18 自带 benchmark | 3 依赖链全部 **0% 改善**；1 链 9.5%；5000 条 2.4s |
| 对照语料（同一批 400 个随机直线块，逐字节同一 seed） | 变更率：pr-59 **332/400**（含 div/rem）、**95/400**（无 div/rem）；topic_18 **16/400**、**0/400** |
| 独立 RV32I 解释器对拍（`/tmp/opencode/sem_check.py`，整数 + 访存 + 符号，含 div/rem/内存读写） | pr-59 **400/400 语义一致**（含 332 次真实换序）；topic_18 **400/400 语义一致**（16 次换序）；两分支均无失败 |
| 随机混合输入 fuzz（3000 例，含宏/条件汇编/CRLF/畸形操作数/多 section） | pr-59 无异常泄漏、无行丢失、strict 仅抛 `ScheduleError` |
| 真实负载 `models/graph/cnn.onnx --schedule` | 基线产物两分支逐字节相同；两分支调度后产物**也相同**（3 处 `li`/`mv` 互换）；pr-59 建模 49/59、模型 59→55 cycles、moved 6；topic_18 将 12 条 `sw sp(...)` 钉死并刷 12 条 warning |
| `main` 对照：`--schedule` 产出 | 标签/指示符丢失、顺序错乱、不可汇编（见 §1.1，证明"修复"对象真实存在） |
| `git diff --shortstat main...pr-59` | 16 files changed, +3180/−690 |

环境说明：仓库 `.venv` 为 Python 3.8（与本项目 3.12 目标不符，`tuple[str, ...]` 直接报错），本次统一用 `/usr/local/bin/python3.11`；本机无 `clang`/`qemu-riscv32`，因此两个分支的「真实汇编 + 执行对拍」用例全部 skip（见 F10）。CI（`.github/workflows/ci.yml`）同样未安装这两个工具。

### 2.3 独立验证脚本（可复现）

- `/tmp/opencode/diff_bench.py`：同一生成器驱动两个实现的对照基准；
- `/tmp/opencode/sem_check.py`：自写 RV32I 子集解释器（add/sub/mul/div/rem/逻辑/移位/比较/li/lui/mv/LW/SW/分支/标签），对调度前后产物做终态寄存器 + 内存逐项比对；
- `/tmp/opencode/one_case.py`、`one_case18.py`：单例分歧定位；
- worktree：`/tmp/opencode/topic18-wt`、`/tmp/opencode/main-wt`（评审结束已清理，ScratchV 工作区保持干净）。

---

## 3. 架构师视角评审

### 3.1 分层与模块边界（优）

pr-59 把原单文件、985 行、自带解析环（`parse_instructions → build_dag → schedule → 文本重拼`）的调度器，重构成单向依赖的四层：

```
schedule_semantics (指令语义，唯一解析点)
      ↑                       ↑
schedule_model (时序/资源)   schedule_verify (独立校验)
      ↑
inst_scheduler (区域/DAG/列表调度/重写/统计)
```

- 语义对象不可变（`frozen=True`），def/use **必须**从 opcode 重算（`schedule_semantics.py:276-282`），从根上消灭"调用方提示覆盖真实语义"与"按操作数位置猜 def-use"两类旧缺陷；未知形式返回 `barrier_reason` 而不是猜。
- `schedule_verify` 的检查项独立于建图：除 DAG 边序外，还独立比较**每次寄存器读取来源**（`(id, reg) → 生产指令 id`）与**最终写值归属**、访存与 FP 副作用序列、pinned/terminator 不动（`schedule_verify.py:35-51`）。即使建图漏了某条依赖，读来源比较仍能拦截——这是比"用同一张图验证同一张图"明显更强的闭环。
- 这是 V1 文本入口下能做到的合理边界；未来若按设计文档迁到 post-RA 结构化 `MachineInstr`，`schedule_model`/`schedule_verify`/调度算法可原样复用，只需替换 `SchedInst` 适配层。**可演进性良好**。

### 3.2 与 topic-18 设计文档的架构对齐度（偏）

设计文档（2026-07-25，Proposed v0.1）要求：post-RA/pre-emission、结构化 `MachineInstr`、输入不解析汇编、验证失败 fail-stop、拒绝 `--reg-alloc linear --schedule`。pr-59 实现的是**计划文档**的收缩版 V1：文本入口、默认 `linear` 也可调度、失败默认局部回退 + strict 才失败。两者是显式记录的范围调整（计划 §1 自认"调整原设计"），Review 文档也支持安全降级。架构上可接受，**但代价是设计文档的验收门槛全部未达成**，且 4 份文档互相冲突（见 F2）。结论：V1 选择务实、方向无误；需要把"这一版是计划 V1、设计仍是 V2 backlog"写清楚，避免后续以设计文档评估本实现时产生误判。

### 3.3 安全体系（优，本仓库同类最强）

pr-59 在五个层次上做了防御，且每层都有测试锚点：

1. **解析层**：`describe()` 只接受精确支持的形式（寄存器别名校验、立即数范围、分支目标必须符号/数字标签、store 两种语法）；`li` 仅小立即数、`jal/jalr/call/auipc/la/lr/sc/amo/csr` 等一律 barrier —— 正确识别了后端多指令展开伪操作（`max` 展开为 6 行、大 `li` 展开为 LUI+ADDI、分支立即数展开为 2 条），从不假设"一行=一条指令"。
2. **布局层**：`_unsafe_layout` 对宏/条件汇编/行内分号/续行/当前地址表达式/数字控制流目标做**整文件熔断**（`inst_scheduler.py:287-335`），宁可零收益也不冒险。
3. **区域层**：标签、指示符、barrier、terminator 全部切区，跨区调度在 `build_dag` 直接拒绝（"Multiple regions"）。
4. **验证层**：候选顺序先过 `verify_schedule` 才允许落盘；收益不严格改善则保留原序；失败恢复整区（strict 则整个编译失败且不写输出）。
5. **集成层**：后端/参数组合先校验；`CompileResult.stats/warnings` 可观测；失败路径不产生部分文件（集成测试 `test_stats_are_per_compilation_and_strict_failure_leaves_output_intact` 断言旧输出文件未被覆盖）。

### 3.4 可观测性与可演进性（良）

- `ScheduleResult.stats` 提供 `input/modeled/moved/applied/skipped/original_cycles/final_cycles/saved_cycles/regions[]`（含逐区依赖计数与 issue cycles），`--schedule-report` 打印人类可读摘要，`diagnostics` 带行号与原因；模型名、override、估算口径（"以就绪输入为前提的局部静态估算"）都写进了 stats，避免把模型数当实测吹。
- 模型可注入 override（`ScheduleModel(overrides=...)`），测试用它验证模型共享与拷贝语义。
- 遗留 API（`InstructionScheduler.estimate_cycles/report`、`machine_instrs_from_scheduled`）保留但语义收紧为 fail-loud，兼容旧测试的同时移除静默回退。

### 3.5 架构层面问题

- **覆盖率与可观测性错配（F3）**：真实产物中 `max`、`slt rd,0,rs`、`mv rd,0`、`slt rd,rd,1` 都是后端合法形式（`_reg_num` 把未知/字面量静默映射为 x0），却被判 barrier；`cnn.onnx` 上 10/59 未建模。这些拒绝只是 `info` 级诊断，默认 CLI 完全不显示，用户会误以为"整份文件都调度过了"。
- **整文件熔断粒度过粗（F5）**：任意一处 `.macro`/`;`/续行导致全文件零调度，`skipped_regions=1` 且诊断固定指向第 1 行。
- **模型是教学模型（F6）**：单发射、资源占用、无 cache/分支预测/转发细节；报告已明确免责，但在硬件校准前不应默认开启（当前 `--schedule` 默认关闭，符合设计）。

---

## 4. 工程师视角评审

### 4.1 代码质量（良~优）

- 类型注解完整，docstring 说明设计意图与 LLVM 参考位置（`ScheduleDAGInstrs::addPhysRegDeps` 等）；命名达意（`barrier_reason`、`occupancy`、`edge_kinds`）。
- 列表调度实现（`inst_scheduler.py:122-177`）用「时间有序 pending 堆 + 每资源优先级堆」，`min(choices)` 三元组 `(-priority, ready_time, index)` 唯一且确定；除法按 `occupancy=latency` 占用资源但不阻塞其它资源——比 topic_18 的"整条 EX 停 16 拍"更贴近真实单发射多流水线结构，且 `div+ALU` 场景模型为 16 拍（topic_18 为 17 拍）。
- 原地重写用 `output[slot.id] = inst.raw_line + lines[slot.id].ending` 把换行按槽位保留（含 CRLF、末行无换行），三个参数化用例覆盖 `\n`/`\r\n`/末行无换行。
- 收益门控、幂等性（重跑输出不变）、确定性（同输入两次调度逐字节一致）都有断言。
- 小瑕疵：`_unsafe_layout` 的 `.` 正则对字符串常量里的 `;`/`\` 会误报（安全方向）；`InstructionScheduler(latency_model={}, model=m)` 因空 dict 非 None 而触发互斥 `ValueError`；`_run_asm_passes` 内重复的后端校验是死分支；`InstructionScheduler.report` 与 `ScheduleResult.report` 两份报告 API 并存。

### 4.2 动态验证结果

- **修复对象真实存在**：`main` 的 `--schedule` 输出直接丢标签、丢点、乱序（§1.1），pr-59 同输入产出与无调度基线逐字节一致（该样例无可调度收益区域）。
- **随机代码语义对拍**：自写解释器在 400 个含 `div/rem/内存读写/WAW/WAR` 的随机块上，pr-59 换序 332 次、0 语义失败；topic_18 换序 16 次、0 语义失败。pr-59 的换序更激进但都被自身验证器 + 解释器双向确认。
- **真实 CNN 产物**：`--schedule` 与基线对比只做了 3 处 `li`/`mv` 互换（模型 59→55 拍，moved 6），产物仍可被仓库编码器接收；更激进收益受限于后端生成的块很小（每块 2~4 条，标签/terminator 密集）以及 `sw sp(...)`/`max` 等边界。
- **故障注入**：裁行/倒序/自环三种注入分别验收「整区恢复 + warning」与「strict 编译失败 + 旧输出不被覆盖」，与设计 §8 一致。

### 4.3 测试质量

- 优点：精确断言为主（边类型 `{"WAR"}`、`{"RAW","WAW"}`、优先级 `[6,4,1]`、issue cycles `[0,2,3,4]→[0,1,2,3]`、stall 1→0），不搞"长度>0"式弱断言；含失败注入、CRLF 保真、unsafe 矩阵、fuzz-style 确定性、配置组合矩阵（3 分配器 × DAG/非 DAG）、`--schedule` 关闭时**禁止**调用调度器的 monkeypatch 哨兵。
- 缺口：没有仓库内的随机/性质测试（随机只测确定性，不测语义；语义测试依赖 clang/qemu，天天 skip）；没有 qemu/clang 的 CI job；未对 `ScheduleModel` 的延迟表做数值回归；未对 benchmark 收益设阈值断言。
- 全量 760 通过说明未伤及既有全部用例。

### 4.4 工程卫生与合入机制

- **F1（必修）**：`ad9c806` 误提交根目录文件 `:memory:.ses`（51 字节，内容为时间戳 + UUID，明显是会话/工具残留），同时把 `2c71bd8` 刚加进 `.gitignore` 的 `/docs/mytopic/` 又删掉了。应删除该文件并把它加入 `.gitignore`。
- **F2（必修）**：`docs/topic-18/` 4 份文档是"实现前快照"，且互相矛盾：设计要求"输入不解析汇编、fail-stop、拒绝 `linear+schedule`"；计划/说明坚持文本入口、局部回退、默认 linear，并写明"尚未修改实现代码"——而同一分支的代码已经实现。文档里引用的 `18-指令调度器真实状态调研.md`、`18-指令调度器SPEC.md`、`18-指令调度器-详细设计-joska.md` 在仓库中不存在。合入前应至少加版本/状态标注与"与实现差异"章节。
- 分支历史干净，特性提交与文档提交分离；第二轮合入 upstream/main 并新增基准提交 `dcf5f09`；无本仓库其他 topic 的改动夹带。

### 4.5 第二轮独立验证：有效性与覆盖度（2026-09-15 新增）

第一轮证明了「调度改得对」，第二轮回答「调度到底有没有用、在哪里没用」。方法全部独立于实现方的教学模型：LLVM 官方微架构模型 llvm-mca、自写 RV32IM 解释器差分执行、全量真实负载编译审计、发射端/识别端静态对照、以及新增 Benchmark 文档逐项复现。产物见 `/tmp/opencode/t18/`。

#### 4.5.1 llvm-mca 独立微架构验证（LLVM 18.1.8，`rocket-rv32` / `sifive-e76`）

- 合成语料（`benchmarks.bench_inst_scheduler._gen_instructions`，sizes 10–1000 × seeds 42/1/2 × dep_chains 1/2/3/8，共 72 个 region）：
  - ScratchV 教学模型声称节省 9642 拍；llvm-mca `rocket-rv32` 实测仅节省 **783 拍（8.1% 兑现）**，win/tie/loss = **25/7/40**；`sifive-e76` 整体 **-236 拍（净负优化）**。
  - 按依赖链分解：`dep_chains=1` → 18/0/0（+1314，兑现 49%）；`=2` → 7/1/10（-58）；`=3` → 0/2/16（-195）；`=8` → 0/4/14（-278）。**只有单依赖链形态在两个 CPU 上一致为正。**
  - 最小负例（size=100, seed=42, dep_chains=3）：模型 240→187，llvm-mca 449→462（**调度后慢 13 拍**）。逐 region 复核确认该 region 内 div 被大幅提前。
  - 根因是模型参数：`schedule_model.py` 中 `div/rem` latency=16 且"只占除法器、不阻塞其他发射"；真实 `rocket` div latency=33 且阻塞发射，`sifive-e76` div=66；`mul` 模型 3 拍，rocket 实测 4 拍。把 div 提前在模型里赚钱、在真实流水线里亏钱。
- 真实产物：只有 `cnn.onnx` 产生 4 个 applied region（均为 3–4 条，把独立 `li`/`mv` 塞进 `mul` 延迟槽），rocket/e76 分别确认 +4/+3 拍；23 个 DSL 用例在两种分配器下几乎全部 `no_improvement`。
- 口径说明：llvm-mca 也是静态模型（无 cache/分支预测、region 局部输入假设），但使用真实 CPU 参数且由 LLVM 维护；结论在 rocket 与 e76 两套不同参数上互相验证，比教学模型可信。

#### 4.5.2 真实负载覆盖审计（CompilerDriver 全量编译 24 个输入）

- CLI 默认（greedy）：211 条指令、建模 180（85.3%）、**有增益的文件仅 3/24**（`014_for_dot` 模型 +1、`019_nested_loop` +1、`cnn.onnx` +4），共移动 10 条指令；库默认（linear）：覆盖率 52.3%、仅 cnn 有 applied。
- 可调度 region 平均 **1.86 条**（26 个 region 只有 1 条，最大 5 条），官方合成块平均约 310 条——相差约 167 倍。真实代码被标签、terminator、barrier 切成碎片，调度器没有施展空间。
- 未建模指令 Top：`j`×28、`max`×13、`bnez`×12、`slt`(字面量操作数)×6、`bge`×5 等；`j`/`bnez`/`bge` 大量缺失的直接原因是上游 linear 分配器不输出分支目标（见 4.5.4）。
- 指令数不变量：24/24 文件调度前后 199→199，逐 opcode 完全一致——**调度不改变动态指令数**，与 §0 的共性结论一致。

#### 4.5.3 独立解释器语义对拍（正确性复核）

- 自写 RV32IM 解释器对随机块与真实产物做调度前后差分执行：**140 对拍、执行 20022 条、mismatch 0**；覆盖 lw-use、两条 div 竞争、WAR/WAW、同址 store/load、x0 写入、小立即数 `li`、`mv` 链等边界；确定性检查 129/129、幂等 33/33（对已调度输出再调度为不动点）。
- 结论：第一轮的「调度保持语义」结论独立复现且样本更大；正确性不是本 topic 的短板。

#### 4.5.4 发射形式静态对照与上游隐患

- `sw sp(16), t0`（`base(offset)` 方言）被 pr-59 正确识别（12 条 spill store 全部可调度）；topic_18 的 T3 缺口在 pr-59 不存在，这是 pr-59 的净优势。
- **上游 linear 分配器 bug（F12）**：`LsInstruction.to_asm`（`regalloc_linear.py:116-126`）只打印操作数，把分支目标留在注释里，产物为 `bge a2, 4  # .Lloop_exit_3`、`j  # .Lloop_header_1`；`RISCVAEncoder.assemble` 对前者直接 `IndexError`（greedy 路径编码正常）。后果：linear 输入被 `_unsafe_layout` 以 `numeric control-flow target` 整文件熔断，看起来像调度器不支持循环，实际是上游发射不完整。
- `_unsafe_layout` 误报（F14）：字符串常量含 `;`/`\`（如 `.asciz "a;b"`）即整文件零调度；`.popsection/.previous` 之后永久 `executable=False` 且无任何诊断；`.section .rodata,"ax"` 会因 flags 覆盖段名把只读段当可执行。
- 后端合法方言 `max`、`slt rd, 0, rs`、`mv rd, 0` 等未建模（barrier），是 cnn 约 16.9% 覆盖率缺口的主要来源（与 F3 一致）。

#### 4.5.5 新增 Benchmark 文档与 CI 的核实（`dcf5f09`）

- `docs/topic-18/18-指令调度器Benchmark.md`（本次重修时刚拉到的提交）**全部声称可复现**：
  - 固定用例：`python3.11 -m benchmarks.run_inst_scheduler_case` 输出 PASS，表格逐项吻合（4 条指令、模型 5→4、停顿 1→0、编码 4 条/16B、TinyFive 执行 4 条、32 寄存器与 16 字内存一致）；换序确为 `lw; addi; add; sw`。
  - CNN 静态 A/B：JSON 为 `assembly-ab`、`comparison_status=not_modeled`、`simulation.status=not_run`、`output_equal=null`；诊断 `numeric control-flow target; entire input preserved`；文档给出的 `bne t4, zero, -48` 实例存在于产物第 22 行。
  - 合成规模：数字与文档一致，5000 条 `skipped=1` 显示 `N/A`；JSON 新字段（`benchmark_type/input_sha256/seed/repeats/max_region_size/execution_verified=false`）齐全。
  - CI：三份 JSON/Markdown 已接入 workflow、Job Summary 与 artifact（`benchmark_reports/` 已被忽略）。
- 小问题（F15）：文档称"本地虚拟环境可使用 `.venv/bin/python`"，但本机 `.venv` 为 Python 3.8，运行即崩在 `asm_parser_for_beautifier.py:35`；standalone CNN 生成命令本地直跑会 `ModuleNotFoundError`，需要 `pip install -e .` 或 `PYTHONPATH=.`（CI 靠安装步骤掩盖了这一点），文档未注明前提。

#### 4.5.6 topic_18 断言复核（供对照分支作者）

T1（单条 `mul` 报 1 拍）、T2（`div + 独立 add` = 17 拍/15 stall，对照 pr-59 = 16 拍/0 stall）、T3（`sw sp(16), t0` 12 条钉死）、T6（benchmark 用 `r{c}_{i}` 非物理寄存器、legacy API、3 链 0%/单链 9.5%）、`ScheduleConfig.strict` 未接线、`except Exception` 兜底：**全部独立复现成立**；其测试 45 passed。原评审对 topic_18 的判断无需修改。

---

## 5. 发现清单（pr-59）

级别定义：P0 = 正确性阻断；P1 = 合入前必修；P2 = 应当修复；P3 = 可延后。

| ID | 级别 | 位置 | 问题 | 证据 | 建议修复 |
|----|------|------|------|------|----------|
| F1 | P1 | 仓库根 `:memory:.ses`（提交 `ad9c806`） | 误提交 51B 会话残留（时间戳 + UUID），并把 `.gitignore` 中 `/docs/mytopic/` 删除 | `git show --stat ad9c806`：`+ :memory:.ses \| 2 +`；`git log pr-59 -- ':(literal):memory:.ses'` 指向该提交 | 删除文件；如为工具产物加 `.gitignore`；提交信息说明 |
| F2 | P1 | `docs/topic-18/18-指令调度器{设计文档,SPEC-Review,实现计划,代码说明}.md` | 4 份文档为不同时点的旧快照且互相冲突（文本入口 vs 禁止文本；fail-stop vs 安全降级；拒绝 linear+schedule vs 默认 linear；计划/说明称"尚未实现"）；引用的 3 份上游文档不在仓库 | `docs/topic-18/` 仅这 4 份；计划 §1 自认范围调整；说明开头"准备改还没有实现"；设计 §4.3/§11 与代码相反 | 给文档加版本状态与本版差异说明；把"实现现状/验收口径"单独成节；补齐或移除失效引用 |
| F3 | P2 | `schedule_semantics.py:122-146,166-175`；`compiler.py:519-524` | 真实后端合法形式未建模：`max`（多指令展开，必须 barrier，无异议）、以字面量 0/1 作寄存器操作数的 `slt`/`mv` 等；`cnn.onnx` 10/59 未建模，且诊断仅 `info`，默认不可见 | 真实产物 line 11/22/33/51 `max`、12/23/34/67 `slt rd,0,rs`、53 `mv rd,0`、58 `slt rd,rd,1`；`riscv_encoder._reg_num` 把字面量静默映射为 x0（既有缺陷） | 上游让代码生成发射规范寄存器名（`zero`/`x0`）；过渡期可把字面量 0 归一化为 x0（与后端编码一致）；`--schedule-report`/stats 输出覆盖率，未建模数达阈值时升为 warning |
| F4 | P2 | `compiler.py:523` | `--schedule-report` 把多行报告塞进 `warnings` 列表，语义怪（warning 不是报告通道） | `warnings.append(result.report())` | 报告放 `stats["schedule"]["report"]` 或独立字段，CLI 单独打印 |
| F5 | P3 | `inst_scheduler.py:287-335,367-370` | 不安全布局熔断整个文件：一个 `.macro`/`;` 命中即全文件零调度；`skipped_regions=1` 与诊断行号固定为第 1 行，定位误导 | `test_unsafe_assembly_layout_preserves_entire_input`；诊断 `line 1` | 按区域/按行定位与熔断；至少把命中行号写进诊断 |
| F6 | P3 | `inst_scheduler.py:203-241,372-392` | 独立注释行/空行/`label: inst` 同行切分区域（保守但降低收益）；注释与指令同行时可随行迁移，独立注释则不能 | `_read_lines` 对非指令行 `region += 1` | 评估"注释行随相邻指令绑定"；明确文档化该保守策略 |
| F7 | P3 | `schedule_semantics.py:258-291`；`inst_scheduler.py:52-55,462-493` | 破坏性 API 变更：`SchedInst` 变 frozen 且忽略构造参数中的 def/use；`parse_instructions+build_dag` 跨区即抛错；`machine_instrs_from_scheduled` 对内存/终止/未知指令抛错 | 测试 `test_unknown_forms_are_explicit_boundaries`、`test_low_level_api_cannot_silently_drop_boundaries` | 在 CHANGELOG/模块 docstring 标注 breaking change 与迁移建议；考虑弃用排期 |
| F8 | P3 | `inst_scheduler.py:259-266,406-409` | `max_region_size=1024`：超限区域整体跳过（bench 5000 条 0 收益）；大函数收益缺口 | benchmark 5000 行 `Skipped=1` | 后续可分窗/提高上限，或对超大区域仅做局部窗口调度 |
| F9 | P3 | `inst_scheduler.py:44-45,314`；`compiler.py:514-515` | 小问题合集：`latency_model={}` 与 `model=` 互斥判断过严；`.` 正则对字符串内 `;`/`\` 误报；`_run_asm_passes` 中后端校验死分支；两份 report API | 代码走读 + 用例 | 顺手清理，非阻塞 |
| F10 | P2 | `tests/test_inst_scheduler_integration.py:152-211`；`.github/workflows/ci.yml` | 最强安全证据（真实汇编 + qemu 执行对拍，19 例）在本环境与 CI 均 skip：CI 未安装 `qemu-riscv32`（ubuntu-latest 亦无）；无性能/收益回归门 | `pytest -rs` 显示 19 个 skip；workflow 仅 `pip install pytest` | 增加安装 `qemu-user`/`gcc-riscv64-unknown-elf` 的 CI job，并让执行对拍成为必跑；对 benchmark 收益设最低阈值 |
| **F11** | **P1** | `schedule_model.py:43-68` | **div/rem 成本模型与真实微架构失配**（16 拍非阻塞 vs rocket 33 拍阻塞/e76 66 拍；mul 3 vs rocket 4）：llvm-mca 合成语料只兑现模型声称的 8.1%，40/72 region 变慢，e76 整体净负 | §4.5.1；最小负例（size=100, seed=42, dep_chains=3）：模型 240→187 vs llvm-mca 449→462 | div/rem 改为阻塞发射模型或禁止提前 div/rem；mul 校准为 4；模型按 CPU 可配置；把 llvm-mca A/B 纳入收益回归门控 |
| **F12** | **P1** | `regalloc_linear.py:116-126`（上游发射层） | linear 路径**分支目标只写注释不写操作数**，产物不可编码（`RISCVAEncoder` IndexError），并使调度器对含循环输入整文件熔断 | 产物 `bge a2, 4  # .Lloop_exit_3`、`j  # .Lloop_header_1`；编码实测 IndexError，greedy 正常 | 修 `LsInstruction.to_asm` 让分支目标回到操作数位；补 linear 输出可编码回归测试；评估调度器是否可识别注释目标（次选） |
| **F13** | P2 | 真实负载覆盖（§4.5.2） | 真实可调度 region 平均 **1.86 条**，仅 3/24 文件有增益，动态指令数 0 变化；后端碎片化锁死收益上限 | 24 文件审计；region 尺寸分布 26×1、17×2…最大 5；官方合成块平均约 310 | 覆盖率与未建模数升级为 warning/报告字段；支持 `max` 与字面量操作数；评估合并小基本块后再调度 |
| **F14** | P2 | `inst_scheduler.py:287-335` | `_unsafe_layout` 误报与静默失效：字符串常量含 `;`/`\` 全文件熔断；`.popsection` 后永久不可调度且无诊断；`.section .rodata,"ax"` 把只读段当可执行 | §4.5.4 实测（`.asciz "a;b"` → 全文件保留；`.popsection` 后指令无诊断消失） | 熔断收窄到命中行/区域并做引号感知；`.popsection/.previous` 恢复可执行状态；段可执行性同时看段名与 flags |
| **F15** | P3 | `docs/topic-18/18-指令调度器Benchmark.md:7,38-42` | 本地运行说明有误：`.venv/bin/python`（3.8）运行即崩；standalone CNN 生成命令缺 `pip install -e .`/`PYTHONPATH=.` 前提 | 文档命令实测：3.8 崩于 `asm_parser_for_beautifier.py:35`；直跑脚本 `ModuleNotFoundError` | 更正 Python 版本口径；补充安装前提 |
| **F16** | P3 | `benchmarks/run_inst_scheduler_case.py:34-42` | 两套指令计数口径未对齐：`source_instructions=876` vs `scheduling.input_instructions=889`（`_op_/layer1/Conv:` 显示标签被调度器当未知指令计入），文档只说"源指令数不计显示标签" | §4.5.5 CNN JSON | 统一计数口径，或在文档与字段说明中显式注明差异 |

**第二轮状态更新（第一轮 F1–F10）**

- **F1 仍未闭环**：`:memory:.ses` 在第二轮 HEAD `035d180` 仍被 git 跟踪（`git ls-files` 命中；工作区干净），合入前必修。
- **F2 部分改善**：新增 `18-指令调度器Benchmark.md` 质量高、口径诚实；但 4 份旧文档的状态/矛盾问题仍在，`18-指令调度器实现计划.md` 开头仍写"尚未修改实现代码"。
- **F3 已被量化**：真实覆盖率见 §4.5.2；未建模根因新增一项"上游分支目标丢失"（F12）。
- **F4/F5/F8/F9 结论不变**；**F6/F7 影响面较小**；**F10 仍未配置 CI 工具链**，但新 Benchmark 的固定用例已用 TinyFive 做了有界执行验证，部分弥补执行证据缺口。
- 新增 **F11/F12 = P1 必修**，**F13/F14 = P2**，**F15/F16 = P3**。

---

## 6. 与 topic_18 的横向对比

### 6.1 对比总表

| 维度 | pr-59（Mastttttter） | topic_18（Seven Gao） | 胜者 |
|------|----------------------|----------------------|------|
| 结构 | 4 模块分层，单向依赖，语义一次性解析 | 单文件 1065 行，自包含 | pr-59（可测试/可演进） |
| 语义建模 | 精确白名单 + barrier，别名校验/立即数范围/两种内存语法 | OpcodeInfo 表 + 操作数宽容（int 字面量当合法操作数） | 各有千秋：pr-59 更严更稳，topic_18 覆盖后端字面量形式 |
| 后端语法覆盖 | 支持 `offset(base)` 与 `base(offset)` 两种，cnn.onnx 建模 49/59 | 只认 `offset(base)`，`sw sp(16), t0` 全部钉死（12/59）并刷 warning | **pr-59** |
| 依赖图 | RAW/WAR/WAW + memory + FP flags + control | RAW/WAR/WAW + memory chain + control | pr-59（FP 副作用序） |
| 成本模型 | 资源占用型（div 只占除法器），周期=完成时间；相对真实硬件仍偏乐观（F11） | 全局 EX 停摆（div 阻塞全流水 16 拍），周期=发射数（不含末条延迟） | **pr-59** |
| 收益门控 | `after < before` 严格更低才应用（同模型同口径） | 同思路，但模型偏差导致大量机会丢失 | pr-59 |
| 验证器 | 边序 + 独立读来源/最终写值 + 副作用序 + 对象同一/边界 | 置换 + 边序 + 末条 pinned（依赖自身 DAG） | **pr-59** |
| 失败处理 | 区域恢复 + 严格模式抛错 + 不写部分输出；只捕获自己的异常 | 区域 fallback + 警告；`except Exception` 兜底（与自身计划矛盾）；strict 未接线 | pr-59 |
| 集成 | 配置校验 + `--schedule-strict/--schedule-report` + stats/诊断 | `_asm_pass_stats` 可变驱动状态 + `ScheduleConfig()`（strict 无效） | pr-59 |
| 测试 | 4 文件 **121 例**（精确断言 + 注入 + CRLF + fuzz 确定性 + 执行对拍 + 新 Benchmark 报告 16 例） | 1 文件 45 例（精确断言，无执行对拍） | pr-59 |
| 基准 | 重写为合法物理寄存器 + 生产入口 + 免责声明；第二轮新增执行验证/静态 A/B/合成三份 CI 报告 | 未更新（`r{c}_{i}` 伪寄存器、走 legacy API、3 链 0% 改善） | pr-59 |
| 文档 | 5 份：`Benchmark.md` 质量高且口径诚实；4 份旧快照互相矛盾（F2） | 无新增文档 | pr-59（Benchmark） |

### 6.2 topic_18 的关键问题（供其分支作者修复）

- **T1（P1）周期口径错误**：`estimate_sequence` 返回 `clock = 末条 issue + 1`，不含末条指令延迟。证据：单条 `mul` 报 **1 拍**（pr-59 报 3 拍）；`lw; add` 两者一致纯属巧合。后果：报告收益被低估，且"把长延迟指令提前"这类收益被收益门控直接丢弃。
- **T2（P1）全局 EX 停摆模型**：非流水除法把整条发射流水停 `latency` 拍（`ex_free = issue + exec_latency`）。证据：`div + 独立 add` 模型 **17 拍 / 15 stall**（pr-59 为 16 拍 / 0 stall）；自带 benchmark 3 依赖链 0% 改善；对照语料无 div 时 0/400 次换序。这直接解释其"收益弱"的表象。
- **T3（P1）后端语法缺口**：`sw sp(16), t0`（后端 Emitter 实际输出、编码器合法形式）不被 `_base_reg` 识别 → 12 条 spill store 全部锚定并输出 warning。在真实 CNN 产物上等于关闭了访存调度。
- **T4（P2）验证器偏弱**：验证依赖同一实例构建的边表；不做"读取来源/最终写值"独立比对，也不校验对象同一与边界。
- **T5（P2）集成毛刺**：`ScheduleConfig.strict` 在 `schedule_asm` 中未被使用；`compiler.py` 的 `except Exception` 兜底违反其自身计划 §8 的"不使用吞掉所有程序错误的通用 except"；`_asm_pass_stats` 为可变驱动状态。
- **T6（P2）基准未同步**：benchmark 仍用 `r{c}_{i}` 等非物理寄存器与 legacy API，未度量生产入口 `schedule_asm`；第二轮实测 3 依赖链全部 0% 改善、单链约 9.5%、5000 条压力测试 0% 且耗时 3.5s（对照 pr-59 生产入口重写版：同规模合成语料模型约 14%~19%，但对真实硬件同样偏乐观，见 F11）。
- **T7（P3）语义边界**：`jalr ra, ...` 被当作 terminator（保守但语义上更像 call）；label 名与寄存器 ABI 名同名时会被误判为寄存器（如名为 `ra` 的标签）。

### 6.3 可互相吸收的点

- topic_18 → pr-59：`_is_int_literal` 式操作数宽容（至少覆盖后端发出的 `slt rd, 0, rs`/`mv rd, 0`）可减少无谓 barrier；`_anchor_diagnostics` 把不支持形式即时告警的思路，可弥补 pr-59 info 级诊断不可见的问题。
- pr-59 → topic_18：资源占用型模型、完成时间口径、独立验证器、strict 模式与配置校验、测试与基准的组织方式。
- 合并策略：**同一 topic 不应同时落入两个实现**。建议以 pr-59 为合入版本，topic_18 转为参考实现或关闭；若团队偏好 topic_18 的自包含风格，至少要修完 T1~T3 再比较。

---

## 7. 测试与验证结论

### 7.1 通过项（第一轮 + 第二轮）

- **pr-59 第二轮 HEAD `035d180`**：调度相关 **121 passed, 19 skipped**（含新增 `tests/test_inst_scheduler_report.py` 16 例）；全仓 **848 passed, 19 skipped**。第一轮基线：调度器 105/105、全仓 760/760。
- topic_18：45/45、594/594；第二轮独立复跑 45 passed。
- **语义正确性（第二轮独立解释器）**：140 对差分执行、20022 条指令、**mismatch 0**；确定性 129/129、幂等 33/33（对已调度输出再调度为不动点）。
- 故障注入、CRLF/EOF 保真、不安全整文件熔断、配置组合矩阵、`--schedule` 关闭时的 monkeypatch 哨兵均有用例锚定。
- **新增 Benchmark（`dcf5f09`）独立复现**：固定用例 PASS 且 TinyFive 真实执行结果一致；CNN 静态 A/B 为 `not_modeled` 且两侧产物逐字节相同；合成 JSON 字段与数字一致；CI 已接入三份报告与 Job Summary。
- 真实产物：`cnn.onnx` 调度后产物可编码，模型估算 59→55 拍（局部静态，非实测）；llvm-mca 独立确认 4 个 region 合计约 +4 拍（rocket-rv32）。

### 7.2 覆盖缺口（第二轮更新）

- **真实收益没有硬件或周期精确模拟背书**：llvm-mca 已给出独立微架构估计（合成语料仅兑现 8.1%、部分配置为负），但它仍是静态模型；无 qemu、无硬件计时（F10/F11）。
- 真实汇编 + 执行对拍（19 例）在 CI 中仍默认跳过（F10）；新 TinyFive 固定用例只覆盖 4 条直线整数代码，不能代表真实程序。
- 未对 `ScheduleModel` 的延迟表做数值回归，也没有 benchmark 收益下限/负收益门控（F11）。
- 未见 FP 语义的独立执行对拍（PR 内有 FP 用例，同样因 qemu 缺失 skip）。
- 未测 `--schedule` 与 `--beautify/--peephole/--const-merge` 的顺序敏感组合之外的交叉影响（第二轮新增 standalone CNN `--const-merge` 后的静态 A/B，覆盖一种组合）。

---

## 8. 修复清单（供修复阶段执行，2026-09-15 第二轮更新）

**合入前必修（P1）**

- [ ] F1：删除 `:memory:.ses`；如系工具产物补 `.gitignore`；验证 `git status` 干净、CI 不受影响。
- [ ] F11：修正 div/rem 成本模型——改为阻塞发射语义或禁止提前 div/rem；`mul` 由 3 校准为 4；先用 llvm-mca（rocket-rv32 / sifive-e76）对合成与真实语料跑 A/B，要求总收益为正、win ≥ loss，再对外声称"有收益"。验证：同一 region 的 llvm-mca A/B 不再出现系统性负优化。
- [ ] F12：修 linear 分配器分支目标发射（`LsInstruction.to_asm`），使 linear 与 greedy 产物同样可编码；验证：`RISCVAEncoder.assemble` 对含循环 DSL 的 linear 产物能产出二进制，调度器不再因 `numeric control-flow target` 整文件熔断。
- [ ] F2：为 `docs/topic-18/` 文档加状态页（设计=V2 backlog，计划/说明=历史快照，Benchmark=现状），修正"尚未修改实现代码"等与代码相反的描述。验证：文档内不再出现与实际代码相反的"当前实现"描述。

**紧跟改进（P2）**

- [ ] F3：后端发射规范寄存器名；过渡期 `describe` 归一化字面量 `0`；覆盖率与未建模数写入 stats/report，超阈值升为 warning。
- [ ] F4：报告移出 `warnings` 通道，CLI 单独打印。
- [ ] F10：CI 增加 `qemu-user` + RISC-V 工具链，让 19 个执行对拍必跑；增加 llvm-mca 收益回归门（可与 F11 合并）。
- [ ] F13：把"真实负载覆盖率"作为可量化指标跟踪；评估合并小基本块后再调度的可行性。
- [ ] F14：`_unsafe_layout` 熔断收窄并做引号感知；修 `.popsection` 状态恢复与 `.section` flags 判断。

**延后（P3）**

- [ ] F5/F6/F7/F8/F9：按 §5 建议排期，非合入阻塞。
- [ ] F15：更正 Benchmark 文档的本地运行说明（Python 版本、`pip install -e .`/`PYTHONPATH` 前提）。
- [ ] F16：统一 `source_instructions` 与 `scheduling.input_instructions` 计数口径，或在文档与字段说明中注明差异。

---

## 9. 综述评价

### 9.1 对 pr-59 的评价

**这是 topic 18 三版实现（旧文本重排 → topic_18 重写 → pr-59）里质量最高的一版。**它的价值不在算法复杂度（经典的块内列表调度），而在工程纪律：

1. **先安全后收益**：未知形式不猜、危险布局熔断、收益不严格改善不应用、验证失败不落盘、strict 不写部分输出——把"面向教学编译器的可选优化"做成了可以放心开开关的部件；`main` 上那个会把汇编标签丢光的旧 `--schedule` 被彻底替换，这是"修复"最直观的证据。
2. **语义精确性**：一型一解析、寄存器别名校验、正确识别多指令伪操作（`max`/大 `li`/分支立即数），避免了两类经典误编译（把伪操作当单指令调度、把字面量当寄存器）。
3. **验证闭环**：独立验证器 + 顺序敏感共享模型 + 收益门控，使得"调度器自己的错"很难逃逸；这也是与其对照实现拉开差距的地方。
4. **可观测、可测试、可演进**：stats/diagnostics/报告、121 个精确断言用例（含新增 Benchmark 报告 16 例）、四个单向依赖模块，为未来迁到 post-RA 结构化调度留了干净接口。

第二轮把「有效性」也纳入评审后，扣分项集中为三类：**合入卫生**（`:memory:.ses` 未删、四份文档矛盾未解）、**真实覆盖率**（真实可调度 region 平均 1.86 条，仅 3/24 文件有增益）、**模型可信度**（div/mul 与真实失配，llvm-mca 只兑现 8.1%，部分配置净负）。这些不涉及正确性（140 对差分执行 0 失败），但直接决定"这个开关在真实工作流里到底省了多少、什么时候不省"，也决定它不应被当作性能达标项。综合评级（第二轮）：**架构 A−，正确性 A，安全 A，测试 A−，有效性 D（模型内有效、真实未证实且可能为负），文档 B（新增 Benchmark 文档质量高，旧文档仍待整理），工程卫生 B**。

### 9.2 对 topic_18 的评价

自包含单文件的实现思路清晰、测试用例设计也不错（45 例，含精确断言与 `.label` 处理），但两个模型级问题（T1 周期口径、T2 全局 EX 停摆）和真实的语法缺口（T3）让它在同样的语料上基本"不敢动"——对照语料无 div 时 400 例 0 次换序（pr-59 为 95/400），自带 benchmark 3 链 0% 改善（pr-59 同规模语料约 14%~19% 模型节省；但第二轮 llvm-mca 表明 pr-59 的真实兑现远低于此，见 F11）。验证器与集成强度也不足。**它更适合作为参考实现**，其中"操作数宽容"和"即时不支持告警"两点值得吸收。

### 9.3 合并策略建议

1. 以 **pr-59** 为 topic 18 的合入版本，定位为"默认关闭的文本层 V1 基础设施"；topic_18 不合入同一功能，保留分支存档或转为参考实现。
2. 合入顺序：先完成 4 项 P1（F1 删残留文件、F11 div/rem 模型、F12 linear 分支目标、F2 文档状态），再合入；F3/F4/F10/F13/F14 紧随。
3. 若短期内无法完成 F11/F12：建议暂缓合入，或合入但明确禁止对外宣称收益——一个默认关闭、开启后可能负优化的调度器对用户的伤害大于价值。
4. 合入后必须在 `CHANGELOG` 标注：调度仍是文本层 V1、默认关闭、收益为局部静态模型估算、动态指令数不变、不代表硬件加速；post-RA 结构化调度列入 V2 backlog。
5. **明确本 topic 与项目核心 KPI 的关系**：LLVM 对比的 4.2x 差距是动态指令数指标，而本调度器不改变指令数（24/24 文件 199→199），不应计入"超越 LLVM"的路径；它的价值在延迟优化基础设施与教学演示。

### 9.4 后续路线（在本次评审结论之上）

- **短期**：F11 模型修正（div/rem 阻塞、mul=4）并用 llvm-mca 做收益门控；F1/F12/F2；F3/F10/F13/F14。让"真实覆盖率"和"llvm-mca A/B"变成 CI 可量化指标。
- **中期**：修上游 linear 发射与分支目标；把 `SchedInst` 适配层接到结构化 `MachineInstr`（设计文档 §5 的 `InstructionView`），复用现有 model/verify/scheduler；统一 `PipelineCycleEstimator` 与新模型口径（设计 §18 的校准项）。
- **长期**：post-RA/pre-emission 位置前移、内存别名放宽、issue width>1、按 CPU 校准模型、默认开启评估——都需要真实硬件校准与执行对拍支撑，不要以教学模型数字对外宣称硬件收益。

---

## 10. 最终结论：是否值得合入当前编译器（2026-09-15 第二轮）

**结论：值得合入，但要按「默认关闭的 V1 基础设施」定位合入，并在合入前完成 F1/F11/F12/F2 四项 P1。**

- **支持合入**：正确性已用独立解释器（140 对拍 0 失败）与全量测试（848 passed）充分验证；安全体系（精确语义 → 区域 → DAG → 独立验证器 → 收益门控 → strict/不写部分输出）是本仓库同类 pass 中最强一档，默认关闭时对主路径零风险；验证器、统计报告、CI 基准与 TinyFive 执行入口具备长期复用价值，是未来 post-RA 结构化调度的干净底座；新增 Benchmark 文档口径诚实，没有把模型数字当硬件收益。
- **保留意见**：它目前不产生可证实的真实收益——llvm-mca 只兑现模型声称的 8.1%，40/72 region 反而变慢，sifive-e76 整体净负；真实负载 24 个输入只有 3 个发生换序；对动态指令数 0 贡献。若合入标准是"带来可测的性能提升"，本 topic 不达标。
- **分场景判定**：
  - 作为「教育/基础设施特性」——**合入**（修完 P1）；
  - 作为「性能达标项 / 对外宣称超越 LLVM 的依据」——**不成立**，不应以此宣传；
  - 在 P1（尤其 F11/F12）未修时——**不建议合入**，因为开启后可能让用户代码变慢。

**一句话**：这是一件正确、安全、诚实、但目前几乎不产生真实收益的基础设施；按默认关闭的 V1 合入并立即修模型，是风险与收益最平衡的选择。
