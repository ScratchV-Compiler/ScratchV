# Topic 18：Review 迭代报告

日期：2026-09-16。基线：`035d180`。评审输入：[topic18-review.md](topic18-review.md) 的第二轮 F1–F16；T1–T7 属于另一个对照分支，不在本轮修复范围内。候选为包含本报告的 review 迭代提交。

本轮修复四项 P1 对应的代码、文档及仓库卫生问题，并补充覆盖率、段状态处理和独立性能审计。**实现仍是默认关闭的文本 V1；独立模型总体验收通过，但仍有负优化样例，不能宣传为普遍硬件加速。**

## 1. 逐项处理结果

| 编号 | 本轮处理 | 状态 / 保留事项 |
|---|---|---|
| F1 | 将 `:memory:.ses` 从 Git 索引移除，添加根目录忽略规则 | 已处理。运行期间该本地会话文件会重建，因此保留本地文件，仅取消跟踪；没有移除 `docs/topic-18` 的跟踪 |
| F2 | 新增文档状态页，重写当前代码说明，历史计划和设计标记版本，移除失效链接的验收依赖 | 已处理；结构化 V2 仍为 backlog |
| F3 | 在机器指令对象中将整数寄存器源位置上的 0 规范为 `zero`；补充覆盖率、未建模 opcode 计数和 warning | 核心改进完成；`max` 及非零字面量占据寄存器位置的形式仍固定，不把 1 等常量错误解释为 x0 |
| F4 | 多行报告移入 `stats["schedule"]["report"]`，CLI 独立打印 | 已处理；warnings 只承载诊断 |
| F5 | 不安全布局诊断指向实际命中行，整文件跳过升为 warning | 定位已修复；宏、条件汇编、数字跳转等仍整份保护，尚未缩小影响范围 |
| F6 | 明确记录独立注释、空行、同行标签作为边界 | 采用文档化保守行为；未猜测注释归属 |
| F7 | 在 CHANGELOG 和代码说明中列出不可变语义对象、跨区拒绝及有损转换拒绝的迁移说明 | 已处理；推荐完整汇编使用 `schedule_assembly` |
| F8 | 保留 1024 条默认上限，并明确配置入口和跳过口径 | 分窗调度延后；5000 条整区仍显示 N/A |
| F9 | 允许空 legacy 延迟字典与显式模型共用；修复引号误报；说明两份 report API 的兼容用途 | 保留 `_run_asm_passes` 后端校验，保护直接调用该入口的情形 |
| F10 | 新增 `scheduler-review` CI job，安装 clang/lld/QEMU/llvm-mca，工具缺失时执行测试必须失败；增加两个 CPU 的 A/B 验收 | 本地执行验证及审计通过；GitHub 新 job 尚待本次修改推送后执行 |
| F11 | 乘法 4 拍、整数除法 33 拍；除法保守阻塞发射并保持相对顺序；补 WAW 完成时间；主模型及另一组延迟必须同时改善 | 本轮总体门槛通过，review 最小负例保留原序；完整 CPU 模型选择与硬件校准仍未实现 |
| F12 | `LsInstruction.target` 独立保存目标，标签发射为 `label:`，分支目标发射为操作数；分支/store 首操作数按读取处理 | 编码回归通过；新增 QEMU 实际执行倒向循环的验证 |
| F13 | 新增 23 个编号 DSL + CNN、greedy/linear 两条路径的全量覆盖审计，保留零收益文件和区域大小 | 测量和告警完成；未跨基本块合并，仍只有 3/24 个真实输入变化 |
| F14 | 引号感知的布局检查，恢复 push/pop/previous 段状态，异常恢复诊断，明确数据段及非执行 flags 保护 | 已修复所列误报与段状态缺陷；真正危险布局仍按 F5 保守保护 |
| F15 | 写明 Python 版本检查、editable 安装与工具链前提；standalone 命令显式设置 `PYTHONPATH=.` | 已处理；不把评审方的 Python 3.8 当成每个环境的现状 |
| F16 | Benchmark 与调度器共用源指令计数范围，显示标签不再计为未知指令 | 已处理；本地 CNN 两侧均为 876 条，不再出现 876 / 889 两种口径 |

## 2. 模型修正为什么这样做

仅把除法延迟从 16 改成 33，或者只禁止除法跨过其他指令，并不能消除模型偏差。本轮采用三层限制：

1. 除法保持相对位置，并在教学模型中保守阻塞后续发射。
2. 对物理寄存器 WAW 计入完成时间，避免把较早的长延迟写入当作已经结束。
3. 用乘法 3、除法/余数 66 的第二组参数检查候选。只有两组估算都严格改善才接受；否则记录 `model_sensitive` 并保留原序。

第二组参数来自 LLVM SiFive7 的整数延迟范围，但不是完整 E76 模拟器。该检查仍采用单发射及简化资源规则；真实 CPU 的发射宽度、旁路、缓存和动态除法时长未被完整表达。

默认三条依赖链、种子 42 的合成规模测试（10–1000 条）在本轮全部保留原序，模型收益为 0；5000 条仍因超限未建模。这里保留零收益，不挑选替代样例制造改善。独立 72 例审计覆盖更多依赖链形态，另行统计总体效果。

## 3. 独立 llvm-mca 前后对照

以下基线和候选均在同一台机器、**LLVM 22.1.8** 下重跑。评审原文使用 LLVM 18.1.8，部分统计不同；不能把不同版本的数字直接混在同一列比较。两边使用相同的 72 个输入及其 SHA-256：规模 10/50/100/200/500/1000 × 种子 42/1/2 × 依赖链 1/2/3/8，每个输入 MCA 迭代数为 1。

| CPU | 实现 | 原序总周期 | 调度后总周期 | 总节省 | 胜 / 平 / 负 |
|---|---|---:|---:|---:|---|
| rocket-rv32 | 基线 | 103055 | 102285 | 770 | 25 / 9 / 38 |
| rocket-rv32 | 本轮 | 103055 | 102105 | 950 | 22 / 41 / 9 |
| sifive-e76 | 基线 | 175522 | 175441 | 81 | 28 / 5 / 39 |
| sifive-e76 | 本轮 | 175522 | 174608 | 914 | 24 / 37 / 11 |

验收规则为：每个 CPU 总节省 > 0，且胜例数 ≥ 负例数。本轮两者均通过，基线两者均未通过。**仍有 Rocket 9 个、E76 11 个负例**，完整结果保存在 [review-iteration-results.json](review-iteration-results.json)，没有从统计中剔除。

review 最小负例（100 条、seed=42、chains=3，输入 SHA-256 `31a9c756dd15e42660ba930fc5020ee119eb2bcb3782cb6591abe99e8098a975`）：

| 实现 | 本地模型 | Rocket MCA | E76 MCA |
|---|---|---|---|
| 基线 | 240→187 | 449→462 | 769→788 |
| 本轮 | 457→457，保留原序 | 449→449 | 769→769 |

两版本地模型的绝对周期没有可比性；可比较的是在同一个独立 MCA 中各自调度前后的差异。上述结果均为静态 CPU 模型估算，不是硬件实测。

## 4. 真实编译输入的覆盖率

审计逐一编译 23 个编号 DSL 与本地 `models/graph/cnn.onnx`，在 greedy 和 linear 两种分配器下分别生成汇编。对每一份汇编只切换调度开关。输入模型、原始汇编及调度后汇编哈希见 JSON。

| 分配器 | 指标 | 基线 | 本轮 |
|---|---|---:|---:|
| greedy | 建模 / 输入指令 | 180 / 211 | 187 / 211 |
| greedy | 覆盖率 | 85.3% | 88.6% |
| greedy | 发生变化的文件 | 3 / 24 | 3 / 24 |
| linear | 建模 / 输入指令 | 104 / 199 | 175 / 199 |
| linear | 覆盖率 | 52.3% | 87.9% |
| linear | 发生变化的文件 | 1 / 24 | 3 / 24 |

本轮平均区域长度为 greedy 2.23、linear 2.08，最大分别为 10、5，仍然很小。两条路径各剩 24 条未建模指令，主要包括 `max`、带非零立即数的寄存器形式及多指令伪操作。没有通过放宽未知语义来提高覆盖率。

本轮两条路径合计 12 个已应用区域，在 MCA 中检查其指令体（固定终止指令除外）：Rocket 87→75，胜/平/负=12/0/0；E76 70→63，胜/平/负=7/5/0。这是已应用区域的局部估算，不能相加解释为 CNN 或完整程序运行时间。

另行复跑 standalone CNN 静态 A/B：源指令数与调度器输入数统一为 876；数字分支位于第 22 行，诊断准确定位到该行，结果仍为 `not_modeled`、两份汇编一致、未运行完整 CNN。这个入口与上面的 CompilerDriver CNN 路径不同。

## 5. 正确性与工程验证

- 全仓：`901 passed`，无跳过；另有 11 个既有 pytest marker warning。
- 原有 19 个 clang + QEMU 对拍实际执行；新增 linear 倒向循环 QEMU 用例通过。
- 三个循环/控制流 DSL 在 linear、调度开/关两种配置下均可由仓库编码器生成非空机器码，目标标签存在，调度不再因丢失目标而整文件熔断。
- 固定 TinyFive 用例：32 个整数寄存器与 16 个数据字一致；源指令/编码指令/动态执行指令均为 4→4，模型 5→4，停顿 1→0。
- 新增回归覆盖引号内特殊字符、嵌套段栈、previous 切换、数据 flags、异常恢复、显示标签、诊断定位、覆盖缺口、敏感性拒绝、WAW 成本与零寄存器规范化。
- 工作流 YAML 可解析；`git diff --check` 通过。新 GitHub Actions job 尚未在远端执行，不能将本地验证等同于远端 CI 已通过。

## 6. 复现与产物

从仓库根目录运行，使用已安装依赖的 Python 3.12+ 环境。完整环境和 CNN 生成命令见 [Benchmark](18-指令调度器Benchmark.md)。

```bash
SCRATCHV_REQUIRE_RISCV_EXECUTION=1 python -m pytest tests/ -q \
  --junit-xml=benchmark_reports/topic18_iteration_tests.xml
python -m benchmarks.run_inst_scheduler_case
python -m benchmarks.bench_inst_scheduler --repeats 3 \
  --json benchmark_reports/inst_scheduler_synthetic.json \
  --markdown benchmark_reports/inst_scheduler_synthetic.md
python -m benchmarks.audit_inst_scheduler --llvm-mca llvm-mca-18
python -m benchmarks.run_inst_scheduler_case \
  benchmark_reports/cnn_scratchv.s --static-only \
  --json benchmark_reports/inst_scheduler_cnn.json \
  --markdown benchmark_reports/inst_scheduler_cnn.md
```

本地实际使用 `.venv/bin/python` 3.14.7，以及下载到 `/tmp/topic18-llvm/usr/bin/llvm-mca` 的 LLVM 22.1.8；没有安装或更换系统 LLVM 包。CI 配置为 Python 3.12 / LLVM 18，并在干净 runner 上生成最小 CNN；该模型可能与本地已有 CNN 不同，比较时必须看输入哈希。

基线复跑方法：将 `035d180` 中的 `scratchv`、`scratchv_dag`、`benchmarks` 用 `git archive` 解到临时目录，通过 `PYTHONPATH` 指向该目录，再运行候选中的 `benchmarks/audit_inst_scheduler.py`。这使同一审计脚本和同一 MCA 驱动旧实现；没有切换或修改工作区代码。

可随 PR 提交的记录是本文与 `review-iteration-results.json`。完整原始 JSON、Markdown、JUnit 和生成日志在被忽略的 `benchmark_reports/` 中；CI 会上传 `scheduler-review-report` artifact。

## 7. 后续工作边界

本轮没有实现完整目标 CPU 配置、消除所有负例、Fast/BURR、结构化 post-RA 调度、跨基本块调度、别名放宽或大区域分窗。建议后续优先针对 JSON 中保留的负例校准发射/旁路模型，并处理上游非零立即数放在寄存器位置的发射问题。在这些工作完成前，继续保持调度默认关闭。
