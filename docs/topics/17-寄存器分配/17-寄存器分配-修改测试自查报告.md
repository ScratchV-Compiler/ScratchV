# Topic17 修改、测试与自查报告

> 主题：寄存器分配支持伪指令，并正确统计寄存器溢出
>
> 分支：`topic17-pseudo-regalloc`
>
> 记录基线：`5de34a1` 及其之前已经进入项目的 Topic17 提交
>
> 最近验证日期：2026-09-14
>
> 状态：本地修改完成，尚未 commit/push

## 1. 范围说明

本报告记录 Topic17 从实验性线性扫描分配器、benchmark、伪指令语义支持，到本轮正确性加固的相关修改。

- 本轮没有修改 `scratchv/backend/regalloc_cfg.py` 或其他 CFG 文件。
- CFG 的正式集成及图算法由对应维护者负责；本工作只消费现有接口。
- `tests/test_inst_counter.py` 的本地改动属于工作区原有修改，不计入本报告。
- 当前可执行验证目标是 RV32IM；F/D 扩展伪指令保持显式拒绝，不宣称已经支持。

## 2. Topic17 历史阶段

| 阶段/提交 | 主要内容 |
|---|---|
| `0394e0a` | 增加 v1.3/v1.5 线性扫描实验实现、瓶颈场景及回归测试。 |
| `5975348` | 增加 simple、dense、CNN 三类寄存器分配 benchmark 及报告生成。 |
| `1a55f83` | 建立 MachineOp 语义表，支持伪指令 def/use，统一 spill/reload 指标并增加 P1 测试。 |
| `bae263c` | 将 `MAX` 的发射逻辑集中到 `_emit_max`，统一普通与扩展指令选择路径。 |
| `2766cf6` | 增加多种压力模型和寄存器溢出对比 benchmark。 |
| `5de34a1` | 增加伪指令分配、编码、TinyFive 执行的端到端测试和 benchmark。 |
| 当前本地修改 | 修复审查发现的正确性问题，强化 UT、benchmark、ABI 和编码范围校验。 |

## 3. 已实现功能

### 3.1 统一机器指令语义

`scratchv/backend/machine_semantics.py` 为每个 `MachineOp` 明确定义：

- 显式 `defs` / `uses`；
- 可使用立即数的位置；
- label、跳转目标、terminator 和 call 属性；
- call 的隐式定义与 caller-saved clobber 集合；
- 是否为伪指令及其物理寄存器需求。

线性扫描分配器、greedy/naive 分配器和 Machine IR 转换共用该语义，避免各自推断操作数字段。

### 3.2 RV32IM 伪指令支持

| 层级 | 指令 | 验证方式 | 状态 |
|---|---|---|---|
| Machine IR | `mv` | 分配 → 编码 → TinyFive 执行 | PASS |
| Machine IR | `li` | 小/大/边界立即数，分配 → 编码 → 执行 | PASS |
| Machine IR | `max` | 正负值、零立即数、临时寄存器场景 | PASS |
| Machine IR | `bnez` | taken/fallthrough 路径执行 | PASS |
| Machine IR | `j` | 跳转目标及不可达指令验证 | PASS |
| Machine IR | `call` | `jal ra` 展开、返回及 clobber 处理 | PASS |
| Machine IR | `.label` | 零字节结构标记和后续执行 | PASS |
| Assembler | `nop` | 与 `addi x0, x0, 0` 对比执行 | PASS |
| Assembler | `ret` | 与 `jalr x0, ra, 0` 对比执行 | PASS |
| 外部扩展 | `fabs.d`、`fneg.d`、`li.d`、`fmv.s` | RV32IM 编码器明确报错 | 按范围拒绝 |

### 3.3 MAX 发射统一

- 普通指令选择与扩展指令选择共用 `_emit_max`。
- 两个立即数直接折叠为 `li`。
- 左侧立即数利用交换律移动到右侧。
- 非零右侧立即数先物化到虚拟寄存器，再发射 `max`。
- 新增全 Program 名称预留，避免内部 `__scratchv_max_rhs_N` 与用户值重名。

### 3.4 spill/reload 与寄存器分配修复

- naive 分配器为同一条二元指令的不同虚拟源分配不同 scratch register。
- naive 目的寄存器不再进行无意义的旧值加载。
- greedy 在控制流边界写回仍活跃值，taken 和 fallthrough 路径共享已初始化的 spill slot。
- call 前保存仍存活且会被 ABI clobber 的虚拟值，call 后按需 reload。
- 线性扫描和 greedy 都会识别显式物理寄存器定义，避免虚拟值被固定 `t0` 等操作数静默覆盖。
- 自定义物理寄存器列表要求为有效、唯一的 RV32 整数可分配寄存器。

### 3.5 ABI 栈帧

新增 `scratchv/backend/abi_frame.py`，并接入生产编译路径：

- 根据 spill slot 和实际使用的 callee-saved register 创建 16 字节对齐栈帧；
- 函数包含 call 时保存和恢复传入 `ra`；
- 保存和恢复实际使用的 `s0`–`s11`；
- 将分配阶段的负 spill offset 重定位到已保留的栈帧范围；
- frame 超过当前 12 位立即数安全上限时明确失败，避免静默截断。

### 3.6 编码器安全检查

`scratchv/backend/riscv_encoder.py` 新增：

- I/S/U/B/J 类型立即数或偏移范围检查；
- branch/jump 偏移对齐检查；
- RV32 `srai` shift amount 范围检查；
- 十进制、十六进制、负十六进制共用立即数解析；
- 超范围输入抛出 `ValueError`，不再通过位掩码静默截断。

### 3.7 可审计的溢出指标

- allocator 插入的 store/load 分别带有 `[regalloc:spill]` 和 `[regalloc:reload]` 标记。
- `spill_slots` 表示唯一栈槽数量。
- `spill_stores` / `reg_spill_count` 表示静态 spill store 站点数量。
- `reloads` 表示静态 reload load 站点数量。
- `pressure_peak` 表示峰值活跃区间数量。
- 普通模型内存访问或用户注释中的 `spill`、`reload` 字样不会污染统计。

## 4. 单元测试与端到端测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_regalloc_pseudo.py` | 每条 RV32IM 伪指令的语义、分配、编码和 TinyFive 执行；外部扩展明确拒绝。 |
| `tests/test_regalloc_p1.py` | 高压力 spill/reload、随机直线程序对拍、call clobber、分支路径交接、物理寄存器冲突、naive/greedy 执行。 |
| `tests/test_regalloc_metrics.py` | spill slot/store/reload/pressure 指标定义及 CNN 验证条件。 |
| `tests/test_regalloc_pseudo_benchmark.py` | 伪指令 benchmark 覆盖矩阵、执行结果和报告序列化。 |
| `tests/test_regalloc_spill_compare.py` | 不同压力模型与不同寄存器数量下的溢出对比。 |
| `tests/test_abi_frame.py` | 栈帧重定位、callee-saved、嵌套 call 的 `ra` 保存恢复及超大 frame 拒绝。 |
| `tests/test_riscv_encoder_validation.py` | 立即数、内存 offset、控制转移 offset 和 shift amount 的范围检查。 |

最终完整回归：

```text
757 passed in 44.97s
```

## 5. Benchmark 完善

### 5.1 防止假通过

- simple/dense/CNN/pseudo benchmark 改用生产版 `scratchv.backend.regalloc_linear`。
- simple/dense 不再使用不可编码的 `r0`–`rN` 假寄存器。
- simple/dense 先由独立解释器计算参考 `a0`，再执行实际分配汇编进行对拍。
- CNN 为 Machine IR live-in 注入确定性值，并由独立 Machine IR 解释器计算参考结果。
- CNN 的线性扫描和 greedy 输出分别经过真实编码和 TinyFive 执行。
- benchmark 的 PASS 同时要求统计条件、汇编合法和执行结果正确。

### 5.2 30 次运行结果

命令：

```powershell
python -m benchmarks.test_regalloc.bench_regalloc_linear `
  --repeats 30 `
  --output-json benchmark_reports/regalloc_bench.json `
  --output-html benchmark_reports/regalloc_bench.html `
  --output-md benchmark_reports/regalloc_bench.md
```

| Benchmark | Mean(ms) | Vregs | Spill stores | Spill slots | Reloads | Pressure peak | 执行 |
|---|---:|---:|---:|---:|---:|---:|---|
| Simple Arithmetic | 0.021 | 5 | 0 | 0 | 0 | 3 | PASS |
| Dense Computation | 0.297 | 20 | 60 | 27 | 74 | 30 | PASS，`a0=59766` |
| CNN Integration | 0.272 | 30 | 0 | 0 | 0 | 11 | linear/greedy 均 PASS，`a0=1` |
| Pseudo Instructions | 0.056 | 7 | 0 | 0 | 0 | 3 | 9/9 PASS |

报告位置：

- `benchmark_reports/regalloc_bench.json`
- `benchmark_reports/regalloc_bench.html`
- `benchmark_reports/regalloc_bench.md`

### 5.3 LLVM 对比边界

本次报告记录：ScratchV 静态指令 69，LLVM RV64FD 静态指令 1101，报告比值 15.96。

该数字只能描述当前两个生成路径的静态输出规模，不能直接证明 ScratchV 性能优于 LLVM：ScratchV CNN 路径是标量近似 Machine IR，而 LLVM 路径包含更完整的模型、浮点、ABI 和运行时工作。

## 6. Topic17 涉及的主要文件

### 后端实现

- `scratchv/backend/machine_semantics.py`
- `scratchv/backend/instruction_select.py`
- `scratchv/backend/register_alloc.py`
- `scratchv/backend/regalloc_linear.py`
- `scratchv/backend/regalloc_linear_v1_5.py`
- `scratchv/backend/regalloc_rewrite.py`
- `scratchv/backend/regalloc_metrics.py`
- `scratchv/backend/riscv_encoder.py`
- `scratchv/backend/abi_frame.py`
- `scratchv/compiler.py`

### Benchmark

- `benchmarks/test_regalloc/bench_simple.py`
- `benchmarks/test_regalloc/bench_dense.py`
- `benchmarks/test_regalloc/bench_cnn.py`
- `benchmarks/test_regalloc/bench_pseudo.py`
- `benchmarks/test_regalloc/bench_regalloc_linear.py`
- `benchmarks/bench_regalloc_spill_compare.py`
- `benchmarks/regalloc_spill_cases/`

### 测试

- `tests/test_regalloc_pseudo.py`
- `tests/test_regalloc_p1.py`
- `tests/test_regalloc_metrics.py`
- `tests/test_regalloc_pseudo_benchmark.py`
- `tests/test_regalloc_spill_compare.py`
- `tests/test_abi_frame.py`
- `tests/test_riscv_encoder_validation.py`

## 7. 自查结论

- 完整 pytest：PASS（757 项）。
- 30 次寄存器分配 benchmark：四组全部 PASS。
- `git diff --check`：PASS。
- 当前本地 CFG 文件修改数：0。
- 用户原有工作区文件和临时目录均保留。

## 8. 已知边界与后续事项

1. CFG 正式接口和后续集成不在本次修改范围内。
2. 当前编码和模拟执行覆盖 RV32IM；F/D 扩展仍需要独立的寄存器类别、编码器和模拟器支持。
3. 远距离 branch/jump 尚未自动生成跳板；当前行为是明确拒绝超范围偏移。
4. 超过 2032 字节的栈帧尚未使用多指令地址生成；当前行为是明确失败，避免错误机器码。
5. CNN benchmark 验证的是当前标量化 Machine IR 的寄存器分配正确性，不等同于完整 ONNX 数值精度验证。
6. 合入前仍需由维护者确认 CFG PR 提供的接口稳定性，并在集成后重新运行本报告中的全部回归与 benchmark。

## 9. 本地状态

本文档和本轮修复目前只存在于本地工作区，尚未创建 commit，也尚未 push 到 GitHub PR。
