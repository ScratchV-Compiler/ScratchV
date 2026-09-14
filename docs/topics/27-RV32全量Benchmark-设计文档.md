# 课题 27：RV32 全量 Benchmark 设计文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/standalone/rv32_bench.py`（RV32 统一对比驱动）、`scratchv/standalone/bench_report.py`（报告渲染）、`scratchv/simulator/tinyfive.py`（`ProfiledMachine` 适配器，只读接口）、`scratchv/standalone/benchmark.py`（解析估算，仅作告警）  
> 功能范围：RV32IMF 统一目标下的 ScratchV/LLVM 对比基准；全量指令仿真与预算控制；诚实报告规范（provenance 字段、ISA 标注、模型身份）；CLI 参数规格；报告 JSON schema v2  

---

## 一、功能介绍

### 1.1 功能概述

课题 27 的目标是让 `rv32_bench.py` 真正执行“全量或明确预算并如实标注”的 RV32 模拟基准，而不是当前的“5000 条指令 + 64KB 内存 + 无权重”的伪全量对比。现状问题如下（均已核对源码与运行记录）：

1. **默认截断**：`rv32_bench.py:118` 默认 `n_instructions: int = 10000`，调用点 `rv32_bench.py:410-412` 实际传 `n_instructions=5000`。任何一次运行中，`ops.total` 恒 ≤ 5000，仅覆盖 `_start` 与输入拷贝循环的开头，与“动态指令数”语义无关。
2. **内存容量错误**：`rv32_bench.py:124` 使用 `ProfiledMachine(mem_size=65536)`（64KB）。而 ScratchV 编译器约定的裸机 ABI 是 `sp=128MiB`、`a0=160MiB`（input）、`a1=192MiB`（output）、权重紧跟在代码之后（`data_offset`，`onnx_to_riscv_standalone.py:2548-2557`）。`models/graph/cnn.onnx` 为 27,677,152 字节（约 26.4MB 权重），64KB 内存连权重区都容纳不下。
3. **权重数据未装载**：编译器只把权重写进 `.bin`（`binary = code_bytes + weight_data`，`onnx_to_riscv_standalone.py:2791`）；`.s` 仅是代码反汇编（`onnx_to_riscv_standalone.py:2802-2807`），不含 26MB 权重。`run_tinyfive` 只把 `.s` 文本喂给 `load_asm`，权重恒为 0。
4. **标签解析失效**：`_prepare_asm_for_tinyfive`（`rv32_bench.py:160-184`）构造了 `label_map` 却从不使用，标签行被丢弃；随后 `load_asm` 经 `RISCVAEncoder` 组装，而未解析标签走 `self.labels.get(label, current_idx)`（`riscv_encoder.py:502`），分支全部退化成“跳到自己”。同时 `.word` 数据行被当指令跳过。
5. **静态计数冒充动态计数**：TinyFive 缺失时 `_tinyfive_static_fallback`（`rv32_bench.py:187-234`）返回 `ops` 字典，其 `total = sum(静态计数)`，并写入 `instr_count=n_instr`；报告把该结果与真实仿真结果同表渲染。
6. **报告声明失真**：`generate_report` 硬编码模型描述（`rv32_bench.py:268` 写死 `cnn.onnx (3×Conv + …)`），输出 `Dynamic instruction ratio`（`rv32_bench.py:303`），页脚宣称 “All metrics sourced from TinyFive ProfiledMachine simulation. No analytical estimates.”（`rv32_bench.py:310`）。
7. **LLVM 侧跨 ISA 与估算**：LLVM IR 生成器固定输出 `riscv64-unknown-elf` 模块头（`onnx_to_llvm_standalone.py:446`），而 `rv32_bench.py:82` 却向 llvmlite 请求 `riscv32` 目标机；两者不一致。llvmlite 未安装时 `status="skipped"`，但报告仍渲染对照表并保留 ratio 行。周边工具 `llvm_cache_compare.py` 的“动态指令数”是解析估算（该文件第 9 行自述），不属于本课题的实测口径。

本课题的功能定义：

- **全量仿真（full simulation）**：从入口 PC=0（`_start`）执行到 `_done: ret` 的停机地址，不设指令数截断；内存按裸机 ABI 尺寸分配；`.bin` 的代码段与权重段分别装载；输入按确定性种子生成。
- **明确预算（budgeted simulation）**：允许 `--max-instructions N` 主动截断；截断必须写入 `completion="budget_exhausted"`，且所有依赖完整轨迹的派生指标（比值、per-MAC）置空。
- **诚实报告（honest reporting）**：每个数字携带来源分类（measured/static/estimated/unavailable）与 provenance；动态指标只能来自真实仿真；ISA 与数值格式显式标注；模型身份由内容哈希与维度定义；不可复现的旧文案全部删除。

### 1.2 设计目标

- **真实性**：报告中的每一列都有 `source` 与 provenance；`audit_provenance()` 空违规列表是产出报告的前置条件。
- **可复现**：相同的模型、种子、参数、依赖版本产出相同的计数（时间类字段除外）；模型 `sha256`、输入种子、内存布局全部入档。
- **全量优先、预算可选**：默认全量；截断是显式选择且显式标注；禁止把截断结果当作全量结果展示。
- **可比性**：两侧统一标注 `rv32/imelf` 与数值格式（Q16.16 vs float32）；不可比时 ratio 为 `null` 并给出 `incomparable_reason`。
- **失败可见**：TinyFive 缺失、内存不足、标签解析失败、ISA 不匹配都必须以非零退出码或显式状态呈现，绝不静默降级。
- **零新依赖**：仅使用 `tinyfive`、`numpy`、标准库；llvmlite 保持可选。
- **兼容性**：保留 `--output-dir/--html/--json/--md` 既有 CLI 语义，既有测试不回归。

---

## 二、设计规范

### 2.1 全量仿真定义

#### 2.1.1 指令上限策略

用运行模式与完成状态替代“隐藏的 10000/5000 常量”：

```
run_mode    ::= full | budgeted(N) | (缺省 = full)
full        ::= "--full" | "--max-instructions 0"
budgeted(N) ::= "--max-instructions" N        ; N > 0
completion  ::= "halted" | "budget_exhausted" | "timeout" | "error" | "not_run"
```

| 运行模式 | TinyFive 调用 | 停止条件 | `completion` |
|----------|---------------|----------|--------------|
| full（默认） | `m.run(instructions=chunk)` 分块循环，累计直到 `pc == halt_addr` | PC 命中停机地址；或墙钟超时 | `halted` / `timeout` |
| budgeted(N) | 同上，剩余额度 = N − executed | 额度耗尽；或 PC 命中停机地址 | `budget_exhausted` / `halted` |
| 参数冲突 | `--full` 与 `--max-instructions N>0` 同时给出 | 启动前拒绝 | 退出码 2（usage） |

规则：

- `--max-instructions 0` 与 `--full` 等价，均为无截断语义；`limit=null`、`executed` 为真实执行数。
- 分块循环使用 `--chunk-instructions`（默认 10,000,000）作为每块大小；块间检查停机、墙钟与预算。块内精确停机由实例级 `exe` shim（2.1.4）保证，两者都不修改 `ProfiledMachine` 公共接口。
- `budget_exhausted` 时 `executed == limit` 必须为等式不变量；`halted` 时 `limit` 可为 `null`（full）或任意（budgeted 提前停机）。
- 超时只在 `--full` 模式下产生 `timeout`；`budgeted` 模式下先到限额即结束，超时属于异常保护（同样标 `timeout`）。
- 派生指标约束：`comparison.dynamic_instruction_ratio` 仅当两侧 `source=="simulated"` 且 `completion=="halted"` 时计算；任一 `budget_exhausted/timeout` 一律 `null` 并写 `incomparable_reason`。

#### 2.1.2 内存布局与容量

内存布局沿用编译器自检脚本的裸机约定（`onnx_to_riscv_standalone.py:2548-2557`），并补充停机字：

| 区域 | 起址 | 大小 | 来源 |
|------|------|------|------|
| code | `0x00000000` | `data_offset` | `.bin[0:data_offset]` |
| weights | `data_offset` | `binary_bytes − data_offset` | `.bin[data_offset:]` |
| HALT（保留 4B） | `halt_addr = align_up(binary_bytes, 16)` | 4 | 本课题注入，不作为指令执行 |
| workspace | `sp` 向上 | `workspace_size` | 编译器 stdout `Workspace: N bytes` |
| stack | `sp` 向下 | 预留 | 由 `sp` 指向栈顶 |
| sp | `128 MiB` | — | ABI 约定 |
| input | `160 MiB` | `input_elements × 4` | Q16.16，确定性种子 |
| output | `192 MiB` | `output_elements × 4` | 仿真结束后读取 |

容量约束（启动前校验，不满足即 `exit 4`）：

```
data_offset % 4 == 0
halt_addr + 4              <= mem_size
192 MiB + out_bytes + 4096 <= mem_size          # output 区必须可用
sp + workspace_size + GUARD <= 160 MiB          # 工作区不得撞 input 区，GUARD = 1 MiB
```

默认 `--mem-size 268435456`（256MiB）。若模型或工作区要求超过该值，必须显式调大而不能静默截断；若小于 `192MiB + out_bytes`，直接拒绝运行（因为 ABI 地址不可压缩）。

#### 2.1.3 权重数据装载

```
image        ::= code_bytes (已 4 字节对齐) ‖ weight_bytes
data_offset  ::= len(code_bytes)
data_size    ::= len(weight_bytes) = len(.bin) − data_offset
```

- `data_offset` 从编译器 stdout 行 `  Data offset: 0x{hex} ({n} bytes)`（`onnx_to_riscv_standalone.py:2794`）解析，正则 `r"Data offset:\s*0x([0-9A-Fa-f]+)"`；`workspace_size` 从 `  Workspace: {n} bytes`（`:2738`）解析。解析失败 → `exit 4`，`reason="binary_layout_unparsed"`。
- 装载顺序：`ProfiledMachine(mem_size)` → `m.load_binary(code_words, origin=0)` → `m.load_data(weight_bytes, data_offset)` → `m.set_reg(2, 128MiB)` / `set_reg(10, 160MiB)` / `set_reg(11, 192MiB)` / `set_reg(1, halt_addr)`。
- `gp` 由代码自身在 `_start` 通过打补丁的 `auipc/addi` 设置（`onnx_to_riscv_standalone.py:2757-2789`），harness **不得**另写 `gp`。
- 输入一次性构造后 `m.load_data(input_blob, 160MiB)`，不得逐元素 `write_mem_i32`（性能与确定性双重原因）。Q16.16 生成算法与编译器自检一致：`random.seed(seed)`、`val = int((random.random() − 0.5) × 0.2 × 65536)`（`onnx_to_riscv_standalone.py:2564-2567`），默认 `seed=42`。

#### 2.1.4 停机条件

- 生成代码结尾为 `_done: ret`（`jalr x0, x1, 0`，`onnx_to_riscv_standalone.py:1553-1556`）。harness 在运行前设 `x1 = halt_addr`，当 PC 到达 `halt_addr` 即判定 `halted`。
- TinyFive 的 `exe(start, end)` 原生支持按 end 地址停止，但 `ProfiledMachine` 未暴露该参数，且其 `exe` 在给定 `end` 时会忽略 `instructions` 预算。harness 因此只对**机器实例**重绑 `exe`：每步执行前同时检查 `pc == halt_addr` 与指令额度，二者先到者停（`_install_tinyfive_compat`，`rv32_bench.py`）；`scratchv/simulator/tinyfive.py` 与 `ProfiledMachine` 公共接口保持不变。
- 外层仍以 2.1.1 的分块 `m.run(instructions=chunk, start=pc, strict=True)` 循环驱动，块间做墙钟检查、预算记账与 `pc` 复核；实例级 shim 保证单块内不会越过停机地址空转解码（否则每次解码都会计入 `ops.total`，虚增动态计数）。
- 若生成代码在返回前用 `jal ra, …` 覆盖了 `ra`，或跳转路径异常，PC 永远不会命中 `halt_addr`：`budgeted` 记 `budget_exhausted`，`full` 记 `timeout`。不得谎报 `halted`。
- TinyFive 遇到不支持的指令时打印错误且 PC 不前进（`dec()` 无匹配分支时不调用 `ipc()`），会表现为“卡死”。启动前必须做**助记符白名单预检**：用 ScratchV 的 `_disasm_one` 解析每个 code word，若出现 TinyFive 不支持/无法识别的助记符 → `completion="not_run"`，`exit 4`，禁止开跑。
- `m.last_error` 非空（适配器捕获到异常）→ `completion="error"`，该侧 `dynamic.source="unavailable"`、`ops=null`。

#### 2.1.5 超时与预算保护

- `--timeout SECONDS`（默认 900）为单侧仿真墙钟上限。实现：主线程 `signal.setitimer(ITIMER_REAL, remaining)` + `SimulationTimeout` 处理器；`m.run(..., strict=True)` 使适配器 `finally` 仍在异常路径更新 `instr_count`，ops 计数器保留部分值。
- 超时判定不依赖异常类型（适配器会把异常包装成 `RuntimeError`）：置模块级 `_timed_out` 标志，捕获后据此写 `completion="timeout"`，并把 `elapsed_s`、`executed` 落盘。
- 平台不支持 `SIGALRM`（如 Windows）时：`--timeout` 退化为告警，要求用户必须给 `--max-instructions`；否则拒绝启动并提示替代方案。
- **不做自动预估**：本驱动不运行 1M 探测，也不调用 `benchmark.estimate_cnn_model()`（该解析模型只属于 `benchmark.py` 自身 CLI），因此报告中不存在 `estimated_*` 墙钟数字。full 模式启动前只在 `warnings` 写一条显式提示：无自动预估、可用 `--max-instructions N` 校准、墙钟超时触发时结果如实标 partial。若未来引入任何预估值，字段名必须带 `estimated_` 前缀，且绝不填入 `dynamic`。

### 2.2 诚实报告规范

#### 2.2.1 字段来源分类

报告所有数值字段必须可归入以下四类之一，渲染时以标签区分：

| 分类 | 允许的字段 | 硬性要求 |
|------|-----------|----------|
| `measured` | `*.dynamic.*`（ops、executed、输出值） | 必须来自 TinyFive 真实执行；`source="simulated"`，且 `completion ∈ {halted, budget_exhausted, timeout}` |
| `static` | `*.compile.static_insns`、`*.static_instruction_mix.*` | `source="asm_scan"`；只统计 `.text` 助记符行；数据指令（`.word/.long/.byte/.float`）不计 |
| `estimated` | 预估值、`est_hw_time_*`、cache 模型输出（如引用） | 字段名含 `estimated`/`model`；不得进入 `dynamic` |
| `unavailable` | 失败/跳过侧 | `source="unavailable"`，`ops=null`，必须有 `reason` |

#### 2.2.2 ISA 标注与统一 RV32

- `targets.scratchv = {isa:"rv32im", abi:"ilp32", numeric_format:"q16.16"}`。
- `targets.llvm = {isa:"rv32imf", abi:"ilp32", numeric_format:"float32", triple:"riscv32-unknown-elf", opt_level:2}`。
- 统一口径动作：llvmlite 可用时，解析 IR 后显式设置 `llmod.triple = "riscv32-unknown-elf"` 再 `emit_assembly`；不可用则 `status="skipped"`。
- **跨 ISA 检测**：对 LLVM 侧 `.s` 做助记符扫描，命中 RV64-only 集合（`ld, sd, lwu, addw, subw, addiw, sllw, srlw, sraw, slliw, srliw, sraiw, mulw, divw, divuw, remw, remuw, fld, fsd, fcvt.l.s, fcvt.s.l` 等）时置 `isa_mismatch=true`、`isa_detected="riscv64"`，禁用该侧动态对照。
- 两侧 ISA 或数值格式不同的比值一律 `null` + `incomparable_reason`；ISA 相同也不得用估算数替代。
- **边界**：LLVM 侧真实可执行镜像（汇编→链接→ABI→输入输出）属于课题 25/27 交界；本课题只负责“标注与统一 RV32 口径”，不实现 LLVM 侧符号重定位与运行时。

#### 2.2.3 模型身份与维度

报告必须包含且只依据以下事实描述模型：

```
model.path, model.sha256, model.bytes,
model.input_name, model.input_shape, model.output_name, model.output_shape,
model.initializer_count, model.weight_bytes
scratchv.compile.binary_sha256, scratchv.compile.data_offset, scratchv.compile.data_bytes
environment.python, environment.numpy, environment.tinyfive, environment.llvmlite
```

禁止在报告模板中硬编码任何具体模型名或层结构描述（现行 `rv32_bench.py:268` 的 “cnn.onnx (3×Conv + …)” 必须删除）。

#### 2.2.4 不可复现数据的禁用

以下做法在 schema v2 中一律违规，由 `audit_provenance()` 检出并阻断：

1. 静态计数标为 `simulated`，或任何 `dynamic` 字段缺少 `simulator/simulator_version/executed/memory_size_bytes/input_seed`。
2. `instr_count = limit` 式伪造：`completion=="halted"` 但 `executed == limit` 且 `limit != null`（未验证停机）。
3. 截断/超时结果参与比值计算。
4. 报告声明与 provenance 不符（例如页脚宣称 “all metrics from simulation” 而存在 `estimated` 列）。
5. 把 `llvm_cache_compare.py` 的解析估算、`benchmark.estimate_cnn_model()` 的结果、`cache_model.py` 的命中率填入 `measured` 分类。
6. 硬编码模型描述与 `model.path/sha256` 不一致。
7. 只记录时间戳不记录哈希/种子/版本。

页脚文案由 provenance 动态生成，例如：
`Simulated by tinyfive {version} | completion={completion} | executed={executed} | limit={limit} | model sha256={sha256[:12]} | seed={seed}`。

### 2.3 CLI 参数规格

| 参数 | 类型/默认 | 语义 | 约束 |
|------|-----------|------|------|
| `model`（位置参数） | path 必填 | ONNX 模型 | 存在且可读 |
| `--output-dir` | path，`benchmark_reports` | 产出目录 | 自动创建 |
| `--html` / `--json` / `--md` | filename，`rv32_bench.{html,json,md}` | 三种格式文件名 | 相对 `--output-dir` |
| `--max-instructions` | int，`0` | `>0` 为预算；`0` 为全量 | 与 `--full` 互斥（N>0 时） |
| `--full` | flag，默认关 | 等价 `--max-instructions 0` | 与 N>0 同时出现 → exit 2 |
| `--mem-size` | int 字节，`268435456` | TinyFive 内存容量 | 必须满足 2.1.2 约束，否则 exit 4 |
| `--timeout` | float 秒，`900` | 单侧仿真墙钟上限 | POSIX `SIGALRM`；否则需预算模式 |
| `--chunk-instructions` | int，`10000000` | 分块粒度 | `>0` |
| `--input-seed` | int，`42` | 输入生成种子 | 与编译器自检同算法 |
| `--skip-llvm` | flag | 跳过 LLVM 编译 | LLVM 侧整体 `not_run` |
| `--allow-missing-simulator` | flag | TinyFive 缺失时输出纯静态报告 | 报告 `dynamic.source="unavailable"`，exit 0 |
| `--fail-on-incomplete` | flag | `completion != "halted"` 时 exit 7 | 供 CI 使用 |
| `--quiet` | flag | 不向 stdout 打印 Markdown | 报告仍落盘 |

退出码：`0` 成功；`1` 未预期错误；`2` 用法错误；`3` 仿真器不可用（未加 `--allow-missing-simulator`）；`4` 模型/镜像/内存布局校验失败；`7` 未完成且指定 `--fail-on-incomplete`。

### 2.4 合法/非法报告示例

#### 2.4.1 合法示例 A：全量完成

```json
{
  "schema_version": "rv32-bench/2",
  "model": {"path": "models/graph/cnn.onnx", "sha256": "0123…cdef", "bytes": 27677152},
  "scratchv": {
    "compile": {"status": "success", "data_offset": 24336, "data_bytes": 26700000,
                 "static_insns": 6084, "static_source": "asm_scan"},
    "dynamic": {"source": "simulated", "simulator": "tinyfive", "simulator_version": "1.0.0",
                 "completion": "halted", "limit": null, "executed": 1844674407,
                 "memory_size_bytes": 268435456, "input_seed": 42,
                 "ops": {"total": 1844674407, "load": 1, "store": 1, "mul": 1,
                          "add": 1, "madd": 0, "branch": 1}}
  },
  "comparison": {"dynamic_instruction_ratio": null,
                  "incomparable_reason": "llvm.dynamic.source!='simulated'"}
}
```
（数值为占位示例；关键点是全字段 provenance 齐备。）

#### 2.4.2 合法示例 B：预算截断（如实标注）

```json
{
  "scratchv": {
    "dynamic": {"source": "simulated", "simulator": "tinyfive", "simulator_version": "1.0.0",
                 "completion": "budget_exhausted", "limit": 1000000, "executed": 1000000,
                 "memory_size_bytes": 268435456, "input_seed": 42,
                 "ops": {"total": 1000000, "load": 0, "store": 0, "mul": 0,
                          "add": 0, "madd": 0, "branch": 0}}
  },
  "comparison": {"dynamic_instruction_ratio": null,
                  "incomparable_reason": "scratchv.completion=='budget_exhausted'"}
}
```

#### 2.4.3 非法示例（应被 `audit_provenance()` 拒绝）

| # | 片段 | 违反条款 |
|---|------|----------|
| 1 | `"dynamic":{"source":"simulated","ops":{"total":3841}}`，无 simulator/version/executed | 2.2.4-1 |
| 2 | `"completion":"halted","limit":5000,"executed":5000` | 2.2.4-2（5000 是截断值） |
| 3 | `"comparison":{"dynamic_instruction_ratio":0.74}` 两侧均 `budget_exhausted` | 2.2.4-3 |
| 4 | 页脚 “All metrics sourced from TinyFive … No analytical estimates” 而 `estimated_hw_time` 列存在 | 2.2.4-4 |
| 5 | `"model":{"path":"resnet18.onnx"}` 而正文描述为 “cnn.onnx (3×Conv + …)” | 2.2.4-6 |

---

## 三、测试设计

### 测试用例 1：小模型全量仿真计数正确性

- **文件**：`tests/test_rv32_bench.py::test_full_run_small_model_matches_reference`
- **输入**：测试内用 `onnx.helper` 构造的迷你模型（如输入 `1×1×8×8`、单层 `Conv 3×3 → 1×1×6×6`，无 FC），写成临时 `.onnx`；随后调用 `compile_scratchv()` + `run_simulation(max_instructions=0, mem_size=268435456, timeout_s=60)`（内存仍按 192MiB 输出区约定，不可缩小）。
- **预期输出**：`scratchv.dynamic.completion == "halted"`；`executed > 0`；`ops.total == executed`；随后用独立功能仿真器 `benchmark.RV32EmulatorFast`（`load_unified_binary` + `run(max_instr=2_000_000_000)`）跑同一 `.bin` 与同一输入/指针布局。
- **验证点**：`tinyfive.total == emulator.total`、`load` 与 `load_count`、`store` 与 `store_count`、`branch` 与 `branch_total` 逐一相等；输出 `Q16.16` 值一致；对同一命令重跑一次，计数完全一致（确定性）。任何不等即失败，禁止容差放过。

### 测试用例 2：预算超限行为

- **文件**：`tests/test_rv32_bench.py::test_budget_exhausted_is_labeled`
- **输入**：对同一迷你模型（或 `cnn.onnx` 若存在）执行 `run_simulation(max_instructions=1000, ...)`。
- **预期输出**：`completion=="budget_exhausted"`、`limit==1000`、`executed==1000`、`ops.total==1000`；`comparison.dynamic_instruction_ratio is null` 且 `incomparable_reason` 含 `budget_exhausted`；Markdown 中出现 `[measured/budget]` 标签而非 “dynamic instruction ratio” 正文。
- **验证点**：`audit_provenance(report) == []`；把该 JSON 传给 `bench_report.validate_report_schema()` 通过；`--fail-on-incomplete` 下退出码为 7。

### 测试用例 3：报告字段与 provenance 校验

- **文件**：`tests/test_rv32_bench.py::test_report_requires_provenance`
- **输入**：monkeypatch `ProfiledMachine` 为不可用（模拟未安装 TinyFive），命令行加 `--allow-missing-simulator`。
- **预期输出**：`scratchv.dynamic.source=="unavailable"`、`ops is null`、`completion=="not_run"`、`reason` 非空；`scratchv.compile.static_insns > 0` 且 `static_source=="asm_scan"`；无 `dynamic_instruction_ratio`；页面含 “static” 标签与显式不可用提示。
- **验证点**：`validate_report_schema()` 对必填 provenance 键（`schema_version/model.sha256/environment/targets/*.compile.static_source/comparison.incomparable_reason`）逐项断言；未加 `--allow-missing-simulator` 时退出码为 3。

### 测试用例 4：内存布局校验

- **文件**：`tests/test_rv32_bench.py::test_memory_layout_validation`
- **输入**：`--mem-size 1048576`（1MiB，小于 output 区 192MiB）。
- **预期输出**：退出码 4，stderr 含 `memory_layout_invalid` 与所需最小字节数；不产生任何带 `dynamic.ops` 的报告文件。
- **验证点**：`write_report` 未被调用（临时目录为空）；错误信息给出可操作建议（明确 “需要 ≥ 192MiB + 输出区 + 4096 字节” 的具体数值）。

### 测试用例 5：标签解析不产生自跳转

- **文件**：`tests/test_rv32_bench.py::test_parsed_labels_cover_all_branches`
- **输入**：迷你模型产出的 `.s`；`parse_labels(asm_text)` 的结果与 `data_offset/4` 的关系校验。
- **预期输出**：`len(labels) > 0`；`_start` 映射 0；`_done` 在文件末尾附近；每个分支/跳转目标标签都能在 labels 中找到；`static_insns == data_offset // 4`。
- **验证点**：若任一分支目标缺失，函数抛 `LabelParseError`（不允许 `labels.get(target, pc)` 回退为自跳转）；该断言直接回归现行 `riscv_encoder.py:502` 的静默语义。

### 测试用例 6：LLVM 跨 ISA 标注

- **文件**：`tests/test_rv32_bench.py::test_llvm_riscv64_flagged_isa_mismatch`
- **输入**：含 `ld/sd/addiw` 的 RV64 汇编 fixture（或跳过 llvmlite 时的真实 `_ll_rv32.s`）。
- **预期输出**：`llvm.compile.isa_detected=="riscv64"`、`isa_mismatch==true`、`llvm.dynamic.source=="unavailable"`、`comparison.dynamic_instruction_ratio is null`。
- **验证点**：报告不出现 “RV32IMF” 对照列；`incomparable_reason` 指名 ISA 不匹配。

---

## 四、修改模块与实现步骤

### 4.1 涉及文件

| 文件 | 角色 | 改动性质 |
|------|------|----------|
| `scratchv/standalone/rv32_bench.py` | 主驱动：编译、装载、仿真、组装报告 | 大幅重构（详见开发文档改动清单） |
| `scratchv/standalone/bench_report.py` | Markdown/HTML/JSON 渲染 | 新增 `render_*(report)` 与 schema 校验；旧函数保留兼容壳 |
| `tests/test_rv32_bench.py` | 新增测试 | 新建 |
| `tests/fixtures/`（或测试内生成） | 迷你 ONNX 与 RV64 汇编 fixture | 新建（测试内用 `onnx.helper` 生成优先） |
| `scratchv/standalone/onnx_to_riscv_standalone.py` | 编译器 | **不改**（只读取 stdout 与 `.bin/.s`） |
| `scratchv/simulator/tinyfive.py` | 仿真适配器 | **不改公共接口**（只使用 `load_binary/load_data/run(strict=True)/pc/get_perf`） |
| `scratchv/standalone/llvm_cache_compare.py` / `cache_model.py` / Spike / `benchmarks/` | 其他课题资产 | **不改**（边界见 4.9） |

（注：上表路径为本仓库真实路径；若后续目录重构，以 `scratchv/standalone/` 下同名文件为准。）

### 4.2 镜像装载与布局解析

1. `compile_scratchv()` 保留子进程调用（`--asm` 输出同目录 `.s`），新增解析 stdout：
   - `Data offset: 0x…` → `data_offset`；`Workspace: N bytes` → `workspace_size`；`Code size: N bytes` → 校验值。
2. `load_scratchv_image(binary_path)` 返回 `(code_words, weight_bytes, data_offset)`：
   - 校验 `data_offset % 4 == 0`、`data_offset ≤ len(binary)`、`len(binary) − data_offset > 0`；
   - `code_size = data_offset`，`binary_bytes = len(binary)`，`static_insns` 由 `.s` 扫描并与 `data_offset // 4` 交叉校验。
3. `halt_addr = align_up(binary_bytes, 16)`；在 TinyFive `mem` 中 `halt_addr` 处无需写指令（停止发生在取指前），但要求 `halt_addr + 4 ≤ mem_size`。

### 4.3 内存与输入初始化

1. `compute_layout()` 按 2.1.2 公式校验并返回全部地址；失败抛 `LayoutError`，`main` 转 exit 4。
2. `ProfiledMachine(mem_size=args.mem_size)`；`available` 为假时按 `--allow-missing-simulator` 决策（exit 3 或静态报告）。
3. `load_binary(code_words, 0)` → `load_data(weight_bytes, data_offset)` → `load_data(input_blob, INPUT_ADDR)`。
4. `set_reg(2, SP_ADDR=128MiB)`、`set_reg(10, 160MiB)`、`set_reg(11, 192MiB)`、`set_reg(1, halt_addr)`；`gp` 留空由代码设置。
5. 输出读取：`read_mem_i32(192MiB)`（单元素）或按 `output_elements × 4` 读字节；记录 `output.raw_hex` 与 `output.q16_16`。

### 4.4 仿真执行器

1. `run_simulation(...)` 分块循环：

```
executed = 0; timed_out = False
while True:
    若 budgeted 且 executed == limit: completion = budget_exhausted; break
    若 m.pc == halt_addr:             completion = halted; break
    若 墙钟超时:                       completion = timeout; break
    chunk = min(chunk_instructions, limit - executed) 若 budgeted 否则 chunk_instructions
    设置 SIGALRM(剩余墙钟)；m.run(instructions=chunk, strict=True)；清除 SIGALRM
    executed = get_perf()["total"]
```

2. 停机/超时后从 `m.get_perf()` 取 ops、从 `(m._machine.x_usage > 0).sum()` 取 `x_registers_used`、`.sum()` 取 `x_usage_total`；`m.last_error` 非空且非超时 → `completion="error"`。
3. 启动前执行助记符白名单预检（2.1.4）；不通过 → `not_run` + exit 4。
4. 整个执行器不写 `scratchv/standalone/` 之外的任何文件；临时产物仅 `--output-dir`。

### 4.5 标签解析与静态统计

1. `parse_labels(asm_text)` 两遍扫描：`.s` 中标签独占一行（`RISCVEmitter.disassemble()`，`:1291-1305`），每遇一行缩进的指令行 word index +1；返回 `{pc: label}`。
2. `static_instruction_mix(asm_text)` 按助记符表分类（复用 `rv32_bench.py:206-220` 的 opcode→类别映射），但输出字段名为 `static_instruction_mix`（`source="asm_scan"`），不与 `dynamic.ops` 同名同表。
3. 若解析出的指令数 ≠ `data_offset // 4`，抛 `LabelParseError`，禁止“尽力而为”。

### 4.6 报告组装与渲染

1. `build_report()` 产出 schema v2 字典（详细字段见开发文档《接口契约》）。
2. `audit_provenance(report)` 返回违规列表；`main` 在写盘前调用，非空则打印并 exit 1。
3. `bench_report.render_markdown/html/github_summary(report)` 负责三格式渲染；所有表格列头或行内带 `[measured]/[static]/[estimated]/[unavailable]` 标签；页脚由 provenance 动态生成。
4. `bench_report.validate_report_schema(report)` 供测试与 CI 做 JSON 结构断言。

### 4.7 LLVM 侧：标注与统一 RV32

1. `compile_llvm_rv32()`：`binding.Target.from_triple("riscv32-unknown-elf")`；llvmlite 缺失/目标不可用 → `status="skipped"` + `reason`，`dynamic.source="unavailable"`。
2. 成功的路径上：解析 IR → 显式 `llmod.triple = "riscv32-unknown-elf"`（覆盖 `onnx_to_llvm_standalone.py:446` 的 riscv64 模块头）→ `verify()` → `emit_assembly()`。
3. 对输出的 `.s` 做 `detect_isa_mismatch()`；命中 RV64-only 助记符 → `isa_mismatch=true`，动态对照关闭。
4. 静态计数只扫 `.text` 段：遇到 `.section .rodata`（权重 `.word/.long`）即暂停计数；`static_source="asm_scan"`。
5. 不实现 LLVM 侧可执行镜像（符号重定位、`a0/a1` 约定、数据段加载）——属课题 25 边界；未实现即 `unavailable`，不得用 `llvm_cache_compare.py` 的估算补位。

### 4.8 集成与回归测试

- 新增 `tests/test_rv32_bench.py`（第三部分 6 个用例），全部可在 `make test` 下运行；无 TinyFive 的环境自动 skip 动态类用例（用 `pytest.importorskip("tinyfive")` 模式，与 `tests/test_simulator.py` 一致），但不允许 skip 掉 provenance 校验类用例。
- 回归：`python .claude/harness/verify/run.py --level L2`；确认 `tests/test_simulator.py`、`tests/test_bench_runner.py`、`tests/test_cnn_pipeline.py` 不回归。
- 手工验收命令（详见开发文档 7 节）：全量（或超时标注）、预算截断、内存拒绝、TinyFive 缺失四条路径各跑一次并人工检查报告。

### 4.9 范围边界

- **不改** `scratchv/standalone/onnx_to_riscv_standalone.py` 的代码生成、ABI 与内存布局；只读取其 stdout 与输出文件。
- **不改** Spike 相关（课题 24：`spike_sim.py`、`run_spike_bench.py`）与基准套件（课题 06：`benchmarks/`、`bench_runner.py`）。
- **不改** `tinyfive.py` 公共接口；若未来需要 `end=` 精确停机，作为独立后续课题，附 ABI 兼容性说明。
- **LLVM 实测对比**：`llvmlite` 目标机切换、RV64→RV32 差异修复、LLVM 侧可执行镜像的构建属于课题 25/27 交界；本课题只做 ISA 标注与 RV32 统一请求，`llvm_cache_compare.py` 的解析估算继续留在课题 25，不得进入本报告 `measured` 分类。
- **cache 指标**：`cache_model.py` 为分析模型（课题 23）；本课题不引入 cache 命中率类指标，若后续引入必须标 `estimated`。

---

## 五、附录

### 5.1 报告 JSON 示例（全量完成，schema v2）

> 以下数值为结构示例（占位），字段名与嵌套关系为规范值。

```json
{
  "schema_version": "rv32-bench/2",
  "generated_at": "2026-09-14T00:00:00Z",
  "generator": {"script": "rv32_bench.py"},
  "model": {
    "path": "models/graph/cnn.onnx",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "bytes": 27677152,
    "input_name": "input.1",
    "input_shape": [1, 3, 250, 250],
    "output_name": "output",
    "output_shape": [1, 1],
    "initializer_count": 16,
    "weight_bytes": 26700000
  },
  "environment": {
    "python": "3.8.10", "numpy": "1.24.4",
    "tinyfive": "1.0.0", "llvmlite": null
  },
  "targets": {
    "scratchv": {"isa": "rv32im", "abi": "ilp32", "numeric_format": "q16.16"},
    "llvm": {"isa": "rv32imf", "abi": "ilp32", "numeric_format": "float32",
              "triple": "riscv32-unknown-elf", "opt_level": 2}
  },
  "scratchv": {
    "compile": {
      "status": "success", "binary": "_sv.bin", "binary_bytes": 26724336,
      "binary_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
      "code_bytes": 24336, "data_offset": 24336, "data_offset_source": "compiler_stdout",
      "data_bytes": 26700000, "static_insns": 6084, "static_source": "asm_scan",
      "elapsed_s": 12.3
    },
    "static_instruction_mix": {
      "source": "asm_scan",
      "load": 0, "store": 0, "mul": 0, "add": 0, "madd": 0, "branch": 0, "other": 0
    },
    "dynamic": {
      "source": "simulated",
      "simulator": "tinyfive", "simulator_version": "1.0.0",
      "completion": "halted", "limit": null, "executed": 1844674407,
      "timeout_s": 900.0, "elapsed_s": 3210.5,
      "memory_size_bytes": 268435456,
      "input_seed": 42, "input_elements": 187500,
      "halt_addr": 26724336,
      "ops": {"total": 1844674407, "load": 0, "store": 0, "mul": 0,
               "add": 0, "madd": 0, "branch": 0},
      "x_registers_used": 31, "x_usage_total": 123456789,
      "f_registers_used": 0,
      "per_label": null,
      "per_label_note": "tinyfive exe() exposes no per-PC trace",
      "last_error": null
    },
    "output": {"addr": 201326592, "elements": 1, "raw_hex": "0x00018000",
               "q16_16": [1.5], "completion": "halted", "partial": false}
  },
  "llvm": {
    "compile": {"status": "skipped", "reason": "llvmlite not available",
                 "isa_detected": null, "isa_mismatch": false,
                 "static_insns": 0, "static_source": "asm_scan", "elapsed_s": 0.0},
    "dynamic": {"source": "unavailable", "simulator": "tinyfive", "completion": "not_run",
                 "reason": "llvm executable image pipeline not implemented (topic 25 boundary)",
                 "ops": null}
  },
  "comparison": {
    "dynamic_instruction_ratio": null,
    "incomparable_reason": "llvm.dynamic.source=='unavailable'"
  },
  "warnings": [
    "full simulation requested; no automatic instruction-count or wall-clock estimate is available (use --max-instructions N to calibrate); results are marked partial if the wall-clock timeout fires"
  ],
  "errors": []
}
```

### 5.2 报告 JSON 示例（预算截断，schema v2）

```json
{
  "schema_version": "rv32-bench/2",
  "model": {"path": "models/graph/cnn.onnx", "sha256": "0123…cdef"},
  "scratchv": {
    "compile": {"status": "success", "data_offset": 24336, "data_bytes": 26700000,
                 "static_insns": 6084, "static_source": "asm_scan"},
    "dynamic": {
      "source": "simulated", "simulator": "tinyfive", "simulator_version": "1.0.0",
      "completion": "budget_exhausted", "limit": 1000000, "executed": 1000000,
      "timeout_s": 900.0, "elapsed_s": 42.0, "memory_size_bytes": 268435456,
      "input_seed": 42, "halt_addr": 26724336,
      "ops": {"total": 1000000, "load": 0, "store": 0, "mul": 0,
               "add": 0, "madd": 0, "branch": 0},
      "x_registers_used": 12, "x_usage_total": 3456, "f_registers_used": 0,
      "per_label": null, "per_label_note": "tinyfive exe() exposes no per-PC trace"
    }
  },
  "comparison": {
    "dynamic_instruction_ratio": null,
    "incomparable_reason": "scratchv.completion=='budget_exhausted'"
  },
  "warnings": ["budget exhausted at 1000000 instructions; dynamic counts are partial"],
  "errors": []
}
```

### 5.3 非法报告片段与拒绝理由（对照 2.2.4）

```json
{"scratchv": {"dynamic": {"source": "simulated", "completion": "halted",
                           "ops": {"total": 3841}},
               "compile": {"_note": "fallback: static counts only"}}}
```
→ 违规：缺 `simulator/simulator_version/executed/memory_size_bytes/input_seed`；`3841` 是静态计数。

```json
{"comparison": {"dynamic_instruction_ratio": 0.31},
 "scratchv": {"dynamic": {"completion": "budget_exhausted"}},
 "llvm":     {"dynamic": {"completion": "budget_exhausted"}}}
```
→ 违规：截断轨迹不可比。

### 5.4 参考资料

- 设计文档模板：`/root/Lab/ScratchV/设计文档模板.md`
- 课题 27 背景：`docs/topics/27-RV32全量Benchmark.md`
- TinyFive 适配器：`scratchv/simulator/tinyfive.py`（`run(instructions=…, strict=True)`、`get_perf`、`pc`）
- 编译器 ABI/布局：`scratchv/standalone/onnx_to_riscv_standalone.py:2548-2557, 2729-2794, 2802-2807`
- LLVM 模块头：`scratchv/standalone/onnx_to_llvm_standalone.py:446`
- 编码器标签回退语义：`scratchv/backend/riscv_encoder.py:502`
- 相邻课题：课题 06（基准套件）、课题 23（Cache 模型）、课题 24（Spike）、课题 25（LLVM 对比工具）、课题 26（TinyFive 对比）
