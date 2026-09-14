# 课题 27：RV32 全量 Benchmark 开发文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 配套设计：同目录《设计文档.md》（术语、全量定义、诚实报告规范以设计文档为准）  
> 涉及文件：`scratchv/standalone/rv32_bench.py`、`scratchv/standalone/bench_report.py`、`tests/test_rv32_bench.py`（新增）  
> 只读依赖：`scratchv/simulator/tinyfive.py`、`scratchv/standalone/benchmark.py`、`scratchv/standalone/onnx_to_riscv_standalone.py`、`scratchv/backend/riscv_encoder.py`

---

## 一、目标与范围

本课题把 `rv32_bench.py` 从“5000 条指令 + 64KB 内存 + 静态计数兜底”改造成“全量或明确预算 + 真实装载 + 诚实报告”的 RV32 统一基准驱动，并让 `bench_report.py` 按 schema v2 渲染带 provenance 的报告。

范围边界（与设计文档 4.9 一致）：

- **不改** `onnx_to_riscv_standalone.py` 代码生成/ABI/内存布局（只读 stdout 与 `.bin/.s`）。
- **不改** TinyFive 适配器公共接口（只使用既有 `load_binary/load_data/run(strict=True)/pc/get_perf`）。
- **不改** Spike（课题 24）、基准套件（课题 06）、`cache_model.py`（课题 23）、`llvm_cache_compare.py`（课题 25）。
- **LLVM 口径**：本课题只做“ISA 标注 + 统一 RV32 请求 + 跨 ISA 检测拒绝”；LLVM 侧可执行镜像（汇编→链接→ABI→数据段）与实测对比属课题 25/27 交界，本次以 `unavailable` 如实标注，不用解析估算补位。

---

## 二、`rv32_bench.py` 改动清单

### 2.1 改动总表

| # | 现行位置 | 改动 | 目的 |
|---|----------|------|------|
| C1 | `:118` `run_tinyfive(..., n_instructions: int = 10000)` | 替换为 `run_simulation(...)`，`max_instructions` 默认 `0`（全量语义） | 去除默认截断 |
| C2 | `:410-412` 调用点 `n_instructions=5000` | 删除该实参，改为 CLI `--max-instructions/--full` 驱动 | 同上 |
| C3 | `:124` `ProfiledMachine(mem_size=65536)` | 改为 `ProfiledMachine(mem_size=args.mem_size)`（默认 256MiB）+ `compute_layout()` 校验 | 内存容量正确 |
| C4 | 新增 | `parse_data_offset()` / `parse_workspace_bytes()` 解析编译器 stdout（`:2738,:2794`） | 拿到 `data_offset/workspace_size` |
| C5 | 新增 | `load_scratchv_image()`：`load_binary(code_words, 0)` + `load_data(weights, data_offset)` + `load_data(input_blob, 160MiB)` | 权重与输入真实装载 |
| C6 | `:160-184` `_prepare_asm_for_tinyfive` | 删除；新增 `parse_labels()`，标签不再被过滤，`.s` 只用于标签映射与静态计数，不用于装载代码 | 修复分支自跳转 |
| C7 | `:187-234` `_tinyfive_static_fallback` | 删除；静态计数改由 `static_instruction_mix()` 产出，字段独立为 `static_instruction_mix` | 静态数不冒充动态数 |
| C8 | `:118-157` `run_tinyfive` | 替换为分块执行器：`chunk` 循环 + 实例级 `exe` 停机 shim（`halt_addr`/预算先到者停）+ `SIGALRM` 超时 + 完整 provenance 输出 | 停机/预算/超时可控 |
| C9 | `:58-104` `compile_llvm_rv32` | llvmlite 可用时 `llmod.triple="riscv32-unknown-elf"`；新增 `detect_isa_mismatch()`；静态计数只扫 `.text`；失败原因入档 | 统一 RV32 口径与诚实降级 |
| C10 | `:241-312` `BenchResult` / `generate_report` | 改为 `build_report()`（schema v2 dict）+ `bench_report.render_*(report)`；删除硬编码模型描述、无依据 ratio 与 “No analytical estimates” 文案 | 诚实报告 |
| C11 | `:383-425` `main` | 新增 CLI 参数、退出码、`audit_provenance` 写盘前检查、`--quiet` | 接口规格化 |
| C12 | `bench_report.py` 全文 | 新增 `render_markdown/render_html/render_github_summary/render_bench_json/validate_report_schema`；旧 `generate_*` 保留兼容 | 报告字段与模板 |

### 2.2 去默认截断 / 改可配（C1、C2、C11）

现行代码：

```python
def run_tinyfive(asm_path: str, n_instructions: int = 10000) -> dict:   # :118
...
sv.tinyfive = run_tinyfive(str(out / "_sv.s"), n_instructions=5000)     # :411
```

改为：

```python
def run_simulation(
    *,
    asm_path: str,                    # 登记用；代码装载走 binary_path
    binary_path: str,
    data_offset: int,
    workspace_bytes: int,
    input_elements: int,
    output_elements: int,
    max_instructions: int = 0,        # 0 = 全量
    mem_size: int = 268435456,
    timeout_s: float = 900.0,
    chunk_instructions: int = 10_000_000,
    input_seed: int = 42,
) -> dict: ...                        # -> {"dynamic": …, "output": …}; halt_addr 由 compute_layout 内部计算
```

调用点：

```python
sv.dynamic = run_simulation(..., max_instructions=args.max_instructions, ...)
```

CLI 语义：`--max-instructions 0`（默认）与 `--full` 等价；两者与 `--max-instructions N>0` 冲突时 `parser.error(...)` → exit 2。`--full` 存在的意义是抵抗“上游/CI 默认值”污染，显式声明无截断。

### 2.3 内存与数据装载（C3、C4、C5）

1. **stdout 解析**（在 `compile_scratchv` 内，`rc==0` 时）：
   ```python
   DATA_OFFSET_RE = re.compile(r"Data offset:\s*0x([0-9A-Fa-f]+)")
   WORKSPACE_RE   = re.compile(r"Workspace:\s*([\d,]+)\s+bytes")
   CODE_SIZE_RE   = re.compile(r"Code size:\s*([\d,]+)\s+bytes")
   ```
   三个正则分别对应 `onnx_to_riscv_standalone.py:2794`、`:2738`、`:2729`。`compile.scratchv` 返回：
   `{status, binary, binary_bytes, binary_sha256, code_bytes, data_offset, data_offset_source:"compiler_stdout", data_bytes, workspace_bytes, static_insns, static_source:"asm_scan", elapsed_s}`。
   任一正则失配 → `status="failed"`, `error="binary_layout_unparsed"` → main exit 4（未显式给预算时）；禁止“猜 `data_offset = len(code_bytes)`”。
2. **镜像装载**：
   ```python
   binary = Path(binary_path).read_bytes()
   assert data_offset % 4 == 0 and 0 < data_offset < len(binary)
   code_words  = [int.from_bytes(binary[i:i+4], "little")
                  for i in range(0, data_offset, 4)]
   weights     = binary[data_offset:]
   m = ProfiledMachine(mem_size=mem_size)
   m.load_binary(code_words, origin=0)
   m.load_data(weights, data_offset)
   ```
3. **输入构造**（一次性）：`random.Random(input_seed)`，每个元素 `int((r.random() - 0.5) * 0.2 * 65536)`，`struct.pack(f"<{n}i", *vals)`，`m.load_data(blob, 160*1024*1024)`。禁止逐元素 `write_mem_i32`（慢且易错）。
4. **寄存器初始化**：`m.set_reg(2, 128*1024*1024)`、`m.set_reg(10, 160*1024*1024)`、`m.set_reg(11, 192*1024*1024)`、`m.set_reg(1, halt_addr)`；`gp` 由代码自身 `_start` 的 AUIPC 补丁设置，harness 不写。
5. **布局校验**（`compute_layout()`，设计文档 2.1.2 公式；失败抛 `LayoutError` → exit 4，消息含所需最小字节数）。
6. `halt_addr = align_up(binary_bytes, 16)`；要求 `halt_addr + 4 <= mem_size`。

### 2.4 标签解析（C6）

`.s` 格式：标签独占一行（列 0，`name:`），指令行缩进两格并可能带 `# 注释`（`RISCVEmitter.disassemble()`，`onnx_to_riscv_standalone.py:1291-1305`）。

```python
def parse_labels(asm_text: str, expected_code_bytes: int) -> dict[int, str]:
    pc, labels, seen_tokens = 0, {}, []
    for raw in asm_text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.endswith(":") and "(" not in line:
            labels[pc] = line.strip()[:-1]
            continue
        if line.strip().startswith("."):     # 理论不存在；防御性跳过
            continue
        pc += 4
        seen_tokens.append(line.strip())
    if pc != expected_code_bytes:            # 与 data_offset 交叉校验
        raise LabelParseError(f"asm/code-size mismatch: {pc} != {expected_code_bytes}")
    return labels
```

关键点：

- **保留标签**（现行 `:174-176` 丢弃标签，`label_map` 未用，导致 `riscv_encoder.py:502` 的 `labels.get(label, current_idx)` 把分支解析为自跳转）。
- 代码装载走 `load_binary`（编译器自己的二进制），不再把 `.s` 喂给 `assemble_to_binary`；`.s` 仅作标签映射与静态 scan。
- `_start` 必须映射 `pc=0`；`_done` 必须存在；`static_insns == data_offset // 4` 作为不变量断言。

### 2.5 失败路径（C7、C11）

| 失败 | 检测点 | 行为 |
|------|--------|------|
| TinyFive 未安装 | `m.available is False` | 无 `--allow-missing-simulator` → exit 3；有 → `dynamic.source="unavailable"`、`ops=null`、`completion="not_run"`，报告只含 static |
| 内存布局不满足 | `compute_layout()` | exit 4，`errors=["memory_layout_invalid: need >= N bytes"]`，不落盘动态报告 |
| `data_offset` 解析失败 | `compile_scratchv` stdout 正则 | exit 4，`reason="binary_layout_unparsed"` |
| 标签/代码尺寸不一致 | `parse_labels` | 抛 `LabelParseError` → exit 4 |
| 助记符预检失败 | 子集扫描 `.s` vs TinyFive 支持表 | `completion="not_run"`，exit 4，列出不支持助记符 |
| 预算耗尽 | 循环额度判断 | `completion="budget_exhausted"`，`executed==limit`，ratio `null`，exit 0（显式预算属正常） |
| 超时 | `SIGALRM` + `_timed_out` 标志 | `completion="timeout"`，保留部分 ops，exit 0（`--fail-on-incomplete` 时 exit 7） |
| `m.last_error` 非空且非超时 | 适配器 `strict=True` 抛错 | `source="unavailable"` + `completion="error"`，`ops=null`，`output.partial=true`，exit 1 |
| 未完成且要求严格 | `--fail-on-incomplete` | exit 7 |

静态兜底已删除：`static_instruction_mix` 只写入 `scratchv.static_instruction_mix`，字段名带 `static_`，永不出现在 `dynamic.ops`。

### 2.6 LLVM 侧改动（C9）

1. `binding.Target.from_triple("riscv32-unknown-elf")`；`ImportError`（llvmlite 未安装）→ `status="skipped"`，`reason` 记录原始异常；成功导入后的目标机/IR/发射阶段异常（如 `RuntimeError`）→ `status="failed"`，`reason` 记录真实原因，`main` 追加 warning 并把它写入 `llvm.dynamic.reason`（不再统一硬编码为 “pipeline not implemented”）。llvmlite 当前环境未安装，该分支即 `skipped`。
2. 成功路径。解析 IR 文本后，用 llvmlite API 覆盖模块头：
   ```python
   llmod = binding.parse_assembly(ir_text)
   llmod.triple = "riscv32-unknown-elf"          # 覆盖 onnx_to_llvm_standalone.py:446 的 riscv64 头
   llmod.data_layout = str(tm.target_data)       # 与目标机一致，避免指针宽度假设漂移
   llmod.verify()
   asm = tm.emit_assembly(llmod)
   ```
3. `detect_isa_mismatch(asm_text) -> list[str]`：扫描助记符集合 `{ld, sd, lwu, addw, subw, addiw, sllw, srlw, sraw, slliw, srliw, sraiw, mulw, divw, divuw, remw, remuw, fld, fsd, fcvt.l.s, fcvt.s.l}`；命中 → `isa_detected="riscv64"`、`isa_mismatch=true`，`llvm.dynamic` 不运行。
4. 静态计数：遇到 `.section .rodata`（权重数组，`onnx_to_llvm_standalone.py:132-157` 的 `private constant [N x float]`）停止计数；`.word/.long/.byte/.float` 不算指令。`static_source="asm_scan"`。
5. 不构造 LLVM 侧可执行镜像；`llvm.dynamic={source:"unavailable", completion:"not_run", reason:"llvm executable image pipeline not implemented (topic 25 boundary)", ops:null}`。

---

## 三、`bench_report.py` 报告字段与模板

### 3.1 新增函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `render_markdown` | `(report: dict) -> str` | schema v2 → Markdown |
| `render_html` | `(report: dict) -> str` | Markdown 内容包进既有 CSS 壳 |
| `render_bench_json` | `(report: dict) -> str` | 规范化 JSON（`sort_keys=False, indent=2`） |
| `render_github_summary` | `(report: dict) -> str` | CI 摘要，只引用 measured/static 分区 |
| `validate_report_schema` | `(report: dict) -> list[str]` | 返回缺失/类型错误列表；空列表为通过 |

旧 `generate_html_report/generate_json_report/generate_github_summary` 保留为兼容壳（内部转调 `render_*` 或维持原行为），避免影响 `onnx_to_riscv_standalone.py --report` 与 CI（课题 06/30）。

### 3.2 Markdown 模板（骨架）

```markdown
# RV32 Benchmark Report — {model.path}
- Generated: {generated_at} | schema: rv32-bench/2
- Model: sha256={model.sha256[:12]} | bytes={model.bytes} | input={model.input_shape}
- Targets: ScratchV {targets.scratchv.isa}/{targets.scratchv.numeric_format}
           LLVM {targets.llvm.isa}/{targets.llvm.numeric_format} (opt={targets.llvm.opt_level})
- Environment: python={environment.python} numpy={environment.numpy}
               tinyfive={environment.tinyfive} llvmlite={environment.llvmlite}

## 1. Compilation [static]
| Metric | ScratchV | LLVM |
| status | … | … |
| code bytes | … | … |
| static insns [asm_scan] | … | … |

## 2. Dynamic Execution [measured]
| Metric | ScratchV | LLVM |
| completion | halted | not_run |
| executed | … | — |
| ops.total | … | — |
| load/store/mul/add/madd/branch | … | — |
> LLVM dynamic unavailable: {llvm.dynamic.reason}

## 3. Comparison [measured]
- dynamic_instruction_ratio: **null** — {comparison.incomparable_reason}
（仅当两侧 source=="simulated" 且 completion=="halted" 时打印比值行，否则打印 null + 原因）

## 4. Analytical Warnings [estimated]
- {warnings[*]}

## Provenance
Simulated by tinyfive {environment.tinyfive} | completion={completion} | executed={executed}
| limit={limit} | memory={memory_size_bytes} | seed={input_seed} | halt=0x{halt_addr:x}
| model sha256={model.sha256} | binary sha256={scratchv.compile.binary_sha256}
output [measured | measured/partial | unavailable]: {output.raw_hex}
| elements={output.elements} | completion={output.completion}
```

渲染规则：

- 每个表标题必带 `[measured]/[static]/[estimated]/[unavailable]` 之一；同一表格中不得混用来源类别。
- 页脚逐字取 provenance 字段，禁止出现“all metrics from simulation / no analytical estimates”之类无法由字段证明的断言。
- 模型名与层结构只能来自 `model.*`；删除现行 `rv32_bench.py:268` 的硬编码描述。
- `ops` 任一计数为 0 时照实显示 `0`，不得显示 `—` 掩盖“未跑到该类别”。

### 3.3 报告字段总表（dotted path）

| 字段 | 类型 | 来源分类 | 说明 |
|------|------|----------|------|
| `schema_version` | str | — | 固定 `"rv32-bench/2"` |
| `generated_at` | str | — | UTC ISO8601 |
| `generator.script` | str | — | `"rv32_bench.py"` |
| `model.path/sha256/bytes/input_name/input_shape/output_name/output_shape/initializer_count/weight_bytes` | — | static | 模型身份 |
| `environment.python/numpy/tinyfive/llvmlite` | str\|null | — | 依赖版本 |
| `targets.scratchv.{isa,abi,numeric_format}` | str | — | `rv32im/ilp32/q16.16` |
| `targets.llvm.{isa,abi,numeric_format,triple,opt_level}` | — | — | `rv32imf/ilp32/float32` |
| `scratchv.compile.status/binary/binary_bytes/binary_sha256/code_bytes/data_offset/data_offset_source/data_bytes/workspace_bytes/static_insns/static_source/elapsed_s` | — | static | 编译与镜像 |
| `scratchv.static_instruction_mix.{source,load,store,mul,add,madd,branch,other}` | int | static | 静态助记符分布 |
| `scratchv.dynamic.{source,simulator,simulator_version,completion,limit,executed,timeout_s,elapsed_s,memory_size_bytes,input_seed,input_elements,halt_addr,ops{total,load,store,mul,add,madd,branch},x_registers_used,x_usage_total,f_registers_used,per_label,per_label_note,last_error}` | — | measured | 动态执行 |
| `scratchv.output.{addr,elements,raw_hex,q16_16,completion,partial}` | — | measured | 输出读出；`q16_16` 恒为 list（不可用时为 null）；`completion != "halted"` 时 `partial=true`，Markdown/HTML 的 Provenance 段落标 `[measured/partial]` |
| `llvm.compile.{status,reason,isa_detected,isa_mismatch,static_insns,static_source,elapsed_s}` | — | static/unavailable | LLVM 侧 |
| `llvm.dynamic.{source,simulator,completion,reason,ops}` | — | unavailable | 失败必须 `ops=null` |
| `comparison.{dynamic_instruction_ratio,incomparable_reason}` | float\|null | measured 派生 | 比值规则见设计文档 2.1.1 |
| `warnings[]` / `errors[]` | list[str] | — | 预估告警与错误 |

---

## 四、与 `benchmark.py` / `ProfiledMachine` 的接口

### 4.1 `ProfiledMachine`（`scratchv/simulator/tinyfive.py`，只读使用）

| 用法 | 契约 |
|------|------|
| `ProfiledMachine(mem_size)` | `mem_size` 为字节数；构造即 `np.zeros(mem_size, uint8)`；256MiB 默认值可接受（本机 numpy 1.24.4） |
| `.available` | `False` 时只能走 `unavailable` 路径，不得继续 `load_*` |
| `load_binary(words, origin=0)` | `words: list[int]` 32 位词；越界抛 `ValueError` |
| `load_data(data: bytes, addr)` | `np.frombuffer` 切片赋值；用于权重（26MB 一次性）与输入 |
| `run(instructions=N, strict=True)` | 适配器内部 `finally` 更新 `instr_count`；异常包装为 `RuntimeError(last_error)` 继续抛出，ops 保留 |
| `.pc` / `.get_perf()` | `get_perf()` 返回 `{total,load,store,mul,add,madd,branch}` 累计值；分块循环用 `pc` 判停机 |
| `.set_reg(idx, value)` | `0 < idx < 32`；本课题写 x1/x2/x10/x11 |
| `.read_mem_i32(addr)` | 输出读取；不越界返回 0 |

不使用：`load_asm`（标签语义不可靠）、`print_perf`、`StubProfiledMachine`。

### 4.2 `benchmark.py`（只读使用）

- `estimate_cnn_model(model_spec=None) -> dict`：**未接入 `rv32_bench.py`**（本课题不做自动预估）；仅 `benchmark.py` 自身 CLI 使用。若未来引入，其字段必须进入 `estimated` 分类，前缀 `estimated_`，不得写入 `dynamic`。
- `RV32EmulatorFast` / `run_benchmark`：仅测试用例 1 作为独立功能对照（`load_unified_binary(binary, code_size_base=..., load_addr=0)` + `run(max_instr=...)`）；其计数分类（`Cat_*`）与 TinyFive 不同，只对照 `total/load_count/store_count/branch_total`。若未来作为正式数据源，须以 `simulator="rv32_emulator_fast"` 独立字段呈现，不与 TinyFive 混算。
- `estimate_cnn_instructions` 的 per-MAC 常量（`CONV_INSNS_PER_MAC=8` 等）属解析模型，禁止用于 `measured`。

### 4.3 `cache_model.py`（不在运行时链路）

`CacheSim`/`create_cache_pair` 为分析模型；本课题报告不包含 cache 指标。若后续引入，字段必须放 `estimated` 分类并标注 `model="cache_model"`。

---

## 五、接口契约

### 5.1 CLI（精确名称）

```
python scratchv/standalone/rv32_bench.py MODEL
    [--output-dir DIR]                    # 默认 benchmark_reports
    [--html FILE] [--json FILE] [--md FILE]   # 默认 rv32_bench.{html,json,md}
    [--max-instructions N]                # int，默认 0；0=全量，>0=预算
    [--full]                              # 无截断；与 N>0 互斥
    [--mem-size BYTES]                    # int，默认 268435456
    [--timeout SECONDS]                   # float，默认 900.0
    [--chunk-instructions N]              # int，默认 10000000
    [--input-seed N]                      # int，默认 42
    [--skip-llvm]                         # flag
    [--allow-missing-simulator]           # flag
    [--fail-on-incomplete]                # flag
    [--quiet]                             # flag
```

退出码：`0` 成功；`1` 未预期错误/审计失败；`2` 参数冲突；`3` 仿真器不可用；`4` 布局/镜像/预检失败；`7` 未完成且 `--fail-on-incomplete`。

### 5.2 函数（精确签名）

```python
# rv32_bench.py
SCHEMA_VERSION = "rv32-bench/2"
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_SIMULATOR, EXIT_LAYOUT, EXIT_INCOMPLETE = 0, 1, 2, 3, 4, 7

class LayoutError(RuntimeError): ...
class LabelParseError(RuntimeError): ...
class SimulationTimeout(RuntimeError): ...

def sha256_file(path: str | Path) -> str
def parse_data_offset(stdout: str) -> int | None
def parse_workspace_bytes(stdout: str) -> int | None
def parse_labels(asm_text: str, expected_code_bytes: int) -> dict[int, str]
def static_instruction_mix(asm_text: str) -> dict          # 键: source,load,store,mul,add,madd,branch,other
def compute_layout(*, data_offset: int, data_size: int, workspace_bytes: int,
                   input_elements: int, output_elements: int,
                   mem_size: int) -> dict                  # 键: sp,input_addr,output_addr,halt_addr,mem_size
def load_scratchv_image(binary_path: str, data_offset: int) -> tuple[list[int], bytes]
def build_input_q16(elements: int, seed: int) -> bytes
def run_simulation(*, asm_path: str, binary_path: str, data_offset: int,
                   workspace_bytes: int, input_elements: int, output_elements: int,
                   max_instructions: int = 0, mem_size: int = 268435456,
                   timeout_s: float = 900.0, chunk_instructions: int = 10_000_000,
                   input_seed: int = 42) -> dict           # -> scratchv.dynamic + output
def compile_scratchv(onnx_path: str, output_bin: str, output_asm: str,
                     timeout_s: float = 120.0) -> dict
def compile_llvm_rv32(onnx_path: str, output_asm: str, *,
                      triple: str = "riscv32-unknown-elf", opt_level: int = 2) -> dict
def detect_isa_mismatch(asm_text: str) -> list[str]
def build_report(model: dict, environment: dict, scratchv: dict, llvm: dict) -> dict
def audit_provenance(report: dict) -> list[str]
def main(argv: list[str] | None = None) -> int

# bench_report.py
def render_markdown(report: dict) -> str
def render_html(report: dict) -> str
def render_bench_json(report: dict) -> str
def render_github_summary(report: dict) -> str
def validate_report_schema(report: dict) -> list[str]
```

### 5.3 报告字段（精确名称）

一级键：`schema_version, generated_at, generator, model, environment, targets, scratchv, llvm, comparison, warnings, errors`。

二级/三级键以设计文档 5.1 与本文 3.3 为准，关键枚举：

- `scratchv.dynamic.source ∈ {"simulated","unavailable"}`
- `scratchv.dynamic.completion ∈ {"halted","budget_exhausted","timeout","error","not_run"}`
- `llvm.dynamic.source ∈ {"simulated","unavailable"}`
- `comparison.dynamic_instruction_ratio: float | null`（仅两侧 `simulated`+`halted`）
- `*.static_source == "asm_scan"`

---

## 六、测试文件与用例

测试文件：`tests/test_rv32_bench.py`（新增）。

| 用例 | 名称 | 要点 |
|------|------|------|
| T1 | `test_full_run_small_model_matches_reference` | `onnx.helper` 造迷你 CNN → 全量 `halted`；与 `RV32EmulatorFast` 对照 `total/load/store/branch` 精确相等；重跑确定性 |
| T2 | `test_budget_exhausted_is_labeled` | `--max-instructions 1000` → `budget_exhausted`、`executed==limit==1000`、ratio `null`、`--fail-on-incomplete` 退出 7 |
| T3 | `test_report_requires_provenance` | monkeypatch TinyFive 不可用 + `--allow-missing-simulator` → `dynamic.source=="unavailable"`、`ops is null`、static 分区存在；无 flag 时退出 3 |
| T4 | `test_memory_layout_validation` | `--mem-size 1048576` → exit 4，stderr 含 `memory_layout_invalid`，不落动态报告 |
| T5 | `test_parsed_labels_cover_all_branches` | 分支/跳转目标全部有标签；`static_insns == data_offset//4`；缺标签抛 `LabelParseError` |
| T6 | `test_llvm_riscv64_flagged_isa_mismatch` | 含 `ld/sd/addiw` 的 fixture → `isa_mismatch=true`、ratio `null` |
| T7 | `test_audit_provenance_rejects_static_fallback` | 构造非法报告（设计文档 5.3 两例）→ `audit_provenance()` 非空 |
| T8 | `test_bench_report_schema_required_keys` | `validate_report_schema()` 对必填键逐一断言（无需 TinyFive，不得 skip） |

运行：

```bash
python -m pytest tests/test_rv32_bench.py -q
make test
```

无 TinyFive 环境下 T1/T2 走 `pytest.importorskip("tinyfive")`；T3–T8 必须运行。

---

## 七、验收标准

1. 全量路径：`python scratchv/standalone/rv32_bench.py models/graph/cnn.onnx --full --mem-size 268435456 --timeout 1800 --output-dir /tmp/rv32_full`
   - 成功产出 3 格式报告；`scratchv.dynamic.completion ∈ {"halted","timeout"}`；
   - 若 `halted`，`executed` 为全量真实数（不受 5000/10000 限制）；若 `timeout`，报告明示部分轨迹且 `comparison=null`；
   - `audit_provenance(report) == []`；`validate_report_schema(report) == []`。
2. 预算路径：`--max-instructions 1000000` → `completion=="budget_exhausted"`、`limit==executed==1000000`、ratio `null`；报告含 `[measured/budget]` 标识。
3. 任何输出（Markdown/HTML/JSON）包含：`model.sha256`、`environment.tinyfive`、`input_seed`、`memory_size_bytes`、`completion`、`static_source`；不再出现 “All metrics sourced from TinyFive … No analytical estimates” 这类无字段支撑的断言。
4. LLVM 侧：无 llvmlite → `status=="skipped"`、`dynamic.source=="unavailable"`、报告不打印比值数字；有 llvmlite → 模块 triple 为 `riscv32-unknown-elf`，`.s` 若含 RV64 助记符则 `isa_mismatch==true`。
5. 布局失败：`--mem-size` 不足时 exit 4 且不产生虚假动态数据。
6. 回归：`make test` 全绿；`python .claude/harness/verify/run.py --level L2` 通过；未改动编译器/Spike/基准套件代码。
7. 文档一致性：报告字段与本文 3.3/5.3 完全同名；CLI 与 5.1 完全同名。

---

## 八、风险与回退

| # | 风险 | 影响 | 缓解/回退 |
|---|------|------|-----------|
| R1 | TinyFive 吞吐低（Python 解释执行），全量 18.5 亿指令可能需数小时 | 全量不可行 | `--max-instructions` 预算并如实标注；**未实现自动预估**（无 1M 探测/`estimate_cnn_model` 接入），full 模式启动前只写一条“无自动预估、可用预算校准”的 warning；超时/预算中断的结果标 `partial` |
| R2 | 256MiB numpy 内存 + 26MB 权重 | 内存压力 | 布局不可压缩（ABI 地址固定）；内存不足只能拒绝并提示，禁止缩容假装成功 |
| R3 | `ra` 被生成代码内部 `jal ra, …` 覆盖，`halt_addr` 永不命中 | 无法 `halted` | `completion` 如实标 `budget_exhausted/timeout`；`--fail-on-incomplete` 供 CI |
| R4 | 平台无 `SIGALRM` | 超时保护缺失 | 退化为要求 `--max-instructions`；否则拒绝启动 |
| R5 | TinyFive 不支持某助记符导致 PC 不前进（卡死） | 挂起 | 启动前助记符白名单预检，`not_run` + exit 4 |
| R6 | `.s` 标签格式未来漂移 | 标签解析失败 | `parse_labels` 与 `data_offset//4` 交叉校验，失败即抛错；T5 固定回归 |
| R7 | llvmlite 未安装/目标不支持 | LLVM 侧不可测 | `skipped`+`unavailable` 如实标注；不引入解析估算 |
| R8 | LLVM IR 为 riscv64 假设（指针宽度/全局） | 强行 RV32 代码错误 | `llmod.triple/data_layout` 覆盖 + `verify()` + RV64 助记符检测；失败标 `isa_mismatch` |
| R9 | 工作区 `[sp, sp+workspace)` 与 input@160MiB 冲突 | 静默数据损坏 | `compute_layout()` 的 GUARD 校验，不满足 exit 4 |
| R10 | 静态兜底被误用回动态展示 | 报告失真 | 字段拆分为 `static_instruction_mix`；`audit_provenance()` 阻断；T7 回归 |

回退策略：若不满足验收（如全量在可接受时间内无法完成），保留本设计的所有“诚实标注”改动，仅把默认改为 `--max-instructions` 显式预算模式，并在报告与 CI 摘要中显著标注 `budget_exhausted`；任何情况下不得恢复“静态计数 + 无 provenance”的报告形态。

---

## 实现结果（2026-09-14 集成）

> **集成 commit**：`12685db`（`feat(topic27): full RV32 benchmark with honest provenance and budget controls`）
> **集成位置**：`Seven_big_summary` 上第 7 个 topic commit（顺序 … → 24 → **27** → 10 → …）
> **集成后全量**：`PYTHONPATH=. python3.11 -m pytest tests/ -q` → **1011 passed / 13 xfailed / 20 xpassed / 0 failed**

### 实现文件与要点

| 文件 | 要点 |
|------|------|
| `scratchv/standalone/rv32_bench.py` | 重写：默认全量 + `--max-instructions` 预算、`audit_provenance`、`compute_layout` 的 GUARD 校验 |
| `scratchv/standalone/bench_report.py` | 报告 schema v2（`model.sha256` / `environment.tinyfive` / `completion` / `static_source` 等） |
| `tests/test_rv32_bench.py` | 9 个新用例 |

### 测试数字

| 口径 | 结果 |
|------|------|
| 定向（`tests/test_rv32_bench.py`） | 9 passed |
| 分支全量（cherry-pick 前） | 574 passed |
| 集成后全量 | 1011 passed / 13 xfailed / 20 xpassed / 0 failed |

### 与本文档的偏差 / 未完成项

- cnn 全量实测**未跑**（TinyFive 约 7 万 instr/s，预计约 7 小时），只做预算中断干跑（`budget_exhausted`）。

### 已知限制

- 对 TinyFive 机器实例的 NumPy 2.x `LW/LH` 兼容 shim 属课题 26 追修（`tinyfive.py` 未改）。
- 停机采用实例级 `exe` 重绑 shim（`halt_addr` 与指令预算先到者停），同样未改 `tinyfive.py`；设计文档 2.1.4 已登记该机制。
- 错误路径统一用 `source=unavailable` 表达（而非 `dynamic=null`），`output.partial=true`。
- LLVM 侧无 llvmlite 时 `status=skipped`；导入后目标机/IR 失败为 `status=failed` 并带真实 `reason`（main 写 warning）。两种情况都不打印比值数字。
