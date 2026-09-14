# ScratchV Spike 仿真工具路径可移植化与降级策略 设计文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 涉及模块：`scratchv/standalone/spike_sim.py`（Spike 工具解析/调用/输出解析/报告）、`scratchv/standalone/run_spike_bench.py`（纯仿真降级后端，使用方适配）  
> 功能范围：spike / spike-dasm / spike-log-parser 三工具的可移植路径解析、环境变量契约、工具缺失时的降级与报错分层、Spike 输出解析约定、报告状态字段；**不含**周期模型、统计口径、TinyFive 路径的修改

---

## 一、功能介绍

### 1.1 功能概述

`spike_sim.py` 的职责是：把 ScratchV 生成的 flat binary 包装成最小 ELF32，调用 Spike（RISC-V 黄金参考模型）执行，解析 commit/cache 输出并生成报告。它依赖三个外部工具：

| 工具 | 用途 | 当前状态 |
|------|------|---------|
| `spike` | 真正执行 ELF | 必需（缺失则无法仿真） |
| `spike-dasm` | 反汇编指令、供指令分类 | 已定义常量但代码路径尚未使用 |
| `spike-log-parser` | 解析 commit log（trace/热点） | 已定义常量但代码路径尚未使用 |

当前问题（`spike_sim.py:37-39`）：

```python
SPIKE = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike"
SPIKE_DASM = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike-dasm"
SPIKE_LOG_PARSER = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike-log-parser"
```

- 三个路径是个人机器绝对路径，换机器后必然 `FileNotFoundError`（`run_spike` 捕获后返回 `exit_code=-2`），但：
  - 无法区分「本机没有 Spike（应跳过）」与「Spike 跑了但失败（应报错）」；
  - 无 CLI/环境变量覆盖入口，CI 或别的开发者无法在不改代码的前提下指定自己的 Spike；
  - `spike-dasm` / `spike-log-parser` 缺失时语义未定义（当前代码甚至不使用它们）。
- 解析层（commit 统计、cache 统计、PC 直方图）与运行层耦合在 `run_spike()` 内联实现，缺失段落只能静默得到 0，没有任何告警信号。

本功能要做的事：

1. **路径解析器**：按固定优先级自动解析三工具路径（CLI > 环境变量 > `SCRATCHV_SPIKE_HOME` > `PATH` > 常见安装目录 > 旧的硬编码常量回退）。
2. **降级/报错分层**：默认情况下「工具不存在」是 skip（带原因），「显式配置无效」「`--require-spike` 且工具不存在」是硬失败，二者退出码不同；`spike-dasm` / `spike-log-parser` 缺失只降级为告警。
3. **解析健壮化**：把内联解析抽成纯函数，容忍千分位数字与缺失段落，并把异常记录进解析告警。
4. **可观测性**：`SpikeResult`、文本报告、JSON 报告增加 `status` / `skip_reason` / `spike_binary` / 告警字段。
5. **使用方适配**：`run_spike_bench.py`（当前无 Spike 时的纯 Python 降级后端）增加后端标注与工具可用性探测，不改变任何统计数据。

### 1.2 设计目标

- **任何机器可 import / `--help`**：模块导入与参数解析阶段不探测、不调用、不写任何外部工具。
- **单点可覆盖**：`--spike-bin` / `SCRATCHV_SPIKE_BIN` 在任意机器上都能强制指定，优先级最高。
- **失败语义清晰**：配置错误（退出码 2）≠ 运行失败（退出码 1）≠ 环境不具备的合法跳过（退出码 0 且 `status=skipped`）。
- **零外部依赖**：只用 `os` / `shutil` / `subprocess` 等标准库，不新增 pip 包。
- **向后兼容**：三个模块常量保留、`run_spike()` 旧调用参数保持不变、JSON 既有字段不删除；`ci_benchmark.py` 对 `run_spike_bench.run_emulator_with_caches` 的直接函数调用不受影响。
- **降级可观测**：所有跳过与告警都带机器可读字段（`status`、`skip_reason`、`tool_warnings`、`parse_warnings`），并与 `tinyfive_compare.py` 现有 `TINYFIVE_AVAILABLE` / `_fallback` 模式保持风格一致。

---

## 二、设计规范

### 2.1 路径解析优先级

对每个工具（`spike`、`spike-dasm`、`spike-log-parser`）独立解析，规则等价 BNF：

```
resolve(tool)      ::= cli(tool)
                     | env(tool)
                     | spike_home(tool)
                     | path(tool)
                     | common(tool)
                     | legacy(tool)
                     | MISSING

cli(tool)          ::= PATH                       -- 来自 --spike-bin / --spike-dasm / --spike-log-parser
env(tool)          ::= PATH                       -- 来自 SCRATCHV_SPIKE_BIN / _DASM / _LOG_PARSER
spike_home(tool)   ::= $SCRATCHV_SPIKE_HOME "/bin/" tool
path(tool)         ::= shutil.which(tool)
common(tool)       ::= dir "/" tool   for dir in COMMON_SPIKE_DIRS
legacy(tool)       ::= SPIKE | SPIKE_DASM | SPIKE_LOG_PARSER   -- 旧硬编码常量，仅当可执行时生效

accept(candidate)  ::= os.path.isfile(candidate) AND os.access(candidate, os.X_OK)
```

| 层级 | 来源 | 显式程度 | 候选无效时的行为 |
|------|------|---------|-----------------|
| 1 | CLI 参数 | 显式（单次运行意图） | `spike`：**硬失败**（抛配置错误，退出码 2）；`spike-dasm` / `spike-log-parser`：**告警**后继续下一层 |
| 2 | 专用环境变量 | 显式（可能跨项目残留） | **告警**：记入 `tool_warnings`，继续下一层 |
| 3 | `SCRATCHV_SPIKE_HOME/bin/<tool>` | 隐式（目录提示） | 静默继续（仅在候选列表中留痕） |
| 4 | `PATH` 自动探测 | 隐式 | 静默继续 |
| 5 | 常见安装目录 | 隐式 | 静默继续 |
| 6 | 旧常量回退 | 隐式（legacy） | 静默继续 |
| — | 全部失败 | — | 返回 `path=None, source="missing"` |

**关键规则**：

- CLI 与 env 的路径值一律先 `strip()`，空串按「未设置」处理，不报错。
- env 值支持 `~` 展开（`os.path.expanduser`），不做 `$VAR` 二次展开。
- 每个工具的解析互不影响：设置了 `SCRATCHV_SPIKE_DASM` 不会影响 `spike` 的解析链。
- 返回的 `source` 取值固定为 `cli | env | spike_home | path | common | legacy | missing`，写入 `SpikeTools.sources`，供测试与报告断言。
- 解析器必须在**调用时**读取模块常量（而非 import 时固化），以便测试通过 monkeypatch 覆盖 legacy 回退。

### 2.2 环境变量命名

| 变量名 | 语义 | 示例 | 生效层级 |
|--------|------|------|---------|
| `SCRATCHV_SPIKE_BIN` | spike 可执行文件路径 | `/opt/riscv/bin/spike` | 2 |
| `SCRATCHV_SPIKE_DASM` | spike-dasm 可执行文件路径 | `/opt/riscv/bin/spike-dasm` | 2 |
| `SCRATCHV_SPIKE_LOG_PARSER` | spike-log-parser 路径 | `/opt/riscv/bin/spike-log-parser` | 2 |
| `SCRATCHV_SPIKE_HOME` | 安装根目录（约定 `<HOME>/bin/<tool>`） | `/opt/coralnpu-spike-rv32` | 3 |

**命名约定**：统一前缀 `SCRATCHV_SPIKE_`；大小写敏感；无缩写；不引入 `SPIKEDASM` 等变体。新增变量不得与既有 `SPIKE`/`SPIKE_DASM`/`SPIKE_LOG_PARSER` 模块常量同名。

### 2.3 工具缺失时的降级/报错策略

**分层原则**：环境不具备（未显式配置、工具真的不存在）→ 跳过并说明原因；用户显式配置了错误路径 → 立即失败；仿真已启动后的异常（超时/非零退出）→ 运行失败。

| # | 场景 | 默认行为 | `--require-spike` | 退出码 | 报告状态 |
|---|------|---------|-------------------|--------|---------|
| 1 | spike 缺失 | `SKIP:` + 搜索位置 + 修复提示；`--json` 时输出 `status=skipped`、`exit_code=-2` 的 JSON | `ERROR:` + 搜索位置 | 0 / 2 | `skipped` / 无报告（仅 stderr） |
| 2 | spike-dasm 缺失 | `WARNING:`（本模块尚未使用该工具） | 同默认（不升级为失败） | 0 | `tool_warnings` |
| 3 | spike-log-parser 缺失 | `WARNING:`（本模块尚未使用该工具） | 同默认 | 0 | `tool_warnings` |
| 4 | `--spike-bin` 无效（不存在/不可执行/是目录） | `ERROR:` | `ERROR:` | 2 | 无报告（仅 stderr） |
| 4b | `--spike-dasm` / `--spike-log-parser` 无效（可选工具） | `WARNING:` + 继续解析 | 同默认 | 0 或后续结果 | `tool_warnings` |
| 5 | env 显式路径无效 | `WARNING:` + 继续解析 | 同默认 | 0 或后续结果 | `tool_warnings` |
| 6 | Spike 超时（`subprocess.TimeoutExpired`） | `ERROR:` | 同默认 | 1 | `timeout` |
| 7 | Spike 非零退出 / 启动失败（含 `OSError`，如无法 exec 的文件） | `ERROR:` | 同默认 | 1 | `failed` |
| 8 | 输出缺少**部分** cache/commit 段 | 继续，零值 + 告警 | 同默认 | 0 | `parse_warnings` |
| 8b | spike 退出 0 但**完全无可解析统计段**（疑似非 Spike 可执行文件） | `status=failed`、`exit_code=-2` + 告警 | 同默认 | 1 | `failed` + `parse_warnings` |
| 9 | import / `--help` | 永不探测外部工具 | 同默认 | 0 | — |

**退出码契约（模块常量）**：

```
EXIT_OK     = 0   # 成功，或合法跳过（status=skipped）
EXIT_RUN_FAIL = 1 # 仿真已启动但失败：超时、非零退出、binary 缺失（历史兼容）
EXIT_CONFIG = 2   # 配置错误：CLI 显式路径无效、--require-spike 且 spike 缺失
```

**消息前缀约定**（stderr，供脚本 grep）：

```
SKIP: spike binary not found.
  searched: --spike-bin, SCRATCHV_SPIKE_BIN, SCRATCHV_SPIKE_HOME/bin/spike, PATH, <常见目录...>, legacy
  hint: export SCRATCHV_SPIKE_BIN=/path/to/spike
ERROR: spike binary not found (--require-spike).
WARNING: SCRATCHV_SPIKE_BIN=/old/path/spike is not executable; ignored
```

### 2.4 Spike 输出解析约定

解析层拆为三个纯函数（输入字符串、输出结构化数据），不依赖子进程，便于用固定样本测试：

| 数据 | 数据流 | 匹配模式（容忍空白/千分位） | 缺失/异常行为 |
|------|--------|---------------------------|--------------|
| 提交指令数 | stderr | `(Commited|Committed)\s+([\d,]+)\s+instructions` | 0 + `parse_warnings` |
| 仿真速度 | stderr | `([\d.]+)\s*MIPS` | 0.0 |
| I$ / D$ 统计 | stderr | 段头 `^I\$:` / `^D\$:`；行内 `hits:\s*([\d,]+)`、`misses:\s*([\d,]+)`、`miss rate:\s*([\d.]+)%` | 0 / 0.0 + `parse_warnings` |
| PC 直方图 | stdout | 段头含 `PC histogram`；数据行 `0x[0-9a-fA-F]+:\s*\d+`；空行结束 | 空 dict，报告省略该小节 |
| commit log | 日志文件 | 本期原样透传（`run_spike_with_log` 不解析） | `log_content=""` |

**容错规则**：

- 数字中的 `,` 一律去除后再 `int()`；非法数字跳过并记 `parse_warnings`，不抛异常。
- 未知行忽略；段头缺失时按上表「缺失行为」处理，绝不把解析失败升级为进程失败。
- `spike` 的已知拼写错误 `Commited` 与修正拼写 `Committed` 均接受（正则二选一）。
- 现有拼写字段 `SpikeResult.commited_insns_per_sec` 不改名，避免破坏已有消费方。

### 2.5 配置合法性规则与示例

**合法配置示例**（均不产生硬失败）：

| # | 配置 | 解析结果 |
|---|------|---------|
| 1 | `--spike-bin /opt/riscv/bin/spike`（文件可执行） | `source=cli` |
| 2 | `SCRATCHV_SPIKE_BIN=$HOME/riscv/bin/spike` | `source=env` |
| 3 | `SCRATCHV_SPIKE_HOME=/opt/coralnpu-spike-rv32`，且 `.../bin/spike` 存在 | `source=spike_home` |
| 4 | 什么都不设，但 `spike` 在 `PATH` 上 | `source=path` |
| 5 | 什么都不设，但 `/usr/local/bin/spike` 存在 | `source=common` |
| 6 | 什么都不设且处处不存在 | `source=missing` → 默认 skip（退出码 0） |

**非法配置示例**（前 4 项必须报错/告警，不得静默成功）：

| # | 配置 | 处理 |
|---|------|------|
| 1 | `--spike-bin ./spike`（不存在） | `SpikeConfigError` → 退出码 2 |
| 2 | `SCRATCHV_SPIKE_BIN=/tmp`（是目录，非文件） | CLI 报错；env 场景告警 + 继续 |
| 3 | `SCRATCHV_SPIKE_BIN=/path/not-executable`（无 `x` 位） | CLI 报错；env 场景告警 + 继续 |
| 4 | `SCRATCHV_SPIKE_BIN` 指向不存在的旧路径，且机器上也没有 spike | 告警 + 最终 skip（告警里保留非法值） |
| 5 | `--spike-bin ""` | 视为未设置（等价于不传），不报错 |
| 6 | `SCRATCHV_SPIKE_HOME=/nonexistent` | 非致命：该层跳过，继续 `PATH`/常见目录 |

---

## 三、测试设计

所有用例都**不要求机器上安装 Spike**，通过 monkeypatch 环境变量、`shutil.which` 与模块常量构造隔离环境。测试文件：`tests/test_spike_sim_paths.py`。

### 测试用例 1：无 Spike 环境下默认跳过

- **输入**：

  ```python
  monkeypatch.delenv("SCRATCHV_SPIKE_BIN", raising=False)   # 及 DASM / LOG_PARSER / HOME
  monkeypatch.setattr(spike_sim, "COMMON_SPIKE_DIRS", ())
  monkeypatch.setattr(spike_sim, "SPIKE", str(tmp_path / "legacy-spike"))  # 不存在
  monkeypatch.setattr(shutil, "which", lambda name: None)
  rc = spike_sim.main(["--binary", str(binary), "--code-size", "64"])
  ```

- **预期输出**：`rc == 0`；stderr 含 `SKIP:`、`spike binary not found`、`SCRATCHV_SPIKE_BIN`；不生成 ELF 文件；无异常。
- **验证点**：合法缺失 ≠ 失败；退出码与 `status` 字段一致；提示包含全部搜索层名称。

### 测试用例 2：`--require-spike` 硬失败

- **输入**：同用例 1，附带 `--require-spike`。
- **预期输出**：`rc == 2`；stderr 含 `ERROR:` 与修复提示；stdout 无报告。
- **验证点**：CI 严格模式可把「环境不具备」升级为可感知失败，且与运行期失败（退出码 1）区分。

### 测试用例 3：mock 路径解析优先级

- **输入**：

  ```python
  fake_env = make_fake_tool(tmp_path, "spike-env")     # 0755 可执行
  fake_cli = make_fake_tool(tmp_path, "spike-cli")
  monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(fake_env))
  tools_env = spike_sim.resolve_spike_tools()
  tools_cli = spike_sim.resolve_spike_tools(cli_spike=str(fake_cli))
  ```

- **预期输出**：`tools_env.spike == str(fake_env)` 且 `tools_env.sources["spike"] == "env"`；`tools_cli.spike == str(fake_cli)` 且 `sources["spike"] == "cli"`。补充断言：`which` 返回假路径时 `source == "path"`；`SCRATCHV_SPIKE_HOME` 命中时 `source == "spike_home"`；常见目录命中时 `source == "common"`。
- **验证点**：完整优先级链（含 legacy 回退）逐层可控，来源标签准确。

### 测试用例 4：非法显式路径的差异化处理

- **输入**：
  - `spike_sim.resolve_spike_tools(cli_spike=str(tmp_path / "nope"))`；
  - `monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(tmp_path / "nope"))` 后调用 `resolve_spike_tools()`。
- **预期输出**：前者抛 `SpikeConfigError`，消息含 `--spike-bin` 与非法值；后者不抛异常，`tools.spike is None`，`tools.warnings` 中含 `SCRATCHV_SPIKE_BIN`；用 `--spike-bin /nonexistent` 调 `main()` 时 `rc == 2`。
- **验证点**：显式单次意图 vs 可能残留的环境变量，失败强度不同。

### 测试用例 5：报告字段与解析容错

- **输入**：
  - `SpikeResult(status="skipped", skip_reason="spike binary not found")` 分别调用 `generate_spike_report()` 与 `build_json_report()`；
  - 固定样本 stderr（含 `Commited 1234 instructions`、`1.5 MIPS`、`I$:`/`D$:` 段、带千分位 `hits: 10,000`）调用 `parse_commit_stats()` / `parse_cache_stats()`；
  - 固定样本 stdout（`PC histogram` 段 + 空行）调用 `parse_pc_histogram()`。
- **预期输出**：文本报告含 `Status:`、`skipped`、`Skip reason:`；JSON 含 `status == "skipped"`、`skip_reason`、`spike_binary is None`，且既有键（`committed_insns`、`icache`、`dcache` 等）不缺失；解析结果 `(1234, 1.5)`、`icache_hits == 10000`、PC dict 非空；缺段样本返回零值并计入 `parse_warnings`。
- **验证点**：报告字段可被下游解析；解析器对真实 Spike 输出格式与常见变体都健壮。

---

## 四、修改模块与实现步骤

### 4.1 涉及文件

| 文件 | 角色 | 改动类型 |
|------|------|---------|
| `scratchv/standalone/spike_sim.py` | 工具解析、调用、解析、报告 | 修改（核心） |
| `scratchv/standalone/run_spike_bench.py` | 无 Spike 时的纯 Python 降级后端 | 小幅适配（后端标注 + `--probe-spike`） |
| `tests/test_spike_sim_paths.py` | 新增单元测试 | 新增 |
| `docs/topics/24-Spike仿真.md` | 课题文档（路径坑与使用方式） | 更新 |
| `docs/ARCHITECTURE.md` | 架构文档（工具解析注记） | 更新一句 |
| `scratchv/standalone/tinyfive_compare.py` | 降级模式参考 | **不改** |

（实际路径以仓库为准；本设计中的行号基于 2026-09-14 版本。）

### 4.2 路径解析器（`spike_sim.py` 新增）

1. 保留 `spike_sim.py:37-39` 三个常量，但改写注释为「legacy 回退，可能不存在」，作为优先级第 6 层的候选。
2. 新增常量：`COMMON_SPIKE_DIRS`（有序元组）、`LEGACY_SPIKE_DIR`、四个环境变量名常量、`EXIT_OK/EXIT_RUN_FAIL/EXIT_CONFIG`。
3. 新增 `is_executable(path: str) -> bool`：`expanduser` + `isfile` + `access(X_OK)`。
4. 新增 `SpikeConfigError(ValueError)`。
5. 新增 `SpikeTools` 数据类：三工具路径 + `sources` + `candidates` + `warnings`，并提供 `missing` 属性与 `as_dict()`。
6. 新增 `resolve_spike_tools(...)`：按 2.1 优先级逐层解析；CLI 无效即抛 `SpikeConfigError`，env 无效记告警后继续；返回 `SpikeTools`。
7. 解析器不在 import 时执行（模块导入零副作用），在 `main()` 与 `run_spike()` 调用时解析。

### 4.3 运行层与解析层改造

1. `SpikeResult` 扩展字段：`status`（`ok|skipped|timeout|failed`）、`skip_reason`、`spike_path`、`tool_warnings`、`parse_warnings`；既有字段与拼写保持不变。
2. 抽取三个纯函数：`parse_commit_stats(stderr)`、`parse_cache_stats(stderr)`、`parse_pc_histogram(stdout)`；`run_spike()` 改为调用它们（移除 `run_spike:348-423` 的内联解析）。
3. `run_spike()` / `run_spike_with_log()` 增加仅关键字参数 `tools: SpikeTools | None = None`；默认在调用时执行 `resolve_spike_tools()`。
4. `run_spike()` 缺 spike：不启动子进程，直接返回 `status="skipped"`、`exit_code=-2` 的结果；`FileNotFoundError` 与超时分别映射 `failed` / `timeout`（保留 `-2` / `-1` 哨兵值兼容旧判读）。
5. `generate_spike_report()`：新增 `Status:`、`Skip reason:`（仅 skip 时）、`Tool warnings:`（有则打印）；`Spike:` 行改用 `result.spike_path`；`spike_sim.py:512` 不再直接引用常量。

### 4.4 CLI 与主导出流程

1. `main()`（`spike_sim.py:592-653`）新增参数：`--spike-bin`、`--spike-dasm`、`--spike-log-parser`、`--require-spike`；既有参数全部不动。
2. 流程顺序：校验 binary → 解析工具（捕获 `SpikeConfigError` → 退出码 2）→ 缺失且非严格 → `SKIP:` + JSON（若 `--json`）→ 退出码 0 → 缺失且严格 → `ERROR:` → 退出码 2 → 否则构建 ELF、运行、报告。
3. 抽取 `build_json_report(result, binary_path, code_size, ic_config, dc_config, max_instr, tools=None) -> dict`（替代 `spike_sim.py:716-745` 内联字典），新增 `status`、`skip_reason`、`spike_binary`、`parse_warnings`、`tool_warnings`、`spike_tools` 字段；既有键全部保留。
4. 成功路径打印一条工具来源摘要到 stderr（例如 `Spike tools: spike=/opt/... (cli), spike-dasm=missing`）。

### 4.5 `run_spike_bench.py` 适配

1. 新增 `--probe-spike`（默认关）：惰性导入 `resolve_spike_tools`，把解析结果打印到 stderr；`--json` 时在报告中加入 `spike_tools` 字段。
2. `generate_json_report(result, spike_tools=None)` 增加常量字段 `"backend": {"kind": "emulator", "spike_style": true}`，用于区分「纯 Python 模拟」与「真实 Spike 实测」。
3. `run_emulator_with_caches()`、`label_counts`、`label_addrs`、`cycle_estimates`、`cat_counts` **零改动**（统计口径不变），`ci_benchmark.py` 的调用不受影响。
4. 该文件本身的 label 表与周期估算失真问题记录为关联问题（见 5.3），本课题不交付。

### 4.6 测试实现

新增 `tests/test_spike_sim_paths.py`：覆盖第三节 5 个用例 + `import`/`--help` 子进程测试（干净 `PATH` 下退出码 0）。全部使用 monkeypatch / 临时可执行文件，不依赖真实 Spike。

### 4.7 文档更新

- `docs/topics/24-Spike仿真.md`：「常见坑」中「Spike 二进制路径」改为解析优先级 + 环境变量；补一段 CLI/env 用法与 skip 行为。
- `docs/ARCHITECTURE.md`：在 standalone 工具清单附近加一句工具解析与降级说明。
- `spike_sim.py` 模块 docstring（`spike_sim.py:15-22`）更新为新用法。

### 4.8 集成与回归测试

- `python -m pytest tests/test_spike_sim_paths.py -q` 全绿。
- `make test`（全量单测）与 `python .claude/harness/verify/run.py --level L2` 通过。
- `python scratchv/standalone/spike_sim.py --help`、`python -c "import scratchv.standalone.spike_sim"` 在无 Spike 机器上退出码 0。
- 在有 Spike 的机器上，用 `--spike-bin` 跑基线 binary，committed insns 与 cache 命中数与改动前一致。
- `git diff` 确认 `tinyfive_compare.py` 与周期/统计代码零改动。

---

## 五、附录

### 5.1 CLI 示例与输出

**示例 1：显式指定（推荐 CI 固定版本）**

```bash
python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140 \
  --spike-bin /opt/riscv/bin/spike --json
```

**示例 2：环境变量（整机安装）**

```bash
export SCRATCHV_SPIKE_HOME=/opt/coralnpu-spike-rv32
python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140
```

**示例 3：工具缺失（默认降级，退出码 0）**

```text
$ python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140
SKIP: spike binary not found.
  searched: --spike-bin, SCRATCHV_SPIKE_BIN, SCRATCHV_SPIKE_HOME/bin/spike, PATH,
            /opt/riscv/bin, /opt/riscv64/bin, /usr/local/bin, /usr/bin,
            ~/riscv/bin, ~/.local/bin, ~/spike/bin, legacy
  hint: export SCRATCHV_SPIKE_BIN=/path/to/spike
$ echo $?
0
```

**示例 4：`--require-spike`（严格模式，退出码 2）**

```text
$ python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140 --require-spike
ERROR: spike binary not found (--require-spike).
  searched: --spike-bin, SCRATCHV_SPIKE_BIN, ... , legacy
$ echo $?
2
```

**示例 5：skip 时的 JSON 报告（节选）**

```json
{
  "status": "skipped",
  "skip_reason": "spike binary not found",
  "spike_binary": null,
  "exit_code": -2,
  "spike_tools": {
    "spike": {"path": null, "source": "missing"},
    "spike_dasm": {"path": null, "source": "missing"},
    "spike_log_parser": {"path": null, "source": "missing"},
    "warnings": []
  },
  "binary": "output.bin",
  "code_size": 3140,
  "max_instr": 50000000,
  "committed_insns": 0
}
```

### 5.2 搜索目录与优先级速查

| 层 | 候选 |
|----|------|
| CLI | `--spike-bin` / `--spike-dasm` / `--spike-log-parser` |
| env | `SCRATCHV_SPIKE_BIN` / `SCRATCHV_SPIKE_DASM` / `SCRATCHV_SPIKE_LOG_PARSER` |
| home | `$SCRATCHV_SPIKE_HOME/bin/<tool>` |
| PATH | `shutil.which(<tool>)` |
| common | `/opt/riscv/bin`, `/opt/riscv64/bin`, `/usr/local/bin`, `/usr/bin`, `~/riscv/bin`, `~/.local/bin`, `~/spike/bin` |
| legacy | 旧硬编码常量（可执行才生效） |

### 5.3 关联问题（本课题不交付）

| 问题 | 位置 | 现象 | 建议归属 |
|------|------|------|---------|
| label 表失真 | `run_spike_bench.py:633-660`、`765-774` | `layer_descs` 硬编码旧标签前缀；`label_addrs` 实际只有 `_start@0`，逐层统计基本为空 | 独立课题（标签映射/统计口径） |
| 周期估算失真 | `run_spike_bench.py:480-512` | `mul_ratio=0.15` 猜测值、`PROFILES` 常量 CPI 与真实流水线不符 | 独立课题（微架构模型） |
| Spike 真实数据与模拟数据混用风险 | `run_spike_bench.py` docstring | 文件名为 spike_bench 但结果来自内置模拟器 | 本课题已加 `backend` 字段缓解，模型修正不在范围 |
| TinyFive 路径 | `tinyfive_compare.py` | 不涉及 | 明确不改 |

### 5.4 参考资料

- 课题文档：`docs/topics/24-Spike仿真.md`
- 降级模式先例：`scratchv/standalone/tinyfive_compare.py:29-86`（`TINYFIVE_AVAILABLE` / `_fallback`）
- Spike 官方仓库：<https://github.com/riscv-software-src/riscv-isa-sim>
- ScratchV 代码仓初始化文档：`/root/Lab/GaoMD/ScratchV/ArcDes/init.md`
