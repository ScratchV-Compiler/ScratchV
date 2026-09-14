# 课题 24 Spike 仿真：路径可移植化与降级 开发文档

> 文档版本：v1.0  
> 编写日期：2026-09-14  
> 目标读者：实现者 / 审查者  
> 配套设计文档：`./设计文档.md`  
> 范围边界：只做路径可移植与降级；**不改**周期模型与统计口径；**不改** TinyFive 路径；**不引入**外部依赖（只用 Python 标准库）

---

## 一、接口契约

本节所有名称即实现时必须使用的精确名称，不得改名。

### 1.1 新增/变更函数

| 名称 | 签名 | 语义 |
|------|------|------|
| `is_executable` | `is_executable(path: str) -> bool` | `expanduser` 后同时满足 `isfile` 与 `X_OK` |
| `resolve_spike_tools` | `resolve_spike_tools(cli_spike=None, cli_dasm=None, cli_log_parser=None, env=None, which=None, common_dirs=None) -> SpikeTools` | 按优先级解析三工具；CLI 路径无效抛 `SpikeConfigError`；env 无效记 `warnings` 后继续；`env` 默认 `os.environ`、`which` 默认 `shutil.which`（均在调用时取值，便于 monkeypatch） |
| `SpikeTools.as_dict` | `SpikeTools.as_dict() -> dict` | 输出 `{"spike": {"path": str|None, "source": str}, "spike_dasm": {...}, "spike_log_parser": {...}, "warnings": [...]}` |
| `parse_commit_stats` | `parse_commit_stats(stderr: str) -> tuple[int, float]` | 返回 `(committed_insns, mips)`；容忍 `Commited/Committed` 与千分位 |
| `parse_cache_stats` | `parse_cache_stats(stderr: str) -> dict[str, int | float]` | 返回键固定为 `icache_hits, icache_misses, icache_miss_rate, dcache_hits, dcache_misses, dcache_miss_rate` |
| `parse_pc_histogram` | `parse_pc_histogram(stdout: str) -> dict[int, int]` | 解析 `PC histogram` 段（`0x...: count`，空行结束） |
| `run_spike` | `run_spike(elf_path, max_instr=..., ic_config=..., dc_config=..., track_pc=..., log_commits=..., timeout_s=..., isa=..., mem_mb=..., *, tools: SpikeTools | None = None) -> SpikeResult` | 新增仅关键字参数 `tools`；其余参数名/默认值不变 |
| `run_spike_with_log` | `run_spike_with_log(elf_path, max_instr=..., log_path=..., isa=..., mem_mb=..., timeout_s=..., *, tools: SpikeTools | None = None) -> tuple[SpikeResult, str]` | 同上；缺 spike 时返回 `(skipped_result, "")` |
| `generate_spike_report` | 签名不变，读取 `result.status/skip_reason/spike_path/tool_warnings` | 文本报告新增 `Status:` 等行 |
| `build_json_report` | `build_json_report(result: SpikeResult, binary_path: str, code_size: int, ic_config: str, dc_config: str, max_instr: int, tools: SpikeTools | None = None) -> dict` | 替代 `main()` 内联字典；既有键保留，新增契约字段见 1.5 |
| `run_spike_bench.generate_json_report` | `generate_json_report(result, spike_tools: SpikeTools | None = None) -> dict` | 新增可选参数；新增 `"backend"` 字段 |

### 1.2 新增类型与异常

```python
@dataclass(frozen=True)
class SpikeTools:
    spike: str | None = None
    spike_dasm: str | None = None
    spike_log_parser: str | None = None
    sources: dict[str, str] = field(default_factory=dict)        # tool -> cli|env|spike_home|path|common|legacy|missing
    candidates: dict[str, tuple[str, ...]] = field(default_factory=dict)  # tool -> 已尝试的候选路径
    warnings: tuple[str, ...] = ()

    @property
    def missing(self) -> list[str]: ...   # 返回缺失工具的规范名列表

    def as_dict(self) -> dict: ...

class SpikeConfigError(ValueError):
    """显式配置（CLI）路径无效时抛出。"""
```

### 1.3 环境变量契约

| 变量 | 精确名称 | 空值语义 | 无效语义 |
|------|---------|---------|---------|
| spike | `SCRATCHV_SPIKE_BIN` | 视为未设置 | WARNING + 继续 |
| dasm | `SCRATCHV_SPIKE_DASM` | 视为未设置 | WARNING + 继续 |
| log parser | `SCRATCHV_SPIKE_LOG_PARSER` | 视为未设置 | WARNING + 继续 |
| 安装根目录 | `SCRATCHV_SPIKE_HOME` | 视为未设置 | 静默跳过该层（候选列表留痕） |

规则：值先 `strip()`，支持 `~` 展开；大小写敏感；三工具互不影响。

### 1.4 CLI 契约（`spike_sim.py`）

| 参数 | 类型/默认 | 说明 |
|------|----------|------|
| `--spike-bin PATH` | str / 无 | spike 可执行文件（必需工具）；无效 → 退出码 2 |
| `--spike-dasm PATH` | str / 无 | spike-dasm（可选工具）；无效 → WARNING + 继续解析，不阻断 |
| `--spike-log-parser PATH` | str / 无 | spike-log-parser（可选工具）；无效 → WARNING + 继续解析，不阻断 |
| `--require-spike` | store_true / 关 | spike 缺失时硬失败（退出码 2） |
| `--probe-spike` | store_true / 关 | **仅 `run_spike_bench.py`**：探测并打印工具可用性 |

既有参数 `--binary`、`--code-size`、`--max-instr`、`--ic`、`--dc`、`--isa`、`--mem`、`--timeout`、`--no-pc-histogram`、`--log-commits`、`--log-instr-limit`、`--keep-elf`、`--elf-output`、`--json` 全部保持不变。

### 1.5 退出码与数据字段契约

```
EXIT_OK = 0        # 成功或 skip
EXIT_RUN_FAIL = 1  # 运行失败（超时/非零退出/binary 缺失，历史兼容）
EXIT_CONFIG = 2    # 配置错误（CLI 路径无效 / --require-spike 且缺 spike）
```

`SpikeResult` 新增字段（既有字段与拼写 `commited_insns_per_sec` 一律不改）：

```python
status: str = "ok"             # ok | skipped | timeout | failed
skip_reason: str = ""
spike_path: str = ""
tool_warnings: list[str] = field(default_factory=list)
parse_warnings: list[str] = field(default_factory=list)
```

`build_json_report()` 新增键：`status`、`skip_reason`、`spike_binary`、`parse_warnings`、`tool_warnings`、`spike_tools`（`tools` 为空时不输出该键）。既有键：`binary`、`code_size`、`static_insns`、`max_instr`、`committed_insns`、`wall_time_s`、`exit_code`、`icache`、`dcache`、`top_pcs`、`stderr_tail` 全部保留。

`exit_code` 哨兵兼容：`-1` 超时、`-2` 工具不可用/启动失败；skip 路径的 JSON 报告同样填 `-2`；`status` 为权威判读字段。此外，spike 退出 0 但输出中完全没有任何可解析统计段时，防呆为 `status=failed`、`exit_code=-2`（疑似非 Spike 可执行文件）。

---

## 二、`spike_sim.py` 逐处改动

行号基于 2026-09-14 版本；改动按文件从上到下排列。

### 2.1 模块 docstring（L15-22）

把硬编码路径示例改为新用法：

```text
Spike tool resolution order:
  1. --spike-bin / --spike-dasm / --spike-log-parser
  2. SCRATCHV_SPIKE_BIN / SCRATCHV_SPIKE_DASM / SCRATCHV_SPIKE_LOG_PARSER
  3. $SCRATCHV_SPIKE_HOME/bin/<tool>
  4. PATH, then common install dirs, then the legacy constants below
If spike is missing, the tool exits 0 with "SKIP: ..." unless --require-spike.
```

### 2.2 导入区（L24-34）

新增 `import shutil`（`os/subprocess/sys` 已有）。不引入第三方包。

### 2.3 路径常量与 legacy 回退（L36-39）

保留三个常量名，改写为「legacy 回退」并新增解析所需常量：

```python
# ── Paths ──────────────────────────────────────────────────────────────────
# Legacy fallback (may not exist on this machine). New code must use
# resolve_spike_tools(); these constants are only the last resolution layer.
SPIKE = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike"
SPIKE_DASM = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike-dasm"
SPIKE_LOG_PARSER = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike-log-parser"

ENV_SPIKE_BIN = "SCRATCHV_SPIKE_BIN"
ENV_SPIKE_DASM = "SCRATCHV_SPIKE_DASM"
ENV_SPIKE_LOG_PARSER = "SCRATCHV_SPIKE_LOG_PARSER"
ENV_SPIKE_HOME = "SCRATCHV_SPIKE_HOME"

COMMON_SPIKE_DIRS: tuple[str, ...] = (
    "/opt/riscv/bin",
    "/opt/riscv64/bin",
    "/usr/local/bin",
    "/usr/bin",
    "~/riscv/bin",
    "~/.local/bin",
    "~/spike/bin",
)

EXIT_OK = 0
EXIT_RUN_FAIL = 1
EXIT_CONFIG = 2
```

**要点**：legacy 常量必须由 `resolve_spike_tools()` 在**调用时**读取（不要 import 时固化进元组），否则测试无法 monkeypatch。

### 2.4 新增解析器（建议插在常量区之后）

```python
def is_executable(path: str) -> bool:
    path = os.path.expanduser(path)
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


class SpikeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SpikeTools:
    spike: str | None = None
    spike_dasm: str | None = None
    spike_log_parser: str | None = None
    sources: dict[str, str] = field(default_factory=dict)
    candidates: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def missing(self) -> list[str]:
        pairs = (("spike", self.spike), ("spike-dasm", self.spike_dasm),
                 ("spike-log-parser", self.spike_log_parser))
        return [name for name, path in pairs if not path]

    def as_dict(self) -> dict:
        def one(name: str, path: str | None) -> dict:
            return {"path": path, "source": self.sources.get(name, "missing"),
                    "candidates": list(self.candidates.get(name, ()))}
        return {"spike": one("spike", self.spike),
                "spike_dasm": one("spike-dasm", self.spike_dasm),
                "spike_log_parser": one("spike-log-parser", self.spike_log_parser),
                "warnings": list(self.warnings)}


def _resolve_one(tool, cli_flag, cli_value, env_name, legacy_const,
                 env, which, common_dirs, cli_required=True):
    """返回 (path|None, source, candidates, warnings)。

    cli_required=True（spike）时 CLI 无效抛错；可选工具（dasm/log-parser）
    传 False，CLI 无效仅告警并继续后续层级。
    """
    candidates: list[str] = []
    warnings: list[str] = []

    if cli_value:                                          # 1. CLI（显式）
        cand = os.path.expanduser(cli_value.strip())
        candidates.append(cand)
        if not is_executable(cand):
            if cli_required:
                raise SpikeConfigError(
                    f"{cli_flag}={cli_value!r} is not an executable file")
            warnings.append(
                f"{cli_flag}={cli_value!r} is not executable; ignored")
        else:
            return cand, "cli", candidates, warnings

    env_value = (env.get(env_name) or "").strip()          # 2. env（显式，告警继续）
    if env_value:
        cand = os.path.expanduser(env_value)
        candidates.append(cand)
        if is_executable(cand):
            return cand, "env", candidates, warnings
        warnings.append(f"{env_name}={env_value} is not executable; ignored")

    home = (env.get(ENV_SPIKE_HOME) or "").strip()         # 3. home 提示
    if home:
        cand = os.path.join(os.path.expanduser(home), "bin", tool)
        candidates.append(cand)
        if is_executable(cand):
            return cand, "spike_home", candidates, warnings

    found = which(tool) if which else None                 # 4. PATH
    if found:
        candidates.append(found)
        return found, "path", candidates, warnings

    for d in common_dirs:                                  # 5. 常见目录
        cand = os.path.join(os.path.expanduser(d), tool)
        candidates.append(cand)
        if is_executable(cand):
            return cand, "common", candidates, warnings

    if legacy_const and is_executable(legacy_const):       # 6. legacy 常量
        candidates.append(legacy_const)
        return legacy_const, "legacy", candidates, warnings

    return None, "missing", candidates, warnings


def resolve_spike_tools(cli_spike=None, cli_dasm=None, cli_log_parser=None,
                        env=None, which=None, common_dirs=None) -> SpikeTools:
    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    common_dirs = COMMON_SPIKE_DIRS if common_dirs is None else common_dirs

    spec = (
        ("spike", "--spike-bin", cli_spike, ENV_SPIKE_BIN, SPIKE, True),
        ("spike-dasm", "--spike-dasm", cli_dasm, ENV_SPIKE_DASM, SPIKE_DASM,
         False),
        ("spike-log-parser", "--spike-log-parser", cli_log_parser,
         ENV_SPIKE_LOG_PARSER, SPIKE_LOG_PARSER, False),
    )
    paths, sources, candidates, warnings = {}, {}, {}, []
    for tool, flag, cli_value, env_name, legacy_const, cli_required in spec:
        path, source, cands, warns = _resolve_one(
            tool, flag, cli_value, env_name, legacy_const,
            env, which, common_dirs, cli_required=cli_required)
        paths[tool], sources[tool], candidates[tool] = path, source, cands
        warnings.extend(warns)
    return SpikeTools(
        spike=paths["spike"], spike_dasm=paths["spike-dasm"],
        spike_log_parser=paths["spike-log-parser"],
        sources=sources, candidates={k: tuple(v) for k, v in candidates.items()},
        warnings=tuple(warnings))
```

### 2.5 `SpikeResult` 扩展（L250-275）

在类中追加 1.5 节五个字段（默认值如上）。不动既有字段顺序与拼写。

### 2.6 解析函数抽取（替换 L348-423 内联逻辑）

```python
def parse_commit_stats(stderr: str) -> tuple[int, float]: ...
def parse_cache_stats(stderr: str) -> dict[str, int | float]: ...
def parse_pc_histogram(stdout: str) -> dict[int, int]: ...
```

实现约定：

- 数字统一 `s.replace(",", "")` 后再 `int()`/`float()`；失败记入告警（函数内可选返回 warnings，若要保持返回类型纯净，则由 `run_spike` 对比「段头存在但值为 0」自行告警；推荐方案：内部 `re` 匹配段头标记，缺失段头时 `run_spike` append 一条 warning）。
- `parse_commit_stats` 正则：`r"(?:Commited|Committed)\s+([\d,]+)\s+instructions"`；MIPS：`r"([\d.]+)\s*MIPS"`。
- `parse_cache_stats`：先定位 `^I\$:` / `^D\$:` 段头，再取 `hits:\s*([\d,]+)`、`misses:\s*([\d,]+)`、`miss rate:\s*([\d.]+)%`。
- `parse_pc_histogram`：段头行含 `PC histogram`；数据行 `r"^(0x[0-9a-fA-F]+):\s*(\d+)$"`；空行结束。

### 2.7 `run_spike()`（L278-425）

1. 签名尾部加 `*, tools: SpikeTools | None = None`。
2. 函数体开头：

   ```python
   tools = tools or resolve_spike_tools()
   if tools.spike is None:
       return SpikeResult(status="skipped",
                          skip_reason="spike binary not found",
                          exit_code=-2,
                          stderr="SKIP: spike binary not found; "
                                 "set SCRATCHV_SPIKE_BIN or pass --spike-bin",
                          tool_warnings=list(tools.warnings))
   ```

3. `cmd` 首元素由 `SPIKE` 改为 `tools.spike`；`result.spike_path = tools.spike`；`result.tool_warnings` 合并 `tools.warnings`。
4. `except FileNotFoundError` / `except OSError`：均 `status="failed"`、`exit_code=-2`，消息带 `tools.spike`（`OSError` 覆盖「文件存在且可执行但内核拒绝 exec」的 ENOEXEC 等场景）。
5. `except subprocess.TimeoutExpired`：`status="timeout"`。
6. 正常返回前用 2.6 的解析函数填充字段；`result.exit_code` 保留 `proc.returncode`（唯一例外：退出 0 但完全无可解析统计段时置 `-2`，见设计文档 §2.3 第 8b 行）。

### 2.8 `run_spike_with_log()`（L432-488）

签名加 `*, tools=None`；开头做与 2.7 相同的缺失判断，返回 `(skipped_result, "")`；`cmd` 首元素改 `tools.spike`；异常映射与 2.7 相同（含 `OSError` → `failed`/`-2`）。

### 2.9 `generate_spike_report()`（L495-585）

- L512 `Spike: {SPIKE}` → `Spike: {result.spike_path or "(not resolved)"}`；若 `status=="skipped"`，在 Timing 段前输出：

  ```text
  ── Run Status ──
  Status:        skipped
  Skip reason:   spike binary not found
  ```

- 有 `tool_warnings` / `parse_warnings` 时各打印一节（`Tool warnings`、`Parse warnings`），每条一行、缩进 `  | `。

### 2.10 `main()` 与 JSON（L592-763）

1. argparse（L592-653）追加 1.4 节的四个参数（`--probe-spike` 不加在 `spike_sim.py`）。
2. binary 校验之后、构建 ELF 之前插入工具解析与分支：

   ```python
   try:
       tools = resolve_spike_tools(args.spike_bin, args.spike_dasm,
                                   args.spike_log_parser)
   except SpikeConfigError as e:
       print(f"ERROR: {e}", file=sys.stderr)
       return EXIT_CONFIG

   if tools.spike is None:
       searched = ", ".join(["--spike-bin", ENV_SPIKE_BIN,
                             f"{ENV_SPIKE_HOME}/bin/spike", "PATH",
                             *COMMON_SPIKE_DIRS, "legacy"])
       if args.require_spike:
           print(f"ERROR: spike binary not found (--require-spike).\n"
                 f"  searched: {searched}", file=sys.stderr)
           return EXIT_CONFIG
       print(f"SKIP: spike binary not found.\n"
             f"  searched: {searched}\n"
             f"  hint: export {ENV_SPIKE_BIN}=/path/to/spike", file=sys.stderr)
       if args.json:
           print(json.dumps(build_json_report(
               SpikeResult(status="skipped",
                           skip_reason="spike binary not found",
                           tool_warnings=list(tools.warnings)),
               args.binary, args.code_size, args.ic, args.dc,
               args.max_instr, tools), indent=2))
       return EXIT_OK
   ```

3. 成功路径：向 `run_spike` / `run_spike_with_log` 传 `tools=tools`。
4. 抽取 `build_json_report(...)`（L716-745 迁移）：在原有键基础上加入 1.5 节新键；`spike_tools` 用 `tools.as_dict()`。
5. 最终返回：

   ```python
   return EXIT_OK if result.status == "ok" else EXIT_RUN_FAIL
   ```

6. 运行日志增加一行工具来源摘要（stderr）：

   ```text
   Spike tools: spike=/opt/riscv/bin/spike (cli), spike-dasm=missing, spike-log-parser=missing
   ```

---

## 三、`run_spike_bench.py` 适配点

**定位**：该文件是「无 Spike 环境」的纯 Python 降级后端，不调用外部 Spike 二进制，因此本课题**不改其任何统计逻辑**。

### 3.1 改动清单

| 位置 | 改动 | 约束 |
|------|------|------|
| `main()` argparse（L732-757） | 新增 `--probe-spike`（`action="store_true"`，默认关） | 其余参数不动 |
| `main()` 输出前 | 若 `--probe-spike`：惰性 `from scratchv.standalone.spike_sim import resolve_spike_tools, SpikeConfigError`，打印 `Spike tools: ...` 或 `Spike tools: not found (...)`；捕获 `SpikeConfigError` 打印 `ERROR:` 但**不**改变退出码 | 默认路径（未传 flag）行为与输出完全不变 |
| `generate_json_report`（L669-725） | 签名 `generate_json_report(result, spike_tools=None)`；顶层新增 `"backend": {"kind": "emulator", "spike_style": true}`；`spike_tools` 非空时新增 `"spike_tools": spike_tools.as_dict()` | 既有键一个不删；不改任何数值 |
| 模块 docstring（L1-30） | 增加一句：本工具输出为模拟器结果，真实 Spike 实测请用 `spike_sim.py`（本课题新增 `backend` 字段标注） | 不改用法示例 |

### 3.2 明确不改的部分（防止越界）

- `run_emulator_with_caches()` 主体（L107-514）：包括 `pc_samples` 采样、`cat_counts`、`branch_*` 计数、cache 访问。
- 周期估算块（L480-512）：`mul_ratio=0.15`、`PROFILES` 循环——关联问题，不交付。
- label 表（L633-660）与 `label_addrs` 构建（L763-774）——关联问题，不交付。
- `ci_benchmark.py` 调用点（`scratchv/ci/ci_benchmark.py:354-389`）：只调 `run_emulator_with_caches`，本适配不影响其行为。

### 3.3 输出示例（`--probe-spike`）

```text
$ python scratchv/standalone/run_spike_bench.py --binary output.bin --code-size 3140 \
    --max-instr 1000000 --probe-spike
Spike tools: spike=NOT FOUND, spike-dasm=NOT FOUND, spike-log-parser=NOT FOUND
  hint: real Spike is unavailable; this run uses the built-in emulator backend
Spike-Style RISC-V Simulation
...
```

---

## 四、文档更新点

| 文件 | 更新内容 |
|------|---------|
| `docs/topics/24-Spike仿真.md` | 1) 「常见坑」表「Spike 二进制路径」改为：路径按 `--spike-bin` > `SCRATCHV_SPIKE_BIN` > `SCRATCHV_SPIKE_HOME` > `PATH` > 常见目录 > legacy 解析，不再硬编码；2) 新增「无 Spike 机器上的行为」段：默认 `SKIP:` 退出码 0，`--require-spike` 退出码 2；3) 动手练习 1 前补 CLI 示例 |
| `docs/ARCHITECTURE.md` | standalone 工具清单（L378-379 附近）加一句：Spike 三工具经统一 resolver 解析，缺失时按 skip/告警分层降级（详见课题 24 设计文档） |
| `scratchv/standalone/spike_sim.py` docstring | 见 2.1 |
| `scratchv/standalone/run_spike_bench.py` docstring | 见 3.1 |
| `CLAUDE.md`（可选） | 关键命令区补 `export SCRATCHV_SPIKE_BIN=...` 示例；非必需，视团队习惯 |

不新增独立 md 文档到仓库；课题文档按既有目录结构更新。

---

## 五、测试文件与用例

**新增文件**：`tests/test_spike_sim_paths.py`。运行：`python -m pytest tests/test_spike_sim_paths.py -q`。所有用例不依赖真实 Spike。

### 5.1 公共夹具

```python
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scratchv.standalone import spike_sim

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_fake_tool(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return p


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    for var in ("SCRATCHV_SPIKE_BIN", "SCRATCHV_SPIKE_DASM",
                "SCRATCHV_SPIKE_LOG_PARSER", "SCRATCHV_SPIKE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spike_sim, "COMMON_SPIKE_DIRS", ())
    monkeypatch.setattr(spike_sim, "SPIKE", str(tmp_path / "legacy-spike"))
    monkeypatch.setattr(spike_sim, "SPIKE_DASM", str(tmp_path / "legacy-dasm"))
    monkeypatch.setattr(spike_sim, "SPIKE_LOG_PARSER", str(tmp_path / "legacy-parser"))
    monkeypatch.setattr(shutil, "which", lambda name: None)
```

### 5.2 用例清单

| 用例 | 名称 | 断言要点 |
|------|------|---------|
| 1 | `test_import_works_without_spike` | 子进程干净 `PATH`/`HOME` 下 `import scratchv.standalone.spike_sim` 退出码 0 |
| 2 | `test_help_exits_zero` | `pytest.raises(SystemExit)`，`code == 0`，不探测工具 |
| 3 | `test_missing_spike_skips_with_reason` | `main()` 返回 0；stderr 含 `SKIP:` 与 `SCRATCHV_SPIKE_BIN` |
| 4 | `test_missing_spike_strict_returns_config_error` | `--require-spike` → 返回 2；stderr 含 `ERROR:` |
| 5 | `test_resolution_cli_over_env` | CLI 命中 `source=="cli"` |
| 6 | `test_resolution_env_over_path` | env 命中 `source=="env"`；无 env 时 `which` 假路径 → `source=="path"` |
| 7 | `test_resolution_spike_home_and_common` | home 命中 `source=="spike_home"`；common 目录命中 `source=="common"` |
| 8 | `test_resolution_legacy_constant` | 仅 legacy 常量可执行时 `source=="legacy"`（monkeypatch 常量指向 tmp 可执行文件） |
| 9 | `test_cli_invalid_path_raises` | `pytest.raises(spike_sim.SpikeConfigError)`，消息含 `--spike-bin` |
| 10 | `test_env_invalid_path_warns_and_falls_through` | 不抛异常；`tools.warnings` 含变量名；最终 `None` |
| 11 | `test_parse_commit_stats` | 样本返回 `(1234, 1.5)`；缺段返回 `(0, 0.0)` |
| 12 | `test_parse_cache_stats` | 千分位样本 → `icache_hits == 10000` 等；缺段全 0 |
| 13 | `test_parse_pc_histogram` | 样本 dict 非空；空行终止；坏行忽略 |
| 14 | `test_report_status_fields` | 文本含 `Status:` / `Skip reason:`；JSON 含 `status` / `skip_reason` / 既有键 |

### 5.3 关键用例示例代码

```python
def test_missing_spike_skips_with_reason(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64"])

    assert rc == spike_sim.EXIT_OK
    err = capsys.readouterr().err
    assert "SKIP:" in err
    assert "SCRATCHV_SPIKE_BIN" in err
    assert not (tmp_path / "output_spike.elf").exists()


def test_missing_spike_strict_returns_config_error(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--require-spike"])

    assert rc == spike_sim.EXIT_CONFIG
    assert "ERROR:" in capsys.readouterr().err


def test_resolver_priority(tmp_path, monkeypatch):
    fake_env = make_fake_tool(tmp_path, "spike-env")
    fake_cli = make_fake_tool(tmp_path, "spike-cli")
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(fake_env))

    t_env = spike_sim.resolve_spike_tools()
    assert t_env.spike == str(fake_env)
    assert t_env.sources["spike"] == "env"

    t_cli = spike_sim.resolve_spike_tools(cli_spike=str(fake_cli))
    assert t_cli.spike == str(fake_cli)
    assert t_cli.sources["spike"] == "cli"


def test_cli_invalid_path_raises(tmp_path):
    with pytest.raises(spike_sim.SpikeConfigError) as ei:
        spike_sim.resolve_spike_tools(cli_spike=str(tmp_path / "nope"))
    assert "--spike-bin" in str(ei.value)


def test_env_invalid_path_warns_and_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(tmp_path / "nope"))

    tools = spike_sim.resolve_spike_tools()

    assert tools.spike is None
    assert any("SCRATCHV_SPIKE_BIN" in w for w in tools.warnings)


CANNED_STDERR = """\
Commited 1234 instructions
core   0: 0x80000000 (0x00000013) 1.5 MIPS
I$: 64 sets × 2 ways × 32 B
  hits: 10,000    misses: 25    miss rate: 0.25%
D$: 128 sets × 4 ways × 32 B
  hits: 20,000    misses: 50    miss rate: 0.25%
"""
CANNED_STDOUT = """\
PC histogram (number of commits per PC):
0x80000014: 123
0x80000018: 456

"""


def test_parse_helpers_from_canned_output():
    committed, mips = spike_sim.parse_commit_stats(CANNED_STDERR)
    assert (committed, mips) == (1234, 1.5)

    stats = spike_sim.parse_cache_stats(CANNED_STDERR)
    assert stats["icache_hits"] == 10_000
    assert stats["icache_misses"] == 25
    assert stats["dcache_hits"] == 20_000

    hist = spike_sim.parse_pc_histogram(CANNED_STDOUT)
    assert hist == {0x80000014: 123, 0x80000018: 456}
```

`test_import_works_without_spike` 的子进程写法：

```python
def test_import_works_without_spike(tmp_path):
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path),
           "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c",
         "import scratchv.standalone.spike_sim as s; print(s.SPIKE)"],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
```

---

## 六、验收标准

在无 Spike 机器上（可先 `unset SCRATCHV_SPIKE_*`）：

| # | 命令 | 预期 |
|---|------|------|
| 1 | `python -c "import scratchv.standalone.spike_sim"` | 退出码 0，无输出/异常 |
| 2 | `python scratchv/standalone/spike_sim.py --help` | 退出码 0，列出 `--spike-bin/--spike-dasm/--spike-log-parser/--require-spike` |
| 3 | `python scratchv/standalone/spike_sim.py --binary output.bin --code-size 3140` | stderr `SKIP: spike binary not found...`，退出码 0，不生成 ELF |
| 4 | 同上加 `--require-spike` | stderr `ERROR:`，退出码 2 |
| 5 | 同上加 `--spike-bin /nonexistent` | stderr `ERROR:`，退出码 2 |
| 6 | 同上加 `--json`（有/无 spike 均可） | JSON 含 `status`、`skip_reason`、`spike_binary`，既有键完整 |
| 7 | `python scratchv/standalone/run_spike_bench.py --binary output.bin --code-size 3140 --max-instr 1000000 --probe-spike` | 退出码 0；stderr 有工具可用性摘要；数值与未加 flag 时一致 |
| 8 | `python -m pytest tests/test_spike_sim_paths.py -q` | 全绿 |
| 9 | `make test` | 全量单测通过 |
| 10 | `python .claude/harness/verify/run.py --level L2` | 通过 |

在有 Spike 的机器上：

| # | 命令 | 预期 |
|---|------|------|
| 11 | `--spike-bin <真实路径>` 跑基线 binary | `status=ok`；`committed_insns`、I$/D$ 命中数与改动前一致（数值零回归） |
| 12 | `SCRATCHV_SPIKE_BIN=<旧路径>`（不存在） | stderr `WARNING:` + 回退/跳过，不崩溃 |

---

## 七、风险与回退

| 风险 | 缓解 | 回退 |
|------|------|------|
| 解析器误选系统里的其他 Spike 版本 | 报告/日志记录 `spike_path` 与 `source`；CI 用 `--spike-bin` 固定 | 显式 `--spike-bin` 永远最高优先级 |
| 残留的 `SCRATCHV_SPIKE_BIN` 指向旧路径 | 只告警并继续后续层级，不阻断 | `unset` 或 `--spike-bin` 覆盖 |
| monkeypatch 失效（常量被 import 时固化） | 2.3/2.4 明确要求调用时读取常量 | 解析器增加 `legacy_const` 显式参数（本次已设计为参数） |
| 新增 CLI/JSON 字段破坏下游解析 | 只增不删；`ci_benchmark.py` 走函数调用不受影响 | 回退单个 commit；JSON 旧键始终保留 |
| 解析函数抽取引入行为差异 | 用固定样本用例锁定正则与容错行为 | 保留旧内联逻辑在 git 历史中可快速恢复 |
| import 变慢/有副作用 | 导入阶段不调用 `resolve`、不探测文件系统 | 解析全部延迟到 `main()` / `run_spike()` |
| legacy 常量机器（原作者）行为回归 | legacy 作为最后一层，仍可命中 | 若出问题，显式 `--spike-bin` 即可 |

---

## 八、关联不交付项（范围边界）

| 项 | 位置 | 说明 |
|----|------|------|
| label 表失真 | `run_spike_bench.py:633-660`、`765-774` | `layer_descs` 使用旧标签前缀，`label_addrs` 实际只有 `_start@0`，逐层统计不可用；建议独立课题重做标签映射 |
| 周期估算失真 | `run_spike_bench.py:480-512` | `mul_ratio=0.15` 为经验猜测；`PROFILES` CPI 常量与真实微架构未校准；建议独立课题 |
| TinyFive 路径 | `tinyfive_compare.py` | 本课题明确不触碰（含 `TINYFIVE_AVAILABLE` 探测逻辑） |
| 统计口径 | `run_spike_bench.py` 全部计数 | 不改采样间隔、分类、命中率定义 |
| commit log 深度解析（dasm/log-parser 实际调用） | `spike_sim.py` | 本课题只解析工具路径并定义缺失语义；日志内容仍原样透传，指令分类接入留待后续 |

---

## 九、参考资料

- 设计文档：`./设计文档.md`
- 课题文档：`docs/topics/24-Spike仿真.md`
- 降级模式先例：`scratchv/standalone/tinyfive_compare.py:29-86`
- 使用方调用点：`scratchv/ci/ci_benchmark.py:341-393`
- 代码风格：Python 3.12+、type hints、argparse、零外部依赖

---

## 实现结果（2026-09-14 集成）

> **集成 commit**：`2f85f32`（`feat(topic24): make Spike toolchain paths portable with graceful degradation`）
> **集成位置**：`Seven_big_summary` 上第 6 个 topic commit（顺序 … → 17 → **24** → 27 → …）
> **集成后全量**：`PYTHONPATH=. python3.11 -m pytest tests/ -q` → **1011 passed / 13 xfailed / 20 xpassed / 0 failed**

### 实现文件与要点

| 文件 | 要点 |
|------|------|
| `scratchv/standalone/spike_sim.py` | 六级路径解析（`--spike-bin` > `SCRATCHV_SPIKE_BIN` > … > legacy）+ 降级分层 |
| `scratchv/standalone/run_spike_bench.py` | 新增 `--probe-spike` |
| `tests/test_spike_sim_paths.py` | 18 个新用例 |
| `docs/ARCHITECTURE.md`、`docs/topics/24-Spike仿真.md` | 随实现更新 |

### 测试数字

| 口径 | 结果 |
|------|------|
| 定向（`tests/test_spike_sim_paths.py`） | 18 passed |
| 分支全量（cherry-pick 前） | 583 passed |
| 集成后全量 | 1011 passed / 13 xfailed / 20 xpassed / 0 failed |

### 与本文档的偏差 / 未完成项

- 新增顶层 `json` / `re` import。
- `main(argv=None)` 签名适配（便于测试注入参数）。
- dasm / log-parser 缺失仅 WARNING，不阻断。

### 已知限制

- 本机 PATH 中无真实 `spike` 二进制，§六 验收 11/12（真实 Spike 端到端与数值零回归）需在装有 Spike 的机器上补跑。
- label 表失真、周期估算失真、日志深度解析等仍属范围外（§八）。
