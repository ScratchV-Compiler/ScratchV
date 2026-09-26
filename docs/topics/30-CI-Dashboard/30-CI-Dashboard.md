# 课题30：CI 基准编排 + Dashboard

> **难度**：高 | **类型**：参考分析 | **源文件**：`scratchv/ci/ci_benchmark.py`, `scratchv/ci/dashboard.py`
> **状态**：✅ 已完成

---

## 概述

CI 基准系统自动运行 ScratchV vs LLVM 的性能对比，生成纯静态 HTML 仪表盘。Dashboard 展示静态/动态指令数对比、缓存命中率，是评估 ScratchV 编译优化进展的核心可视化工具。

---

## 理解背景

### 是什么？

CI 基准系统自动运行 ScratchV vs LLVM 的性能对比，生成**纯静态 HTML 仪表盘**，展示：

- **静态指令数对比**：按算子和指令类别拆分
- **动态指令数对比**：带百分比差异
- **缓存命中率 & 时间估算**
- **综合评分**：绿/黄/红 badge 标记差距大小

```
make bench-ci
    │
    ├─→ ci_benchmark.py      ← 编排器：运行 LLVM 对比
    │       └─→ llvm_cache_compare.py
    │
    └─→ dashboard.py         ← 可视化：JSON → HTML dashboard
            └─→ benchmark_reports/dashboard.html (纯静态, 零依赖)
```

### 为什么？

- **持续监控**：每次 push 自动运行，防止性能衰退
- **决策依据**：优化是否有效？数据说话
- **可视化**：好看的图表比一页数字更有说服力

---

## 代码走读

### 一键运行

```bash
# 完整 CI 基准（LLVM 对比 + Dashboard 生成）
make bench-ci
```

实际输出：
```
python3 scratchv/ci/ci_benchmark.py \
    --model-registry ci_models.json \
    --output-dir benchmark_reports/ \
    --html dashboard.html \
    --json-out ci_data.json \
    --md github_summary.md \
    --embed-json \
    --skip-cache

Dashboard: benchmark_reports/dashboard.html
JSON data: benchmark_reports/ci_data.json
GitHub summary: benchmark_reports/github_summary.md
```

### 只生成 Dashboard（使用已有数据）

```bash
# 先单独运行 LLVM 对比
python3 scratchv/standalone/llvm_cache_compare.py \
    --json-output /tmp/llvm_vs_scratchv.json

# 再生成 Dashboard
python3 scratchv/ci/dashboard.py --run -o /tmp/dashboard.html
```

Dashboard 是纯静态 HTML（CSS 内嵌，零外部依赖），直接用浏览器打开即可。

### 当前性能关键数据

来自 `llvm_cache_compare.py` 的实际运行结果：

| 指标 | LLVM | ScratchV | 比值 |
|------|------|----------|------|
| **静态指令数** | 1059 | 887 | 0.84× (ScratchV 更少!) |
| **动态指令数** | ~18.5 亿 | ~22.0 亿 | 1.19× |
| **LLVM IR 静态** | 1102 | — | — |
| **估算时间 @100MHz** | 24.3s | 39.1s | 1.6× |
| **CPI** | ~1.32 | ~1.34 | — |

### Dashboard 数据流

```
llvm_cache_compare.py → llvm_vs_scratchv.json
                                │
                       dashboard.py
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
           dashboard.html              ci_data.json
           (静态指令+Cache)            (结构化数据)
```

---

## 动手练习

### 练习 1: 生成自己的 Dashboard

运行 `make bench-ci`，用浏览器打开 `benchmark_reports/dashboard.html`，找出 ScratchV 和 LLVM 差距最大的维度。

### 练习 2: 添加新指标

在 `dashboard.py` 中添加一个新指标列（比如"每 MAC 指令数"），重新生成 dashboard。

### 练习 3: 自定义样式

修改 `dashboard.py` 中的 CSS 变量，改变 dashboard 的颜色主题。

---

## 常见坑

| 坑 | 说明 |
|----|------|
| **CI 超时** | `bench-ci` 运行较久（LLVM 对比 + HTML 生成），CI 要设置足够的 timeout |
| **JSON 缓存** | 如果之前的 JSON 文件存在，dashboard 可能用旧数据。使用 `--skip-cache` 强制重新运行 |
| **Python 版本** | 推荐 Python 3.12，本地开发如果用其他版本可能结果略有差异 |

---

## 进阶阅读

- [03-指标解读指南](../../guide/03-指标解读指南.md) — 如何解读 Dashboard 中的数据
- GitHub Actions 文档：[Workflow syntax](https://docs.github.com/en/actions/writing-workflows)
- 相关课题: [课题6 — 性能基准套件](../06-性能基准套件/06-性能基准套件.md) | [课题19 — Standalone RISC-V](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md) | [课题25 — LLVM对比工具](../25-LLVM对比工具/25-LLVM对比工具.md)
