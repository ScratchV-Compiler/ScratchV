"""Small human summaries; complete machine evidence remains in report.json."""
from __future__ import annotations

from html import escape
import math

from probes.numeric_summary import compact_metric_table, compact_metric_table_html, METRIC_EXPLANATION

# The medium gate uses the seven fixed input_cases from w2_qwen3_small.run.
_MEDIUM_CASE_COUNT = 7
_INDEX_LIMIT = 20


def _short(value, limit=240):
    """Bound human-facing text without truncating the machine evidence."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…（完整内容见 report.json）"


def _first_issue(row, depth=0):
    """Find one failed check, without printing its entire nested result."""
    if depth > 6:
        return ""
    if isinstance(row, list):
        return next((text for item in row if (text := _first_issue(item, depth + 1))), "")
    if not isinstance(row, dict) or row.get("passed") is True:
        return ""
    for key in ("error", "reason"):
        if row.get(key):
            return _short(row[key])
    first = row.get("first_divergence")
    if first:
        if isinstance(first, dict):
            first = " / ".join(str(first[key]) for key in ("comparison", "name", "checkpoint", "reason")
                               if first.get(key)) or "结构化定位数据见 report.json"
        return "首个差异：" + _short(first)
    # Visit execution/check result structures, never provenance or array data.
    for key in ("comparison", "comparisons", "ort", "ir", "normal", "diagnostic",
                "none", "basic", "all", "executions", "workers", "pytorch_comparison",
                "ort_comparison", "capture_preserves_logits", "pack_layout", "checkpoints"):
        child = row.get(key)
        if key == "comparisons" and isinstance(child, dict):
            child = list(child.values())
        issue = _first_issue(child, depth + 1)
        if issue:
            return issue
    return ""


def _index_status(row):
    status = row.get("status")
    if status in ("PARTIAL", "COMPLETE"):
        return status
    if row.get("passed") is True:
        return "PASS" if status in (None, "PASS", "success") else "状态冲突（见 JSON）"
    if row.get("passed") is False:
        return _short(status) if status and status != "PASS" else "FAIL"
    return _short(status) if status else "未确认"


def evidence_index_views(report):
    """A bounded index of recorded results, with the JSON as source of detail."""
    entries = []
    for key, label in (("cases", "用例"), ("gates", "子门禁"), ("invariants", "隔离检查")):
        for index, row in enumerate(report.get(key, []), 1):
            name = row.get("name", row.get("case", f"{label} {index}"))
            entries.append((f"{label}：{_short(name, 100)}", row))
    if report.get("gate") == "w3_layer_diff":
        entries.extend(("检查点：" + _short(row.get("name", "未命名"), 100), row)
                       for row in report.get("checkpoints", []))
    if not entries:
        entries.append(("流程阶段：" + _short(report.get("stage", "未记录"), 100), report))
    # A late failure must not disappear behind the first twenty successful rows.
    entries.sort(key=lambda item: item[1].get("passed") is True)
    selected = entries[:_INDEX_LIMIT]
    rows = [(name, _index_status(row), _first_issue(row) or "—") for name, row in selected]
    coverage = f"结果索引：共 {len(entries)} 项，展示 {len(selected)} 项（失败/未确认优先）。"
    if len(entries) > len(selected):
        coverage += f" 另 {len(entries) - len(selected)} 项见 report.json。"
    explanation = ("完整逐用例、检查点、命令、环境和源码指纹保留在同目录 report.json。"
                   "在 GitHub Actions Summary 中，请到本次运行的 Artifacts 下载对应报告；"
                   "下面的相对链接用于下载后同目录查看。")
    markdown = "<details><summary>简短结果索引与完整证据入口（展开详情）</summary>\n\n"
    markdown += coverage + "\n\n| 项目 | 结果 | 首个问题 |\n| --- | --- | --- |\n"
    for row in rows:
        markdown += "| " + " | ".join(escape(value).replace("|", "&#124;") for value in row) + " |\n"
    markdown += "\n" + explanation + "\n\n[完整机器报告](report.json)\n\n</details>\n\n"
    header = "<tr><th>项目</th><th>结果</th><th>首个问题</th></tr>"
    body = "".join("<tr>" + "".join("<td>" + escape(value) + "</td>" for value in row) + "</tr>" for row in rows)
    page = ('<details><summary>简短结果索引与完整证据入口（展开详情）</summary><p>'
            + escape(coverage) + '</p><table><thead>' + header + '</thead><tbody>' + body
            + '</tbody></table><p>' + escape(explanation)
            + '</p><p><a href="report.json">完整机器报告</a></p></details>')
    return markdown, page


def _checkpoint(comparison, name="logits"):
    return _named(comparison.get("checkpoints", []), name)


def _named(rows, name):
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) > 1:
        return {"passed": False, "reason": f"重复的输出比较：{name}"}
    return matches[0] if matches else {}


def _subgraph_output(row):
    """Adapt the existing subgraph schema without rewriting saved evidence."""
    result = dict(row)
    if "max_abs_error" in row:
        result["max_abs"] = row["max_abs_error"]
    first = row.get("firstdiff")
    if isinstance(first, dict) and first.get("reason"):
        result.setdefault("reason", first["reason"])
    return result


def numerical_summary(report):
    """Select explicit output comparisons, never a recursive max over checkpoints."""
    gate, cases = report.get("gate"), report.get("cases", [])
    comparisons, notes = [], []
    description = _short(report.get("scope", "详细范围见完整报告。"), 600)
    if gate in ("numeric:ir-full-qwen3", "audit:w3-full-saved-evidence"):
        mode = report.get("fp32_mode", "未记录")
        description = (f"完整预训练 28 层 Qwen3，FP32、L=256、无 KV cache；同一模型、权重和输入，"
                       f"IR（{mode} 模式）对照 ORT，比较全部位置，包括补齐位置。")
        for filename, label in (("logits.npy", "普通最终 logits"), ("diagnostic_logits.npy", "诊断图最终 logits")):
            comparisons.append((f"IR {mode} 对 ORT：{label}", [
                _named(case.get("comparison", {}).get("logits", []), filename) for case in cases]))
        notes.append("native 是默认 NumPy 算术；reference 是对齐 ORT 浮点行为的兼容模式，两者的通过结果不能互相代替。")
        notes.append("中间检查点用于定位，不混入最终 logits 指标。完整 RISC-V/QEMU 模型执行不在此项范围内。")
        if gate.startswith("audit:"):
            notes.append("本项只重算保存数组，没有重新执行模型，也不算第二人独立运行。")
        notes.append("PPL / zero-shot：未评测；需另行固定真实语料、Tokenizer、上下文窗口和评分方式。")
    elif gate == "w3-medium-host":
        description = "六层随机权重 Qwen3，FP32、L=256；同权重同输入，对比 PyTorch、ORT 和原生 NumPy IR 的最终 logits。"
        recorded_cases = cases
        cases = cases + [{}] * max(0, _MEDIUM_CASE_COUNT - len(cases))
        comparisons.append(("ORT 对 PyTorch", [_checkpoint(c.get("ort", {}).get("normal", {}).get("pytorch_comparison", {})) for c in cases]))
        for level in ("none", "basic", "all"):
            comparisons.append((f"IR 原生 NumPy（{level}）对 ORT", [
                _checkpoint(c.get("ir", {}).get(level, {}).get("normal", {}).get("ort_comparison", {})) for c in cases]))
        notes.extend(("none/basic/all 分别表示关闭、基础、全部 ScratchV IR 优化，不是 C 编译器的优化等级。",
                      "中间检查点和因果/补齐隔离检查仍参与原验收，逐项明细保留在 report.json。",
                      "PPL / zero-shot：未评测；随机权重小模型不用于评价语言能力。"))
        notes.append(f"固定场景覆盖：已记录 {len(recorded_cases)}/{_MEDIUM_CASE_COUNT} 个；"
                     "尚未执行的场景或比较按未记录展示，不作为通过。")
    elif gate == "w3_qwen3_subgraphs":
        length = report.get("sequence_length", "未记录")
        description = (f"预训练第一层的真实维度子图，FP32、L={length}；分别检查 ORT、PyTorch"
                       "与原生 NumPy IR 的子图输出，不是完整模型执行。")
        if length != 256:
            notes.append("本次未记录完整 L=256 验收范围；短序列调试结果不能替代正式验收。")
        comparisons.append(("ORT 对 PyTorch：子图输出", [
            _subgraph_output(c.get("comparisons", {}).get("ordinary_ort_vs_torch", {})) for c in cases]))
        for level in ("none", "basic", "all"):
            comparisons.append((f"IR 原生 NumPy（{level}）对 ORT：子图输出", [
                _subgraph_output(c.get("comparisons", {}).get(f"ordinary_ir_{level}_vs_ort", {})) for c in cases]))
    elif gate == "unit:attention-backend":
        description = "小 Attention 组合图：IR→C→RISC-V→QEMU，同一输入对照 ORT 和独立 NumPy 公式；不是完整 Qwen3。"
        for reference, label in (("ir_vs_ort", "Attention IR 原生 NumPy 对 ORT"),
                                 ("qemu_vs_ort", "Attention C→RISC-V/QEMU 对 ORT"),
                                 ("qemu_vs_numpy", "Attention C→RISC-V/QEMU 对独立 NumPy")):
            comparisons.append((label, [
                next((e.get(reference, {}) for e in c.get("executions", []) if e.get("optimization") == level), {})
                for c in cases for level in ("none", "all")]))
        notes.append("每项汇总关闭/开启 ScratchV IR 优化的结果；C 交叉编译均使用 -O2。")
    elif gate == "preparation:w3":
        notes.append("本项汇总预备检查；子门禁结果见折叠索引，不代表完整模型或 W3 团队验收。")
    elif gate == "w3_layer_diff":
        coverage = report.get("coverage", {})
        notes.append(f"检查点覆盖：{coverage.get('compared_checkpoints', '未记录')}/"
                     f"{coverage.get('total_checkpoints', '未记录')}。只核对保存的检查点，不代表重新执行完整模型。")
        if report.get("partial") is True or coverage.get("complete") is False:
            notes.append("本次仅比较选定检查点（PARTIAL），不能作为完整覆盖通过。")
        if report.get("first_divergence"):
            notes.append("首个差异检查点：" + _short(report["first_divergence"]))
    elif gate == "numeric:w3-full-worker":
        notes.append("本项只执行单个后端并检查普通/诊断图一致性；IR 对 ORT 精度由上层门禁核验。")
    elif gate == "preflight:w3-full-assets":
        notes.append("本项只核验模型资产和静态规模，没有执行完整模型或数值验收。")
    elif gate == "diagnostic:w3-full-localize":
        notes.append("COMPLETE 只表示诊断完成，不表示数值门槛通过；numeric_gate_passed 和团队验收仍独立。")
    return description, comparisons, notes


def summary_views(report):
    description, comparisons, notes = numerical_summary(report)
    if comparisons:
        threshold = report.get("atol")
        if isinstance(threshold, (int, float)) and math.isfinite(threshold):
            notes.insert(0, f"原验收门槛：最大绝对误差 < {threshold:g}，rtol=0；新增两项指标用于观察，不放宽门槛。")
        notes.insert(0, METRIC_EXPLANATION)
        notes.append("表内结果只表示对应输出比较；总体结论仍包含执行、覆盖率和原有诊断检查。")
    if report.get("coverage_complete") is False:
        notes.append("验收覆盖不完整，已测输出通过不能替代全部用例验收。")
    required, selected = report.get("required_cases"), report.get("selected_cases")
    if isinstance(required, list) and isinstance(selected, list):
        notes.append(f"本次选择 {len(selected)}/{len(required)} 个规定用例；已记录 {len(report.get('cases', []))} 个用例。")
    seconds = report.get("elapsed_seconds", report.get("seconds"))
    if isinstance(seconds, (int, float)) and math.isfinite(seconds):
        notes.append(f"本次流程耗时 {seconds:.3f} 秒，包含准备和核验，不代表纯推理速度。")
    if report.get("error"):
        notes.insert(0, "失败原因：" + _short(report["error"], 600))
    if report.get("passed") is not True and report.get("stage"):
        notes.insert(0, "当前/失败阶段：" + _short(report["stage"]))
    failed_cases = [_short(c.get("name", "未命名"), 100) for c in report.get("cases", []) if c.get("passed") is False]
    if failed_cases:
        names = "、".join(failed_cases[:_INDEX_LIMIT])
        remainder = f"，另 {len(failed_cases) - _INDEX_LIMIT} 项" if len(failed_cases) > _INDEX_LIMIT else ""
        notes.insert(0, "未通过的用例：" + names + remainder + "。首个问题见折叠索引，完整定位数据见 report.json。")
    failed_gates = [row for row in report.get("gates", []) if row.get("passed") is not True]
    if failed_gates:
        notes.insert(0, f"子门禁未通过/未确认 {len(failed_gates)} 项，名称与首个错误见折叠索引。")
    failed_invariants = [row for row in report.get("invariants", []) if row.get("passed") is False]
    if failed_invariants:
        notes.insert(0, f"因果/补齐隔离检查未通过 {len(failed_invariants)} 项，见折叠索引。")
    md = escape(description) + "\n\n"
    page = "<p>" + escape(description) + "</p>"
    if comparisons:
        md += compact_metric_table(comparisons) + "\n\n"
        page += compact_metric_table_html(comparisons)
    md += "\n\n".join(escape(note) for note in notes) + "\n\n"
    page += "".join("<p>" + escape(note) + "</p>" for note in notes)
    return md, page
