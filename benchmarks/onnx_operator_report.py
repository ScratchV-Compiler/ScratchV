"""Standalone Markdown/HTML reports for ONNX operator baseline comparisons.

Renderers consume the runner's schema_version=1 dictionary and perform no I/O.
Only successful runs on both revisions are eligible for a timing ratio.
"""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any


_GROUPS = {
    "operators": "独立算子",
    "control": "已有支持对照",
    "controls": "已有支持对照",
    "regressions": "错误回归",
    "composite": "组合链路",
}
_ERRORS = {"PARSE_ERROR", "VERIFY_ERROR", "EXECUTION_ERROR", "NUMERIC_ERROR"}
_HEADERS = (
    "用例", "分组", "修改前", "修改后", "前最大绝对误差", "后最大绝对误差",
    "前解析 ms", "后解析 ms", "前执行 ms", "后执行 ms", "执行耗时比 前/后",
)
_SCOPE = (
    "执行计时覆盖完整 IRInterpreter.run，包括 IR 校验与输入/权重复制；"
    "不含 ONNX 解析和 ONNX Runtime 参考计算。ONNX → IR 解析单独计时。"
)
_LIMIT = (
    "本次改动扩展算子支持并修正计算语义；计时用于本地观测，"
    "不代表性能优化结论。只有修改前后均 PASS 的用例才计算执行耗时比。"
    "不支持或缺少有效测量时显示“—”，不能解释为零耗时或无限加速。"
)


def _text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _cell(value: Any) -> str:
    return _escape(value).replace("|", "&#124;").replace("\r\n", "\n").replace("\n", "<br>")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _error(value: Any) -> str:
    number = _number(value)
    return f"{number:.3e}" if number is not None and number >= 0 else "—"


def _status(row: dict) -> str:
    return str(row.get("status") or "UNKNOWN")


def _milliseconds(row: dict, field: str) -> str:
    if _status(row) == "UNSUPPORTED":
        return "—"
    if field == "run_median_s" and _status(row) != "PASS":
        return "—"
    value = _number(row.get(field))
    if value is None or value <= 0 or not math.isfinite(value * 1000):
        return "—"
    return f"{value * 1000:.4f}"


def _ratio(before: dict, after: dict) -> str:
    if _status(before) != "PASS" or _status(after) != "PASS":
        return "—"
    left, right = _number(before.get("run_median_s")), _number(after.get("run_median_s"))
    if left is None or right is None or left <= 0 or right <= 0:
        return "—"
    ratio = left / right
    return f"{ratio:.3f}×" if math.isfinite(ratio) else "—"


def _transition(case: dict) -> str:
    before, after = _status(case.get("before", {})), _status(case.get("after", {}))
    if after == "PASS" and before == "UNSUPPORTED":
        return "原先不支持 → 现通过"
    if after == "PASS" and before in _ERRORS:
        return "原先错误 → 现正确"
    if before == after == "PASS":
        return "前后均通过"
    if before == "PASS" and after != "PASS":
        return "出现回归"
    return "仍需检查" if after != "PASS" else "参考恢复后通过"


def _counts(results: list[dict]) -> dict:
    return {
        "total": len(results),
        "passed": sum(_status(case.get("after", {})) == "PASS" for case in results),
        "supported": sum(_transition(case) == "原先不支持 → 现通过" for case in results),
        "corrected": sum(_transition(case) == "原先错误 → 现正确" for case in results),
        "regressed": sum(_transition(case) == "出现回归" for case in results),
    }


def _outcome(report: dict) -> str:
    return "通过" if report.get("passed") is True and not report.get("runner_error") else "未通过 / 待检查"


def _summary(report: dict, results: list[dict]) -> str:
    count = _counts(results)
    return (
        f"总验收：{_outcome(report)}。修改后 {count['passed']}/{count['total']} 个用例通过；"
        f"原先不支持、现通过的用例 {count['supported']} 个，"
        f"原先错误、现通过的用例 {count['corrected']} 个，"
        f"出现回归的用例 {count['regressed']} 个。用例数不等于新增算子种类数。"
    )


def _group_rows(results: list[dict]) -> list[list[Any]]:
    groups: dict[str, list[dict]] = {}
    for case in results:
        group = case.get("group", "unknown")
        group = "control" if group == "controls" else group
        groups.setdefault(group, []).append(case)
    order = {name: index for index, name in enumerate(("operators", "control", "regressions", "composite"))}
    rows = []
    for group in sorted(groups, key=lambda name: (order.get(name, len(order)), name)):
        cases = groups[group]
        total = len(cases)
        before = sum(_status(case.get("before", {})) == "PASS" for case in cases)
        after = sum(_status(case.get("after", {})) == "PASS" for case in cases)
        rows.append([_GROUPS.get(group, group), total, f"{before}/{total}", f"{after}/{total}"])
    return rows


def _metadata(report: dict) -> list[tuple[str, Any]]:
    baseline, current = report.get("baseline", {}), report.get("current", {})
    execution_order = report.get("execution_order")
    if execution_order == ["before", "after", "after", "before"]:
        execution_order = "ABBA：修改前 → 修改后 → 修改后 → 修改前"
    return [
        ("报告 schema", report.get("schema_version")),
        ("基线提交", baseline.get("commit")),
        ("基线源码 SHA-256", baseline.get("source_sha256")),
        ("当前 HEAD", current.get("head")),
        ("当前包含未提交改动", current.get("dirty")),
        ("当前源码 SHA-256", current.get("source_sha256")),
        ("执行顺序", execution_order),
        ("每版本轮数", report.get("rounds_per_variant")),
        ("每轮预热次数", report.get("warmup")),
        ("每轮正式计时次数", report.get("repeats_per_round")),
        ("每版本正式样本总数", report.get("repeats")),
        ("计时统计", "合并各轮正式样本后计算中位数（ms），预热不计入；原始秒数见用例详情"),
        ("runner 计时说明", report.get("timing_scope")),
    ]


def _case_cells(case: dict) -> list[Any]:
    before, after = case.get("before", {}), case.get("after", {})
    return [
        case.get("name", "未命名"), _GROUPS.get(case.get("group"), case.get("group", "—")),
        _status(before), _status(after), _error(before.get("max_abs_error")),
        _error(after.get("max_abs_error")), _milliseconds(before, "parse_median_s"),
        _milliseconds(after, "parse_median_s"), _milliseconds(before, "run_median_s"),
        _milliseconds(after, "run_median_s"), _ratio(before, after),
    ]


def _case_metadata(case: dict) -> list[tuple[str, Any]]:
    return [
        ("算子", case.get("operators", [])),
        ("绝对容差 atol", case.get("atol")),
        ("相对容差 rtol", case.get("rtol")),
        ("模型 SHA-256", case.get("model_sha256")),
        ("输入 SHA-256", case.get("inputs_sha256")),
        ("参考输出 SHA-256 (expected_sha256)", case.get("expected_sha256")),
        ("初始化器 SHA-256 (initializers_sha256)", case.get("initializers_sha256")),
    ]


def _detail_fields(row: dict) -> list[tuple[str, Any]]:
    return [
        ("状态 / 阶段", f"{_status(row)} / {row.get('phase') or '—'}"),
        ("实际 shape / dtype", f"{_text(row.get('actual_shape'))} / {_text(row.get('actual_dtype'))}"),
        ("参考 shape / dtype", f"{_text(row.get('expected_shape'))} / {_text(row.get('expected_dtype'))}"),
        ("最大绝对误差", _error(row.get("max_abs_error"))),
        ("最大相对误差", _error(row.get("max_rel_error"))),
        ("执行 IR 步数", row.get("executed_steps")),
        ("解析原始样本 s", row.get("parse_samples_s", [])),
        ("执行原始样本 s", row.get("run_samples_s", [])),
    ]


def _markdown_table(headers: tuple | list, rows: list) -> str:
    lines = ["| " + " | ".join(_cell(item) for item in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(_cell(item) for item in row) + " |" for row in rows]
    return "\n".join(lines)


def _fenced(text: str) -> str:
    longest = max((len(part) for part in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _audit_rows(audit: dict) -> list[list[Any]]:
    before, after = audit.get("before", {}), audit.get("after", {})
    keys = sorted(set(before) | set(after))
    if "status" in keys:
        keys.remove("status")
        keys.insert(0, "status")
    return [[key, before.get(key), after.get(key)] for key in keys]


def render_markdown(report: dict) -> str:
    """Render schema version 1 as Markdown without writing any files."""
    results = list(report.get("results", []))
    lines = ["# ONNX 算子支持与正确性：修改前后对照", "", _summary(report, results), ""]
    if report.get("runner_error"):
        lines += ["**runner 准备/执行失败：本次运行未完成，不能作为通过报告。**", "",
                  _fenced(str(report["runner_error"])), ""]
    lines += [_LIMIT, ""]
    if report.get("binding_policy"):
        lines += ["## 基线兼容适配与输入绑定", "",
                  "输入绑定兼容适配不属于基线编译器实现；以下为 runner 的实际策略，需与数值结果一起审阅。", "",
                  "> " + _cell(report["binding_policy"]), ""]
    lines += ["## 按组通过情况", "",
             _markdown_table(("分组", "用例总数", "修改前 PASS", "修改后 PASS"), _group_rows(results)), "",
             "## 版本与测量条件", "", _markdown_table(("项目", "值"), _metadata(report)),
             "", _SCOPE, "", "环境：", "",
             _markdown_table(("项目", "值"), sorted(report.get("environment", {}).items())),
             "", "## 全部用例", "",
             "耗时比 = 修改前 / 修改后；大于 1 表示该次观测中修改后较快，小于 1 表示较慢。", "",
             _markdown_table(_HEADERS, [_case_cells(case) for case in results]), "", "## 用例与错误详情", ""]
    for case in results:
        title = f"{case.get('name', '未命名')} · {_transition(case)}"
        lines += [f"<details><summary>{_escape(title)}</summary>", "",
                  _cell(case.get("description", "")), "",
                  _markdown_table(("项目", "值"), _case_metadata(case)), ""]
        for label, row in (("修改前", case.get("before", {})), ("修改后", case.get("after", {}))):
            lines += [f"**{label}**", "", _markdown_table(("项目", "值"), _detail_fields(row)), ""]
            if row.get("error"):
                lines += ["错误详情：", "", _fenced(str(row["error"])), ""]
        lines += ["</details>", ""]
    audit = report.get("full_model_audit")
    if isinstance(audit, dict):
        lines += ["## 完整 Qwen 模型结构审计（仅解析）", "",
                  "此项只检查 ONNX → IR 解析及结构校验，未执行完整模型推理，不提供推理耗时或加速比。", "",
                  _markdown_table(("元数据", "值"), [(key, value) for key, value in audit.items()
                                                      if key not in {"before", "after"}]), "",
                  _markdown_table(("项目", "修改前", "修改后"), _audit_rows(audit)), ""]
    return "\n".join(lines).rstrip() + "\n"


def _html_table(headers: tuple | list, rows: list) -> str:
    head = "".join(f"<th scope='col'>{_escape(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(item)}</td>" for item in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _badge(status: str) -> str:
    css = "pass" if status == "PASS" else "unsupported" if status == "UNSUPPORTED" else "error"
    return f"<span class='badge {css}'>{_escape(status)}</span>"


_CSS = """
:root{font-family:Inter,'Segoe UI','Microsoft YaHei',sans-serif;color:#172b4d;background:#f4f7fb;line-height:1.6}
*{box-sizing:border-box}body{margin:0}main{max-width:1680px;margin:0 auto;padding:36px 28px 64px}
h1{font-size:28px;line-height:1.35;margin:0 0 12px}h2{font-size:21px;margin:30px 0 12px}h3{font-size:17px;margin:14px 0 8px}
p{margin:10px 0}.lead{font-size:17px}.note{background:#edf4fc;border-left:4px solid #2c64a8;padding:14px 18px;border-radius:4px}
.runner-error{border:2px solid #a02332;border-radius:8px;background:#fff0f1;padding:16px;margin:18px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:22px 0}
.card{padding:16px;background:white;border:1px solid #dde5ef;border-radius:10px}.card strong{display:block;font-size:27px}.card span{color:#52647d;font-size:14px}
.table-wrap{overflow-x:auto;border:1px solid #dde5ef;border-radius:8px;background:white;margin:10px 0 18px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:11px 13px;border-bottom:1px solid #e3eaf2;text-align:left;vertical-align:top}
th{background:#eaf0f8;font-weight:650;white-space:nowrap}td{overflow-wrap:anywhere;white-space:pre-wrap}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f7faff}
.cases{min-width:1450px}.cases td:nth-child(n+5){font-variant-numeric:tabular-nums;white-space:nowrap}.cases td:first-child{min-width:230px}
.badge{display:inline-block;padding:2px 7px;border-radius:5px;font-size:12px;font-weight:650;white-space:nowrap}
.pass{color:#116344;background:#e2f5eb}.unsupported{color:#885000;background:#fff0d6}.error{color:#a02332;background:#fce6e9}
.muted{color:#63748c;font-size:12px}.transition{display:block;margin-top:5px;color:#355c82;font-size:12px}
details{background:white;border:1px solid #dde5ef;border-radius:8px;padding:12px 16px;margin:10px 0}summary{cursor:pointer;font-weight:650;overflow-wrap:anywhere}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.columns>section{min-width:0}pre{background:#172b4d;color:#e7edf6;border-radius:6px;padding:14px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.filter{display:flex;align-items:center;gap:12px;margin:14px 0}.filter input{font:inherit;padding:9px 12px;width:min(560px,100%);border:1px solid #aabbd0;border-radius:6px;background:white}
[hidden]{display:none!important}footer{margin-top:30px;color:#63748c;font-size:12px}@media(max-width:800px){main{padding:24px 14px}h1{font-size:23px}.columns{grid-template-columns:1fr}}
@media print{body{background:white}main{padding:0}.filter{display:none}.table-wrap{overflow:visible}.cases{min-width:0;font-size:9px}th,td{padding:5px}details{break-inside:avoid}}
"""


def render_html(report: dict) -> str:
    """Render a self-contained, escaped HTML report with optional local filtering."""
    results = list(report.get("results", []))
    count = _counts(results)
    cards = [("修改后通过", f"{count['passed']} / {count['total']}"),
             ("原先不支持、现通过的用例数", count["supported"]),
             ("原先错误、现通过的用例数", count["corrected"]), ("出现回归的用例数", count["regressed"])]
    parts = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>ONNX 算子支持与正确性对照</title>", f"<style>{_CSS}</style></head><body><main>",
             "<h1>ONNX 算子支持与正确性：修改前后对照</h1>",
             f"<p class='lead'>{_escape(_summary(report, results))}</p>"]
    if report.get("runner_error"):
        parts += ["<div class='runner-error' role='alert'><strong>runner 准备/执行失败：本次运行未完成，不能作为通过报告。</strong>",
                  f"<pre>{_escape(report['runner_error'])}</pre></div>"]
    parts.append("<div class='cards'>")
    parts += [f"<div class='card'><strong>{_escape(value)}</strong><span>{_escape(label)}</span></div>"
              for label, value in cards]
    parts += [f"</div><p class='note'>{_escape(_LIMIT)}</p>"]
    if report.get("binding_policy"):
        parts += ["<h2>基线兼容适配与输入绑定</h2>",
                  "<p>输入绑定兼容适配不属于基线编译器实现；以下为 runner 的实际策略，需与数值结果一起审阅。</p>",
                  f"<p class='note'>{_escape(report['binding_policy'])}</p>"]
    parts += ["<h2>按组通过情况</h2>",
              _html_table(("分组", "用例总数", "修改前 PASS", "修改后 PASS"), _group_rows(results)),
              "<h2>版本与测量条件</h2>",
              _html_table(("项目", "值"), _metadata(report)), f"<p>{_escape(_SCOPE)}</p>",
              "<details><summary>运行环境</summary>",
              _html_table(("项目", "值"), sorted(report.get("environment", {}).items())), "</details>",
              "<h2>全部用例</h2><p>耗时比 = 修改前 / 修改后；大于 1 表示该次观测中修改后较快，小于 1 表示较慢。</p>",
              "<label class='filter'>筛选用例<input id='case-filter' type='search' placeholder='用例、算子、分组或状态' autocomplete='off'></label>",
              "<div class='table-wrap'><table class='cases'><thead><tr>"]
    parts += [f"<th scope='col'>{_escape(label)}</th>" for label in _HEADERS]
    parts.append("</tr></thead><tbody>")
    for index, case in enumerate(results):
        cells = _case_cells(case)
        searchable = " ".join(_text(value) for value in [*cells[:4], case.get("operators", []), case.get("description", "")])
        parts.append(f"<tr data-case='{_escape(searchable.casefold())}'>")
        parts.append(f"<td><a href='#case-{index}'>{_escape(cells[0])}</a>"
                     f"<span class='transition'>{_escape(_transition(case))}</span></td>")
        for column, value in enumerate(cells[1:], 1):
            parts.append(f"<td>{_badge(value) if column in (2, 3) else _escape(value)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div><h2>用例与错误详情</h2>")
    for index, case in enumerate(results):
        searchable = " ".join(_text(value) for value in [*_case_cells(case)[:4], case.get("operators", []), case.get("description", "")])
        parts.append(f"<details id='case-{index}' data-case='{_escape(searchable.casefold())}'>"
                     f"<summary>{_escape(case.get('name', '未命名'))} · {_escape(_transition(case))}</summary>")
        parts.append(f"<p>{_escape(case.get('description', ''))}</p>")
        parts.append(_html_table(("项目", "值"), _case_metadata(case)))
        parts.append("<div class='columns'>")
        for label, row in (("修改前", case.get("before", {})), ("修改后", case.get("after", {}))):
            parts.append(f"<section><h3>{label} {_badge(_status(row))}</h3>")
            parts.append(_html_table(("项目", "值"), _detail_fields(row)))
            if row.get("error"):
                parts.append(f"<h3>错误详情</h3><pre>{_escape(row['error'])}</pre>")
            parts.append("</section>")
        parts.append("</div></details>")
    audit = report.get("full_model_audit")
    if isinstance(audit, dict):
        parts += ["<h2>完整 Qwen 模型结构审计（仅解析）</h2>",
                  "<p class='note'>此项只检查 ONNX → IR 解析及结构校验，未执行完整模型推理，不提供推理耗时或加速比。</p>",
                  _html_table(("元数据", "值"), [(key, value) for key, value in audit.items()
                                                if key not in {"before", "after"}]),
                  _html_table(("项目", "修改前", "修改后"), _audit_rows(audit))]
    parts += ["<footer>独立 HTML 报告，无外部资源依赖。误差、状态及版本指纹由同一份 runner JSON 提供。</footer></main>",
              """<script>
const filter = document.getElementById('case-filter');
filter.addEventListener('input', () => {
  const query = filter.value.trim().toLowerCase();
  document.querySelectorAll('[data-case]').forEach(el => {
    el.hidden = !el.dataset.case.includes(query);
  });
});
document.querySelectorAll('a[href^="#case-"]').forEach(link => {
  link.addEventListener('click', () => {
    const detail = document.getElementById(link.getAttribute('href').slice(1));
    if (detail) detail.open = true;
  });
});
</script></body></html>"""]
    return "\n".join(parts) + "\n"
