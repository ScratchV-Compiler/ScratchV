"""Report W6 performance and numerical checks for one fresh run directory."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys


MATMUL = {"dense": "2x4-column-panel", "attention_scores": "4x4-checked-parallel-copy"}
VOCAB = 151936
ROWS = 256
ATOL = 1e-3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"invalid measured value {value!r}")
    return value


def error_ok(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value < ATOL


def load(path, commit, key="commit"):
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report[key] == commit, "result belongs to a different commit")
    return report


def timed_trials(rows, variant, name, timer, numeric):
    group = [r for r in rows if r["variant"] == variant and r["case"] == name]
    expected = {(0, "warmup"), (1, "measure"), (2, "measure"), (3, "measure")}
    require(len(group) == 4 and {(r["round"], r["phase"]) for r in group} == expected,
            f"{name}/{variant}: expected one warmup and three distinct measured rounds")
    require(all(numeric(r) for r in group), f"{name}/{variant}: invalid numerical output")
    return statistics.median(positive(r[timer]) for r in group if r["phase"] == "measure")


def matmul(root, commit, case):
    data = load(root / "matmul/result.json", commit)
    require(data["cores"] == 4 and data["workspace_layout"] == "none" and data["dtype"] == "float32",
            "unexpected MatMul configuration")
    require(data["required_variants"] == MATMUL, "measured MatMul selections differ from the fixed gate")
    shapes = {"dense": [[256, 1024], [1024, 1024]], "attention_scores": [[16, 256, 128], [16, 128, 256]]}
    require(data["shapes"][case] == shapes[case], f"unexpected {case} shape")
    baseline, candidate = "1x4-original", MATMUL[case]
    def numeric(row):
        return error_ok(row["max_abs_vs_ort"]) and row["bitwise_equal"] is True
    old, new = (timed_trials(data["runs"], v, case, "forward_s", numeric) for v in (baseline, candidate))
    gain = old / new
    return gain >= 1.5, f"{baseline} → {candidate}, {old:.4f} / {new:.4f} s = {gain:.3f}× (≥1.5×)"


def lm_head(root, commit):
    report = load(root / "lm-head/result.json", commit)
    gains = {}
    for name, vocab in (("small", 257), ("real", VOCAB)):
        case = report["cases"][name]
        setup = case["setup"]
        require((setup["vocab"], setup["cores"], setup["dtype"], setup["padding"]) ==
                (vocab, 4, "float32", "none"), f"{name}: unexpected LM Head configuration")
        def numeric(r):
            return (r["elements"] == ROWS * vocab and r["greedy_matches"] == ROWS
                    and r["finite"] is True and r["greedy_mismatch_rows"] == []
                    and error_ok(r["max_abs_vs_ort"]))
        old, new = (timed_trials(case["runs"], v, name, "guest_s", numeric)
                    for v in ("baseline", "vocab_blocked"))
        gains[name] = old / new
    return gains["real"] >= 2, f"151,936 words {gains['real']:.3f}× (≥2×); 257 words {gains['small']:.3f}×"


def full_result(root, commit):
    report = load(root / "full/comparison.json", commit, key="git_head")
    require(report["case"] == "full_seed_0" and report["layers"] == 28 and
            report["shape"] == [1, ROWS, VOCAB] and set(report["arms"]) == {"original", "lm-head-only", "vocab-blocked"},
            "expected full_seed_0 28-layer A/B")
    require(all(report[k] is True for k in ("inputs_equal", "ort_bitwise_equal", "same_weight_manifest",
                                            "same_other_rhs_panels", "same_candidate_elf", "same_candidate_output",
                                            "same_candidate_bundle")), "inputs, reference, weights or controls differ")
    require(report["candidate_weight_transport"] == "file-backed-ram",
            "full comparison did not use the selected candidate transport")
    require(report["controls"] == ["lm-head-only"], "missing LM Head-only measurement")
    flags = report["flags"]
    require(type(flags) is list and len(flags) % 2 == 0 and len(set(flags[::2])) == len(flags[::2]),
            "invalid full forward flags")
    expected_flags = {"--case": "full_seed_0", "--cores": "4", "--matmul-policy": "blocked_fma",
                      "--matmul-tile": "2x4", "--matmul-rhs-layout": "column-panel",
                      "--mean-policy": "two-four-lane-last-axis", "--nonlinear-policy": "fp32-reference",
                      "--qemu-workspace-layout": "none"}
    measured_flags = dict(zip(flags[::2], flags[1::2]))
    require(all(measured_flags.get(k) == v for k, v in expected_flags.items()),
            "unexpected full forward settings (case, cores, layout or arithmetic policy)")
    for name, arm in report["arms"].items():
        require(arm["lm_head_policy"] == ("original" if name == "original" else "vocab-blocked"),
                f"{name}: wrong LM Head build policy")
        expected = "file-backed-ram" if name == "vocab-blocked" else "loader"
        require(arm["qemu_weight_transport"] == expected, f"{name}: wrong QEMU transport")
    return report


def memory(root, commit):
    report = full_result(root, commit)
    arms = report["arms"]
    for name, arm in arms.items():
        duration = positive(arm["qemu_s"])
        # collect_pair samples every 50 ms; the runner's independent QEMU RSS sampler every 200 ms.
        for count, errors, interval in (("samples", "read_errors", .05), ("qemu_samples", "qemu_read_errors", .2)):
            n, failed = arm[count], arm[errors]
            require(type(n) is int and n > 0 and type(failed) is int and failed >= 0
                    and n * interval >= .75 * duration and failed / (n + failed) <= .05,
                    f"{name}: insufficient RSS coverage ({count}={n}, errors={failed}, QEMU={duration:.1f}s)")
        controller, guest = (positive(arm[k]) for k in ("controller_bytes_at_peak", "qemu_bytes_at_peak"))
        peak = positive(arm["combined_peak_bytes"])
        qemu_peak = positive(arm["qemu_sampled_peak_bytes"])
        require(peak == controller + guest and peak >= .95 * qemu_peak,
                f"{name}: RSS peak components or independent QEMU peak disagree")
        exit_peak = arm["qemu_exit_peak_bytes"]
        if exit_peak is not None:
            require(max(guest, qemu_peak) <= positive(exit_peak) * 1.05,
                    f"{name}: sampled QEMU memory exceeds process high water")
    old, new = (positive(arms[k]["combined_peak_bytes"]) for k in ("original", "vocab-blocked"))
    gain = (old - new) / old
    return gain >= .20, f"controller + QEMU peak {old / 2**20:.1f} → {new / 2**20:.1f} MiB, {gain:.2%} (≥20%)"


def regression(root, commit):
    report = full_result(root, commit)
    require(report["status"] == "PASS", "full comparison failed")
    expected = report["ort_greedy_ids"]
    require(len(expected) == ROWS and all(type(i) is int and 0 <= i < VOCAB for i in expected),
            "missing ORT token IDs")
    maxima = []
    for name in report["arms"]:
        result = report["numeric"][name]
        require(report["arms"][name]["fma_passed"] is True and result["finite"] is True
                and result["elements"] == ROWS * VOCAB and error_ok(result["max_abs_vs_ort"])
                and result["token_ids"] == expected and result["greedy_matches"] == ROWS
                and result["greedy_mismatch_rows"] == [], f"{name}: full numerical or token mismatch")
        maxima.append(result["max_abs_vs_ort"])
    return True, f"full_seed_0, all three arms compared across complete logits and all 256 token IDs; max abs {max(maxima):.6g} (<0.001)"


def summarize(root, commit, mode):
    rows = []
    def check(label, operation, *args, selected=True):
        if not selected:
            rows.append((label, "NOT RUN", "full mode only"))
            return
        try:
            ok, message = operation(root, commit, *args)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            ok, message = False, f"missing or invalid evidence: {exc}"
        rows.append((label, "PASS" if ok else "FAIL", message))

    for name in MATMUL:
        check(f"perf:matmul / {name}", matmul, name)
    full = mode == "full"
    check("perf:lmhead", lm_head, selected=full)
    check("perf:memory", memory, selected=full)
    check("regression:numeric / W6 full_seed_0", regression, selected=full)
    success = all(status != "FAIL" for _, status, _ in rows)
    title = ("PASS" if full else "small checks passed") if success else "FAIL"
    lines = [f"## W6 {mode} · {title}", "", f"Commit `{commit}`", "",
             "| Check | Result | Measurement |", "| --- | --- | --- |"]
    for label, status, message in rows:
        lines.append(f"| {label} | {status} | {message} |")
    lines += ["", "Coverage: the selected W6 measurements from this run. "
              "Earlier-stage numerical checks and W5 generation have independent evidence. "
              "Milestone sign-off is not evaluated here."]
    return "\n".join(lines) + "\n", success


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--mode", choices=("small", "full"), required=True)
    args = cli.parse_args()
    text, passed = summarize(args.run_dir, args.commit, args.mode)
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
