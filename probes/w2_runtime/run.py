#!/usr/bin/env python3
"""W2 host-runtime gate against pinned official tokenizer assets and two oracles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST_PATH = ROOT / "probes/w1_qwen3_export/manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
CORPUS = Path(__file__).with_name("corpus.json")
PINNED = {"numpy": "2.2.6", "tokenizers": "0.21.4", "transformers": "4.51.3"}
MAX_FILE_BYTES = 16 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(directory, manifest=MANIFEST):
    directory = Path(directory).resolve()
    rows, seen = [], set()
    for item in manifest["source_files"]:
        name = item["name"]
        if (not name or name in seen or name in (".", "..")
                or any(c in name for c in "/\\:")):
            raise ValueError(f"Non-flat or duplicate asset name: {name}")
        seen.add(name)
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise FileNotFoundError(f"Missing tokenizer asset: {name}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES or sha256(path) != item["sha256"]:
            raise ValueError(f"Tokenizer asset SHA256/size mismatch: {name}")
        rows.append({**item, "bytes": size})
    required = {"tokenizer.json", "tokenizer_config.json", "config.json",
                "generation_config.json", "vocab.json", "merges.txt", "LICENSE"}
    if seen != required:
        raise ValueError("Tokenizer gate requires exactly the seven pinned source files")
    return rows


def download_file(url, destination):
    """Bound downloads; retry transient server failures without trusting a cache."""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ScratchV-W2-runtime"})
            with urllib.request.urlopen(request, timeout=30) as incoming, Path(destination).open("wb") as out:
                count = 0
                while block := incoming.read(1024 * 1024):
                    count += len(block)
                    if count > MAX_FILE_BYTES:
                        raise ValueError("Tokenizer download exceeds per-file limit")
                    out.write(block)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(attempt + 1)


def acquire_assets(directory, manifest=MANIFEST):
    directory = Path(directory).resolve()
    if directory.exists():
        return {"reused": True, "files": verify_assets(directory, manifest)}
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen3-tokenizer-", dir=directory.parent) as tmp:
        stage = Path(tmp).resolve()
        for item in manifest["source_files"]:
            name = item["name"]
            if not name or name in (".", "..") or any(c in name for c in "/\\:"):
                raise ValueError(f"Invalid tokenizer asset name: {name}")
            url = f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}/{name}"
            download_file(url, stage / name)
        rows = verify_assets(stage, manifest)
        if not stage.is_relative_to(directory.parent) or directory.exists():
            raise ValueError("Tokenizer destination changed during download")
        stage.rename(directory)
    return {"reused": False, "files": rows}


def require_environment():
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if sys.version_info[:2] != (3, 12) or versions != PINNED:
        raise RuntimeError(f"Use Python 3.12 and the pinned CPU probe environment: {versions}")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "reference": "Official Transformers fast and slow tokenizers, offline"}


def checkout_evidence():
    """Source archives remain runnable; never attribute them to a parent repo."""
    result = {"root": str(ROOT), "head": None, "status": None, "clean": None}
    if not (ROOT / ".git").exists():
        result["unavailable"] = "Source directory has no .git entry; use source_sha256 for identity"
        return result
    try:
        for key, arguments in (("head", ["rev-parse", "HEAD"]),
                               ("status", ["status", "--porcelain=v1", "--untracked-files=all"])):
            result[key] = subprocess.check_output(
                ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                stderr=subprocess.PIPE, timeout=10,
            ).rstrip("\r\n")
        result["clean"] = not bool(result["status"])
    except (OSError, subprocess.SubprocessError) as exc:
        result["unavailable"] = f"{type(exc).__name__}: {exc}"
    return result


def tokenizer_cases(adapter, fast, slow, corpus, rows):
    if not corpus["cases"]:
        raise ValueError("Tokenizer corpus must not be empty")
    names = set()
    for case in corpus["cases"]:
        if case["name"] in names:
            raise ValueError("Duplicate corpus case name")
        names.add(case["name"])
        row = {"name": case["name"], "passed": False, "checks": 0}
        rows.append(row)
        # Exercise public defaults as well as explicit flags. A default-only
        # regression would otherwise leave every explicit-option check green.
        default_ids = adapter.encode(case["text"])
        default_text = adapter.decode(default_ids)
        for reference in (fast, slow):
            if default_ids != reference.encode(case["text"]) or default_ids != case["ids"]:
                raise ValueError(f"{case['name']}: default encoding differs from official/golden IDs")
            row["checks"] += 1
            if default_text != reference.decode(default_ids) or default_text != case["decoded"]:
                raise ValueError(f"{case['name']}: default decode differs from official/golden text")
            row["checks"] += 1
        for special in (False, True):
            actual = adapter.encode(case["text"], add_special_tokens=special)
            for reference in (fast, slow):
                expected = reference.encode(case["text"], add_special_tokens=special)
                if actual != expected or actual != case["ids"]:
                    raise ValueError(f"{case['name']}: encoding differs from official/golden IDs")
                row["checks"] += 1
            for skip in (False, True):
                for cleanup in (False, True):
                    result = adapter.decode(actual, skip_special_tokens=skip,
                                            clean_up_tokenization_spaces=cleanup)
                    for reference in (fast, slow):
                        expected = reference.decode(actual, skip_special_tokens=skip,
                                                    clean_up_tokenization_spaces=cleanup)
                        if result != expected:
                            raise ValueError(f"{case['name']}: decode differs (skip={skip}, cleanup={cleanup})")
                        row["checks"] += 1
                    if not skip and not cleanup and result != case["decoded"]:
                        raise ValueError(f"{case['name']}: decode differs from golden text")
        row.update(passed=True, token_count=len(case["ids"]),
                   raw_text_preserved=case["decoded"] == case["text"])


def tokenizer_metadata(adapter, fast, slow, directory):
    """Check metadata independently: model rows are not tokenizer membership."""
    directory = Path(directory)
    model = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((directory / "generation_config.json").read_text(encoding="utf-8"))
    expected_eos = generation["eos_token_id"]
    if isinstance(expected_eos, int):
        expected_eos = [expected_eos]
    if adapter.model_vocab_size != model["vocab_size"]:
        raise ValueError("Tokenizer model_vocab_size differs from the official model configuration")
    if adapter.generation_eos_token_ids != tuple(expected_eos):
        raise ValueError("Tokenizer generation EOS IDs differ from the official generation configuration")
    if adapter.pad_token_id != generation["pad_token_id"]:
        raise ValueError("Tokenizer pad ID differs from the official generation configuration")
    for reference in (fast, slow):
        if (adapter.base_vocab_size != reference.vocab_size
                or adapter.tokenizer_vocab_size != len(reference)
                or adapter.valid_token_ids != frozenset(reference.get_vocab().values())):
            raise ValueError("Tokenizer vocabulary metadata differs from the official reference")
        for name in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
            if getattr(adapter, name) != getattr(reference, name):
                raise ValueError(f"Tokenizer {name} differs from the official reference")
    return {"model_vocab_size": adapter.model_vocab_size,
            "base_vocab_size": adapter.base_vocab_size,
            "tokenizer_vocab_size": adapter.tokenizer_vocab_size,
            "pad_token_id": adapter.pad_token_id,
            "eos_token_id": adapter.eos_token_id,
            "bos_token_id": adapter.bos_token_id,
            "unk_token_id": adapter.unk_token_id,
            "generation_eos_token_ids": list(adapter.generation_eos_token_ids)}


def input_cases(adapter, rows):
    import numpy as np
    from scratchv.runtime.llm_inputs import prepare_inputs, greedy_next_token, generation_stop_reason
    # Explicit loop oracle, independent of the vectorized production mask.
    for length in (1, 17, 255, 256):
        ids = [42] * length
        ids[0] = adapter.pad_token_id  # A pad-like ID inside the prompt remains a valid token.
        prepared = prepare_inputs(ids, pad_token_id=adapter.pad_token_id,
                                  vocab_size=adapter.model_vocab_size)
        expected = np.full((1, 1, 256, 256), np.finfo(np.float32).min, dtype=np.float32)
        for query in range(256):
            for key in range(min(query + 1, length)):
                expected[0, 0, query, key] = 0
        if (prepared.valid_length != length or prepared.input_ids.dtype != np.int64
                or prepared.input_ids.shape != (1, 256)
                or not np.array_equal(prepared.input_ids[0, :length], ids)
                or not np.all(prepared.input_ids[0, length:] == adapter.pad_token_id)
                or prepared.attention_mask.dtype != np.float32
                or not np.array_equal(prepared.attention_mask, expected)):
            raise ValueError(f"Input/mask mismatch for valid length {length}")
        logits = np.zeros((1, 256, 8), dtype=np.float32)
        logits[0, -1, 7] = 100
        logits[0, length - 1, 7] = 0
        logits[0, length - 1, [2, 5]] = 1
        if greedy_next_token(logits, length, vocab_size=8) != 2:
            raise ValueError("Greedy used a padding row or wrong tie rule")
        rows.append({"name": f"mask-and-greedy-{length}", "passed": True})
    for eos in adapter.generation_eos_token_ids:
        reason = generation_stop_reason(valid_length=2, generated_tokens=1, max_new_tokens=3,
                                        last_token_id=eos, eos_token_ids=adapter.generation_eos_token_ids)
        if reason != "eos":
            raise ValueError(f"Generation EOS not recognized: {eos}")
    if generation_stop_reason(valid_length=256, generated_tokens=0, max_new_tokens=1) != "capacity":
        raise ValueError("Full context must stop before a new forward call")
    if generation_stop_reason(valid_length=1, generated_tokens=0, max_new_tokens=0) != "max_new_tokens":
        raise ValueError("Zero generation budget must stop")
    rows.append({"name": "eos-capacity-budget", "passed": True})


def run_checks(args, report):
    report["stage"] = "environment"
    report["environment"] = require_environment()
    report["stage"] = "assets"
    if args.mode == "download":
        report["acquisition"] = acquire_assets(args.tokenizer_dir)
    report["files"] = verify_assets(args.tokenizer_dir)
    report["stage"] = "tokenizer"
    from transformers import AutoTokenizer
    from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer
    adapter = Qwen3Tokenizer.from_directory(args.tokenizer_dir)
    fast = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True, local_files_only=True,
                                        trust_remote_code=False)
    slow = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True,
                                        trust_remote_code=False)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if corpus["revision"] != MANIFEST["revision"] or corpus["model_id"] != MANIFEST["model_id"]:
        raise ValueError("Corpus/model identity mismatch")
    if (type(fast).__name__, type(slow).__name__) != ("Qwen2TokenizerFast", "Qwen2Tokenizer"):
        raise ValueError("Expected two distinct official tokenizer reference classes")
    report["tokenizer"] = tokenizer_metadata(adapter, fast, slow, args.tokenizer_dir)
    tokenizer_cases(adapter, fast, slow, corpus, report["tokenizer_cases"])
    report["stage"] = "inputs-sampling"
    input_cases(adapter, report["runtime_cases"])
    report.update(passed=True, stage="complete")


def save_report(directory, report):
    """Keep the primary failure and make incomplete evidence an explicit FAIL."""
    def write(name, data):
        try:
            (directory / name).write_text(data, encoding="utf-8")
        except OSError as exc:
            return f"{name}: {type(exc).__name__}: {exc}"
        return None

    lines = ["# W2 unit:runtime", "", f"Result: **{'PASS' if report['passed'] else 'FAIL'}**",
             f"Stage: {report['stage']}; seconds: {report['seconds']:.3f}", "",
             "Fixed official tokenizer assets; encode/decode vs Transformers fast + slow and golden corpus.",
             "NFC normalization is official behavior; equivalence does not mean arbitrary raw-text byte preservation.",
             "No pretrained weights, full-model inference or text generation is executed.", "",
             "| Case | Checks | Passed |", "|---|---:|---|"]
    for row in report["tokenizer_cases"] + report["runtime_cases"]:
        lines.append(f"| {row['name']} | {row.get('checks', 1)} | {row['passed']} |")
    if "error" in report:
        lines.extend(["", "Error: " + report["error"]])
    markdown = "\n".join(lines) + "\n"
    failures = []
    documents = {
        "report.md": markdown,
        "report.html": '<!doctype html><meta charset="utf-8"><title>W2 runtime gate</title><pre>'
        + html.escape(markdown) + "</pre>",
    }
    for name, content in documents.items():
        if error := write(name, content):
            failures.append(error)
    def record_failure():
        report["report_write_errors"] = failures
        if report["passed"]:
            report.update(passed=False, stage="report-write", error="Incomplete runtime gate evidence")

    def recover_views():
        # Preserve the primary probe error. Try both human-readable formats even
        # when one destination is unwritable.
        failure_text = (
            "# W2 unit:runtime\n\nResult: **FAIL**\n\n"
            + f"Stage: {report['stage']}\n\nError: {report['error']}\n\n"
            + "Report write errors:\n" + "\n".join(f"- {failure}" for failure in failures) + "\n"
        )
        recovery = {
            "report.md": failure_text,
            "report.html": '<!doctype html><meta charset="utf-8"><pre>' + html.escape(failure_text) + "</pre>",
        }
        for name, content in recovery.items():
            write(name, content)

    if failures:
        record_failure()
        recover_views()
    # Publish machine-readable status only after the view outcomes are known.
    # In particular, never write PASS and rely on a later corrective write: that
    # second write could fail and leave a misleading successful JSON artifact.
    data = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # A write can complete its payload and still fail while flushing/closing.
    # Keep even a fully written PASS private until write_text has returned;
    # same-directory replacement then publishes the completed evidence.
    pending = directory / ".report.json.tmp"
    error = None
    try:
        pending.write_text(data, encoding="utf-8")
        pending.replace(directory / "report.json")
    except OSError as exc:
        error = f"report.json: {type(exc).__name__}: {exc}"
        try:
            pending.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            error += f"; temporary cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
    if error:
        failures.append(error)
        record_failure()
        recover_views()
    if failures:
        print(f"[unit:runtime] {report['error']}; report write errors: {'; '.join(failures)}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=("verify", "download"), default="verify")
    ap.add_argument("--output-dir", type=Path, default=Path("output/w2-runtime"))
    args = ap.parse_args(argv)
    args.tokenizer_dir = args.tokenizer_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error("--output-dir must be absent or empty; preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = {"gate": "unit:runtime", "passed": False, "stage": "initialize",
              "created_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "model_id": MANIFEST["model_id"], "revision": MANIFEST["revision"],
              "tokenizer_dir": str(args.tokenizer_dir), "tokenizer_cases": [], "runtime_cases": []}
    started = time.perf_counter()
    try:
        report["source_sha256"] = {p: sha256(ROOT / p) for p in (
            "probes/w2_runtime/run.py", "probes/w2_runtime/corpus.json",
            "probes/w1_qwen3_export/manifest.json", "scratchv/runtime/llm_inputs.py",
            "scratchv/runtime/qwen3_tokenizer.py")}
        report["checkout"] = checkout_evidence()
        run_checks(args, report)
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    report["seconds"] = time.perf_counter() - started
    save_report(out, report)
    print(f"[unit:runtime] {'PASS' if report['passed'] else 'FAIL'}; stage={report['stage']}; report={out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
