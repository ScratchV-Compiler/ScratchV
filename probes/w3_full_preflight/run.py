"""Verify fixed full-model assets and size accounting without executing full IR."""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
from probes.w1_qwen3_export.run import MANIFEST, verify_files, inspect_model, tensors
from probes.w3_common import new_output_dir, source_evidence, write_reports, process_peak_rss_bytes

def tensor_size(tensor):
    return math.prod(tensor.dims) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize

def value_size(value):
    kind = value.type.tensor_type
    if not kind.HasField("shape"):
        return None
    dimensions = []
    for dim in kind.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dimensions.append(dim.dim_value)
    return math.prod(dimensions) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(kind.elem_type)).itemsize

def size_accounting(model):
    names = {}
    unknown = []
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        count = value_size(value)
        if count is None:
            unknown.append(value.name)
        else:
            if value.name in names and names[value.name] != count:
                raise ValueError(f"Conflicting shape information: {value.name}")
            names[value.name] = count
    initializer_bytes = sum(tensor_size(t) for t in model.graph.initializer)
    all_tensor_bytes = sum(tensor_size(t) for t in tensors(model))
    graph_names = {v.name for v in model.graph.input} | {t.name for t in model.graph.initializer}
    graph_names.update(name for node in model.graph.node for name in node.output if name)
    known = set(names) | {t.name for t in model.graph.initializer}
    return {
        "initializer_logical_bytes": initializer_bytes,
        "all_embedded_and_external_tensor_logical_bytes": all_tensor_bytes,
        "sum_named_value_logical_bytes": sum(names.values()),
        "named_values_with_static_size": len(names), "unknown_size_values": unknown,
        "graph_value_count": len(graph_names),
        "values_without_persisted_static_size": sorted(graph_names - known),
        "output_logical_bytes": {v.name:value_size(v) for v in model.graph.output},
        "largest_named_values": sorted(names.items(), key=lambda x:(-x[1],x[0]))[:10],
        "scope": "Static ONNX logical sizes, including aliases. Not measured IR live storage or a process RAM requirement.",
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = new_output_dir(args.output_dir)
    report = {
        "gate":"preflight:w3-full-assets", "status":"FAIL", "passed":False, "stage":"source_evidence",
        "full_ir_executed":False, "w3_exit_accepted":False,
        "scope":"Asset integrity and static size preparation only; no ORT/IR/full-model numerical acceptance.",
        "required_before_full_acceptance":[
            "W1 human acceptance and interface/risk decisions",
            "Medium and pretrained subgraph numeric/memory baselines",
            "Measured host capacity and bounded full IR/ORT runner",
            "Full IR vs ORT <1e-4, three independent reproductions and successful Nightly",
        ],
    }
    start = time.perf_counter()
    try:
        report.update(source_evidence())
        report["stage"] = "asset_hashes"
        files = verify_files(args.model_dir)
        report["files"] = files
        report["source_revision"] = MANIFEST["revision"]
        report["stage"] = "onnx_contract"
        report["onnx_contract"] = inspect_model(args.model_dir, files)
        report["stage"] = "static_size_accounting"
        model = onnx.load(str(args.model_dir / "model.onnx"), load_external_data=False)
        report["node_count"] = len(model.graph.node)
        report["operator_counts"] = dict(Counter(node.op_type for node in model.graph.node))
        report["sizes"] = size_accounting(model)
        report["weights_file_bytes"] = sum(item["bytes"] for item in files if item["name"] != "model.onnx")
        report.update(passed=True, status="PASS", stage="complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(elapsed_seconds=time.perf_counter()-start,
                  process_peak_rss_bytes=process_peak_rss_bytes(),
                  peak_scope="This metadata-only preflight process, not full parser/IR execution")
    write_reports(out,report)
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
