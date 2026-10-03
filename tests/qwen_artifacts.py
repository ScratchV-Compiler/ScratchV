"""Validate explicitly selected, current-source Qwen integration inputs."""
from pathlib import Path

import numpy as np


def load_qwen_artifact_inputs(directory):
    from probes.w2_qwen3_small.riscv import validated_model_artifacts
    from probes.w2_qwen3_small.run import arrays_sha256, input_cases, provenance

    directory = Path(directory).resolve()
    files, schema, evidence = validated_model_artifacts(directory)
    current = provenance()
    exported = evidence.get("provenance") or {}
    if exported.get("source_sha256") != current["source_sha256"]:
        raise ValueError("Qwen artifacts were not exported by the current source; regenerate them")
    expected = input_cases()
    paths = {path.name for path in directory.glob("inputs_*.npz")}
    if paths != {f"inputs_{name}.npz" for name, _, _ in expected}:
        raise ValueError("Qwen artifacts require exactly the seven named input files")
    feeds = {}
    for name, _, expected_feed in expected:
        with np.load(directory / f"inputs_{name}.npz", allow_pickle=False) as archive:
            feed = dict(archive)
        if arrays_sha256(feed) != arrays_sha256(expected_feed):
            raise ValueError(f"Qwen input file disagrees with the export report: {name}")
        feeds[name] = feed
    evidence["current_provenance"] = current
    return files["normal"], feeds, evidence
