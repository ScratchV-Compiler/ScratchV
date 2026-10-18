"""Shared seeded two-layer Qwen3 exports and cached real executions."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest

from probes.w2_qwen3_small.diagnostics import build_diagnostic_model, unpack_trace
from probes.w2_qwen3_small.run import input_cases


@pytest.fixture(scope="session")
def qwen3_small_inputs() -> dict:
    return {name: (length, feed) for name, length, feed in input_cases()}


@pytest.fixture(scope="session")
def qwen3_small_model(torch_cpu):
    from probes.w2_qwen3_small.model import build_model

    return build_model(seed=0)


@pytest.fixture(scope="session")
def qwen3_small_torch(qwen3_small_model, qwen3_small_inputs, torch_cpu):
    cache = {}

    def reference(name: str) -> tuple[dict, np.ndarray]:
        if name not in cache:
            _, feed = qwen3_small_inputs[name]
            args = tuple(torch_cpu.from_numpy(feed[key]) for key in ("input_ids", "attention_mask"))
            with torch_cpu.inference_mode(), qwen3_small_model.capture():
                values = qwen3_small_model(*args)
                named = {key: value.numpy().copy()
                         for key, value in zip(qwen3_small_model.output_names, values)}
            with torch_cpu.inference_mode():
                plain = qwen3_small_model.model(
                    input_ids=args[0], attention_mask=args[1],
                    position_ids=torch_cpu.arange(256).reshape(1, 256),
                    use_cache=False, return_dict=False, logits_to_keep=0,
                )[0].numpy().copy()
            cache[name] = named, plain
        return cache[name]

    return reference


@pytest.fixture(scope="session")
def qwen3_small_artifacts(tmp_path_factory, qwen3_small_model, qwen3_small_inputs,
                          qwen3_small_torch, torch_cpu) -> dict:
    from probes.w2_qwen3_small.model import export_onnx

    directory = tmp_path_factory.mktemp("qwen3-two-layer")
    _, feed = qwen3_small_inputs["full_seed_0"]
    args = tuple(torch_cpu.from_numpy(feed[key]) for key in ("input_ids", "attention_mask"))
    export_onnx(qwen3_small_model, directory / "checkpoints.onnx", *args)
    exported = onnx.load(directory / "checkpoints.onnx")
    normal = copy.deepcopy(exported)
    logits = next(copy.deepcopy(value) for value in normal.graph.output if value.name == "logits")
    del normal.graph.output[:]
    normal.graph.output.append(logits)
    diagnostic, schema = build_diagnostic_model(exported, qwen3_small_torch("full_seed_0")[0])
    assert len(schema) == 29
    assert tuple(entry["name"] for entry in schema) == qwen3_small_model.output_names
    paths = {"normal": directory / "model.onnx", "diagnostic": directory / "diagnostics.onnx"}
    for kind, model in (("normal", normal), ("diagnostic", diagnostic)):
        onnx.checker.check_model(model)
        onnx.save(model, paths[kind])
    return {"paths": paths, "schema": schema}


@pytest.fixture(scope="session")
def qwen3_two_layer_path(qwen3_small_artifacts) -> Path:
    return qwen3_small_artifacts["paths"]["normal"]


@pytest.fixture(scope="session")
def qwen3_diagnostic_path(qwen3_small_artifacts) -> Path:
    return qwen3_small_artifacts["paths"]["diagnostic"]


@pytest.fixture(scope="session")
def qwen3_small_ort(qwen3_small_artifacts, qwen3_two_layer_path, qwen3_diagnostic_path,
                    qwen3_small_inputs, ort_session):
    sessions = {"normal": ort_session(qwen3_two_layer_path),
                "diagnostic": ort_session(qwen3_diagnostic_path)}
    cache = {}

    def reference(name: str) -> dict:
        if name not in cache:
            _, feed = qwen3_small_inputs[name]
            normal = sessions["normal"].run(None, feed)[0]
            values = sessions["diagnostic"].run(None, feed)
            names = [value.name for value in sessions["diagnostic"].get_outputs()]
            cache[name] = {"normal": normal, "diagnostic": values[0],
                           "named": dict(zip(names[1:], values[1:]))}
        return cache[name]

    return reference


@pytest.fixture(scope="session")
def qwen3_small_result(qwen3_small_artifacts, qwen3_small_inputs, run_ir, request):
    cache = {}

    def execute(name: str, kind: str, engine: str, level: str):
        key = name, kind, engine, level
        if key not in cache:
            path = qwen3_small_artifacts["paths"][kind]
            _, feed = qwen3_small_inputs[name]
            if engine == "ir":
                output, seconds = run_ir(path, feed, level), None
            else:
                assert engine == "qemu"
                # Discover QEMU only when a QEMU test is actually selected.
                result = request.getfixturevalue("run_riscv")(path, feed, level)
                output, seconds = result.output, result.elapsed_s
            named = (unpack_trace(output, qwen3_small_artifacts["schema"])
                     if kind == "diagnostic" else {"logits": output})
            cache[key] = output, named, seconds
        return cache[key]

    return execute
