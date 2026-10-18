"""Four decomposed Qwen3 patterns and their original semantic mutations."""
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest

from probes.w2_backend_ops.cases import build_cases
from probes.w2_backend_ops.run import FRONTEND_FAMILIES
from probes.w2_qwen3_small.diagnostics import tensor_diff

CASES = [case for case in build_cases() if case.family in FRONTEND_FAMILIES]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("level", ["none", "all"])
def test_frontend_patterns(case, level, tmp_path, ort_session, run_ir, assert_tensor):
    path = tmp_path / "pattern.onnx"
    onnx.save(case.model, path)
    reference = ort_session(path).run(None, case.feed)[0]
    assert_tensor(reference, case.expected)
    actual = run_ir(path, case.feed, level)
    assert_tensor(actual, reference)
    assert_tensor(actual, case.expected)


@pytest.mark.parametrize("family", FRONTEND_FAMILIES)
def test_patterns_detect_wrong_math(family, tmp_path, ort_session, run_ir, assert_tensor):
    case = next(copy.deepcopy(case) for case in CASES if case.family == family)
    graph = case.model.graph
    if family == "rmsnorm":
        epsilon = next(value for value in graph.initializer if value.name == "epsilon")
        epsilon.CopyFrom(numpy_helper.from_array(np.array(1e-3, np.float32), "epsilon"))
    elif family == "rope":
        next(node for node in graph.node if node.op_type == "Neg").op_type = "Identity"
    elif family == "swiglu":
        next(node for node in graph.node if node.op_type == "Sigmoid").input[0] = "up"
    else:
        graph.initializer.append(numpy_helper.from_array(np.array([0, 2, 1, 3], np.int64), "wrong_heads"))
        for name in ("k", "v"):
            nodes = list(graph.node)
            position = next(i for i, node in enumerate(nodes) if f"{name}_repeated" in node.output)
            nodes.insert(position + 1, helper.make_node(
                "Gather", [f"{name}_repeated", "wrong_heads"], [f"{name}_wrong"], axis=1,
            ))
            for node in nodes[position + 2:]:
                for index, value in enumerate(node.input):
                    if value == f"{name}_repeated":
                        node.input[index] = f"{name}_wrong"
            del graph.node[:]
            graph.node.extend(nodes)
    onnx.checker.check_model(case.model)
    path = tmp_path / "mutated.onnx"
    onnx.save(case.model, path)
    actual = ort_session(path).run(None, case.feed)[0]
    assert_tensor(run_ir(path, case.feed, "all"), actual)
    assert not tensor_diff(actual, case.expected)["passed"], family
