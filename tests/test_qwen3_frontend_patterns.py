"""unit:frontend-ops: four Qwen3 decomposed patterns, before/after optimization."""
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest

from probes.w2_backend_ops.cases import build_cases
from probes.w2_backend_ops.run import ATOL, FRONTEND_FAMILIES, interpret, reference_case
from probes.w2_qwen3_small.diagnostics import tensor_diff

CASES = [case for case in build_cases() if case.family in FRONTEND_FAMILIES]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("level", ["none", "all"])
def test_four_qwen3_patterns_match_independent_formula_and_ort(tmp_path, case, level):
    expected, parser, program = reference_case(case, tmp_path / "pattern.onnx")
    assert tensor_diff(expected, case.expected, ATOL)["passed"]
    actual = interpret(program, parser.initializers, case.feed, level)
    assert tensor_diff(actual, expected, ATOL)["passed"]
    assert tensor_diff(actual, case.expected, ATOL)["passed"]


@pytest.mark.parametrize("family", FRONTEND_FAMILIES)
def test_pattern_cases_detect_semantic_mutations(tmp_path, family):
    case = next(copy.deepcopy(case) for case in CASES if case.family == family)
    graph = case.model.graph
    if family == "rmsnorm":
        epsilon = next(value for value in graph.initializer if value.name == "epsilon")
        epsilon.CopyFrom(numpy_helper.from_array(np.array(1e-3, np.float32), "epsilon"))
    elif family == "rope":
        # Wrong rotate_half sign; rotation still preserves shape/dtype.
        next(node for node in graph.node if node.op_type == "Neg").op_type = "Identity"
    elif family == "swiglu":
        next(node for node in graph.node if node.op_type == "Sigmoid").input[0] = "up"
    else:
        # Wrong interleaved KV repetition still has the expected head count.
        graph.initializer.append(numpy_helper.from_array(np.array([0, 2, 1, 3], np.int64), "wrong_heads"))
        for name in ("k", "v"):
            nodes = list(graph.node)
            position = next(index for index, node in enumerate(nodes) if f"{name}_repeated" in node.output)
            nodes.insert(position + 1, helper.make_node("Gather", [f"{name}_repeated", "wrong_heads"],
                                                       [f"{name}_wrong"], axis=1))
            for node in nodes[position + 2:]:
                for index, value in enumerate(node.input):
                    if value == f"{name}_repeated":
                        node.input[index] = f"{name}_wrong"
            del graph.node[:]
            graph.node.extend(nodes)
    onnx.checker.check_model(case.model)
    actual, parser, program = reference_case(case, tmp_path / "mutated.onnx")
    ir = interpret(program, parser.initializers, case.feed, "all")
    assert tensor_diff(actual, ir, ATOL)["passed"]
    assert not tensor_diff(actual, case.expected, ATOL)["passed"], family
