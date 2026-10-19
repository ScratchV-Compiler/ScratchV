"""Sixteen independent formulas, two optimization levels, IR and QEMU."""
import onnx
import pytest

from probes.w2_backend_ops.cases import build_cases
from probes.w2_backend_ops.run import FAMILIES

CASES = build_cases()


def test_backend_case_coverage():
    assert len(CASES) == len({case.name for case in CASES}) == 16
    assert {case.family for case in CASES} == set(FAMILIES)


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.name)
def operator_case(request, tmp_path_factory, ort_session):
    case = request.param
    path = tmp_path_factory.mktemp(case.name) / "model.onnx"
    onnx.save(case.model, path)
    return case, path, ort_session(path).run(None, case.feed)[0]


@pytest.mark.parametrize("level", ["none", "all"])
def test_backend_ir(operator_case, level, run_ir, assert_tensor):
    case, path, reference = operator_case
    assert_tensor(reference, case.expected)
    actual = run_ir(path, case.feed, level)
    assert_tensor(actual, reference)
    assert_tensor(actual, case.expected)


@pytest.mark.parametrize("level", ["none", "all"])
def test_backend_riscv(operator_case, level, run_riscv, assert_tensor, record_property):
    case, path, reference = operator_case
    assert_tensor(reference, case.expected)
    result = run_riscv(path, case.feed, level)
    record_property("qemu_process_wall_seconds", result.elapsed_s)
    assert_tensor(result.output, reference)
    assert_tensor(result.output, case.expected)
