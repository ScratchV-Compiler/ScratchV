"""Optional memory observations preserve execution and count view owners once."""

import numpy as np
import pytest

from benchmarks.ir_interpreter_cases import CASE_FACTORIES, builder, make_case
from scratchv.ir.types import DataType as D, Value
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def test_default_has_no_stats_and_does_not_observe(monkeypatch):
    import scratchv.verification.ir_interpreter as module

    def unexpected(*args, **kwargs):
        raise AssertionError("default execution must not collect memory")

    monkeypatch.setattr(module, "_MemoryObservation", unexpected)
    b = builder()
    b.ret(b.make_const(2.0))
    assert IRInterpreter(b.program).run({}).memory_stats is None


def test_views_count_logical_bindings_but_not_repeated_storage():
    x = Value("x", shape=(2, 3))
    b = builder(x)
    reshaped = b.reshape(x, (3, 2))
    transposed = b.transpose(reshaped, (1, 0))
    b.ret(transposed)
    source = np.arange(6, dtype="float32").reshape(2, 3)
    interpreter = IRInterpreter(b.program)
    result = interpreter.run({"x": source}, collect_memory_stats=True)
    stats = result.memory_stats
    assert stats["input_logical_bytes"] == 24
    assert stats["initializer_logical_bytes"] == 0
    assert stats["retained_values"] == 3
    assert stats["retained_logical_bytes"] == 72
    assert stats["retained_numpy_storage_bytes"] == 24
    assert stats["return_logical_bytes"] == 24
    assert stats["peak_live_logical_bytes"] == 96
    assert stats["peak_numpy_storage_bytes"] == 48
    assert not np.shares_memory(result.return_value, source)
    result.return_value[:] = 99
    np.testing.assert_array_equal(source, np.arange(6).reshape(2, 3))
    second = interpreter.run({"x": source}, collect_memory_stats=True)
    np.testing.assert_array_equal(second.return_value, source.reshape(3, 2).T)
    assert second.memory_stats == stats


def test_initializer_bytes_and_independent_initializer_copy():
    x, weight = Value("x", shape=(2,)), Value("weight", shape=(2,))
    b = builder(x)
    b.program.global_values.append(weight)
    b.ret(weight)
    # Even aliases supplied by the caller keep the pre-existing separate copies.
    source = np.array([1, 2], dtype="float32")
    result = IRInterpreter(b.program).run({"x": source}, initializers={"weight": source},
                                         collect_memory_stats=True)
    assert result.memory_stats["input_logical_bytes"] == 8
    assert result.memory_stats["initializer_logical_bytes"] == 8
    assert result.memory_stats["retained_numpy_storage_bytes"] == 16
    assert result.memory_stats["peak_numpy_storage_bytes"] == 24
    result.return_value[0] = 123
    assert source[0] == 1


@pytest.mark.parametrize("name", list(CASE_FACTORIES))
def test_observer_preserves_all_direct_cases(name):
    case = make_case(name)
    interpreter = IRInterpreter(case.program)
    baseline = interpreter.run(case.inputs, initializers=case.initializers)
    observed = interpreter.run(case.inputs, initializers=case.initializers, collect_memory_stats=True)
    np.testing.assert_array_equal(observed.return_value, baseline.return_value)
    assert observed.executed_steps == baseline.executed_steps
    assert observed.diagnostics == baseline.diagnostics
    assert observed.memory_stats["retained_values"] >= len(case.inputs)
    assert observed.memory_stats["peak_numpy_storage_bytes"] >= observed.memory_stats["retained_numpy_storage_bytes"]


@pytest.mark.parametrize("flag", [0, 1])
def test_only_executed_branch_contributes_memory(flag):
    condition = Value("condition", D.INT32)
    x = Value("x", shape=(2,))
    b = builder(condition, x)
    b.br_if(condition, "left", "right")
    b.new_block("left")
    b.ret(b.neg(x))
    b.new_block("right")
    b.ret(x)
    result = IRInterpreter(b.program).run({"condition": np.array(flag, dtype="int32"),
                                           "x": np.array([2, 3], dtype="float32")},
                                          collect_memory_stats=True)
    assert result.memory_stats["retained_values"] == 2 + flag
    assert result.memory_stats["retained_logical_bytes"] == 12 + 8 * flag


def test_loop_counts_retained_slots_and_overwrites_not_cumulative_allocations():
    def run(end):
        b = builder()
        slot = b.alloca(16, D.INT32)
        b.store(slot, b.make_const(0, D.INT32))
        i = b.for_loop(0, end)
        b.store(slot, b.add(b.load(slot), i))
        b.endfor()
        b.ret(b.load(slot))
        return IRInterpreter(b.program).run({}, collect_memory_stats=True)

    short, long = run(3), run(30)
    assert short.return_value == 3
    assert long.return_value == 435
    assert short.memory_stats == long.memory_stats
    assert short.memory_stats["retained_logical_bytes"] >= 16
    assert long.executed_steps > short.executed_steps


def test_void_and_empty_arrays_are_counted():
    x = Value("x", shape=(0,))
    b = builder(x)
    b.ret()
    result = IRInterpreter(b.program).run({"x": np.empty(0, dtype="float32")},
                                         collect_memory_stats=True)
    assert result.return_value is None
    assert result.memory_stats["retained_values"] == 1
    assert result.memory_stats["peak_live_logical_bytes"] == 0
    assert result.memory_stats["peak_numpy_storage_bytes"] == 0


@pytest.mark.parametrize("bad", [0, 1, None, "yes"])
def test_invalid_observation_options(bad):
    b = builder()
    b.ret()
    with pytest.raises(IRExecutionError, match="collect_memory_stats must be boolean"):
        IRInterpreter(b.program).run({}, collect_memory_stats=bad)
