"""Return metadata is an ABI contract, not an alternative buffer interpretation."""
import pytest
import numpy as np

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def program_for(value, returned=None, declarations=()):
    builder = IRBuilder()
    function = builder.new_function("main", [value])
    builder.new_block("entry")
    function.returns = list(declarations)
    builder.ret(value if returned is None else returned)
    return builder.program


@pytest.mark.parametrize("storage", ["inline", "external"])
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.INT64])
def test_return_reference_cannot_reinterpret_defined_dtype(storage, dtype):
    value = Value("x", DataType.FLOAT32, shape=(2,))
    returned = Value("x", dtype, shape=(2,))
    with pytest.raises(TensorCCodegenError, match="RETURN.*dtype"):
        TensorCCodegen(program_for(value, returned), constant_storage=storage).generate()


@pytest.mark.parametrize("storage", ["inline", "external"])
@pytest.mark.parametrize("shape", [(3,), (1, 2), (), (True,), ("N",)])
def test_return_reference_checks_known_input_shape(storage, shape):
    value = Value("x", shape=(2,))
    returned = Value("x", shape=shape)
    with pytest.raises(TensorCCodegenError, match="RETURN.*shape|static"):
        TensorCCodegen(program_for(value, returned), constant_storage=storage).generate()


@pytest.mark.parametrize("storage", ["inline", "external"])
@pytest.mark.parametrize("declarations", [
    [Value("result", DataType.INT64, shape=(2,))],
    [Value("result", shape=())],
    [Value("result", shape=(1, 2))],
    [Value("result", shape=(True,))],
    [Value("result", shape=(2,)), Value("other", shape=(2,))],
])
def test_function_return_signature_cannot_disagree_with_actual_output(storage, declarations):
    value = Value("x", shape=(2,))
    with pytest.raises(TensorCCodegenError, match="return|RETURN|static"):
        TensorCCodegen(program_for(value, declarations=declarations), constant_storage=storage).generate()


@pytest.mark.parametrize("storage", ["inline", "external"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_explicit_return_signature_accepts_scalar_empty_and_tensor(storage, shape):
    value = Value("x", shape=shape)
    declaration = Value("public_result", shape=shape)
    artifact = TensorCCodegen(program_for(value, declarations=[declaration]), constant_storage=storage).generate()
    assert artifact.output.name == "x"
    assert artifact.output.shape == shape


@pytest.mark.parametrize("storage", ["inline", "external"])
def test_builder_intermediate_return_keeps_inferred_shape(storage):
    builder = IRBuilder()
    builder.new_function("main", [Value("x", shape=(2, 3))])
    builder.new_block("entry")
    result = builder.neg(builder.program.functions[0].params[0])
    assert result.shape == ()  # Builder default, not an explicit signature.
    builder.ret(result)
    artifact = TensorCCodegen(builder.program, constant_storage=storage).generate()
    assert artifact.output.shape == (2, 3)


@pytest.mark.parametrize("storage", ["inline", "external"])
def test_inferred_intermediate_return_rejects_explicit_wrong_shape(storage):
    builder = IRBuilder()
    builder.new_function("main", [Value("x", shape=(2, 3))])
    builder.new_block("entry")
    result = builder.neg(builder.program.functions[0].params[0])
    builder.ret(Value(result.name, shape=(3, 2)))
    with pytest.raises(TensorCCodegenError, match="RETURN.*shape"):
        TensorCCodegen(builder.program, constant_storage=storage).generate()


@pytest.mark.parametrize("shape", [None, 0, False, ""])
@pytest.mark.parametrize("consumer", ["inline", "external", "interpreter"])
def test_only_empty_sequence_is_an_inferred_intermediate_shape(shape, consumer):
    builder = IRBuilder()
    x = Value("x", shape=(2,))
    builder.new_function("main", [x])
    builder.new_block("entry")
    result = builder.neg(x)
    builder.ret(Value(result.name, shape=shape))
    if consumer == "interpreter":
        with pytest.raises(IRExecutionError, match="ShapeError"):
            IRInterpreter(builder.program).run({"x": np.ones(2, dtype=np.float32)})
    else:
        with pytest.raises(TensorCCodegenError, match="static"):
            TensorCCodegen(builder.program, constant_storage=consumer).generate()
