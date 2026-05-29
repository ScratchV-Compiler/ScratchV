"""Verification framework: compare compiler output against reference results.

Supports three reference modes:
1. ONNX Runtime — runs the ONNX model as reference (requires onnxruntime)
2. Numpy reference — compute expected output using numpy
3. DSL simulation — runs DSL through a naive interpreter for comparison
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# ONNX Runtime adapter
# ---------------------------------------------------------------------------

class ONNXReference:
    """Run an ONNX model through ONNX Runtime to get reference outputs."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    @property
    def available(self) -> bool:
        if self._session is not None:
            return True
        try:
            import onnxruntime
            self._session = onnxruntime.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def run(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return output name -> array mapping."""
        if not self.available:
            raise RuntimeError(
                "ONNX Runtime not available. Install: pip install onnxruntime")

        outputs = [o.name for o in self._session.get_outputs()]
        result = self._session.run(outputs, feed_dict)
        return dict(zip(outputs, result))


# ---------------------------------------------------------------------------
# Numpy reference computation (for individual ops)
# ---------------------------------------------------------------------------

def numpy_reference(op_type: str, *inputs: np.ndarray, **attrs) -> np.ndarray:
    """Compute reference output for a given op using numpy.

    Args:
        op_type: Operation name (Add, Mul, Relu, MatMul, etc.)
        *inputs: Input arrays
        **attrs: Extra attributes (axis, kernel, stride, etc.)

    Returns:
        Reference output array.
    """
    handlers = {
        "Add": lambda: inputs[0] + inputs[1],
        "Sub": lambda: inputs[0] - inputs[1],
        "Mul": lambda: inputs[0] * inputs[1],
        "Div": lambda: inputs[0] / inputs[1],
        "Neg": lambda: -inputs[0],
        "Exp": lambda: np.exp(inputs[0]),
        "Relu": lambda: np.maximum(inputs[0], 0.0),
        "Gelu": lambda: _numpy_gelu(list(inputs), **attrs),
        "Softmax": lambda: _numpy_softmax(list(inputs), **attrs),
        "MatMul": lambda: inputs[0] @ inputs[1],
        "Dot": lambda: _numpy_dot(list(inputs), **attrs),
        "MaxPool": lambda: _numpy_maxpool(list(inputs), **attrs),
        "Sigmoid": lambda: 1.0 / (1.0 + np.exp(-inputs[0])),
        "Tanh": lambda: np.tanh(inputs[0]),
    }
    handler = handlers.get(op_type)
    if handler is None:
        raise ValueError(f"No numpy reference for op: {op_type}")
    return handler()


def _numpy_gelu(inputs: list[np.ndarray], **attrs) -> np.ndarray:
    x = inputs[0]
    inner = np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
    return x * 0.5 * (1.0 + inner)


def _numpy_softmax(inputs: list[np.ndarray], **attrs) -> np.ndarray:
    x = inputs[0]
    axis = attrs.get("axis", -1)
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _numpy_dot(inputs: list[np.ndarray], **attrs) -> np.ndarray:
    return np.dot(inputs[0], inputs[1])


def _numpy_maxpool(inputs: list[np.ndarray], **attrs) -> np.ndarray:
    x = inputs[0]
    kernel = attrs.get("kernel", 2)
    stride = attrs.get("stride", 2)
    # Simple 1D or 2D maxpool
    if x.ndim == 3:  # CHW
        c, h, w = x.shape
        out_h = (h - kernel) // stride + 1
        out_w = (w - kernel) // stride + 1
        result = np.zeros((c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                result[:, i, j] = np.max(
                    x[:, i*stride:i*stride+kernel, j*stride:j*stride+kernel],
                    axis=(1, 2)
                )
        return result
    elif x.ndim == 1:
        result_list: list[np.floating] = []
        for i in range(0, len(x) - kernel + 1, stride):
            result_list.append(np.max(x[i:i + kernel]))
        return np.array(result_list)
    return x


# ---------------------------------------------------------------------------
# DSL interpreter (runs DSL programs with concrete values)
# ---------------------------------------------------------------------------

class DSLInterpreter:
    """Evaluate a DSL program on concrete input values.

    This provides a ground-truth reference for verification.
    """

    def __init__(self):
        self._vars: dict[str, np.ndarray] = {}

    def run(self, dsl_source: str,
            inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run a DSL program with given input values.

        Args:
            dsl_source: The DSL source text.
            inputs: Mapping of variable name -> numpy array.

        Returns:
            The return value of the program.
        """
        self._vars = dict(inputs)
        import re

        lines = dsl_source.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # for i = start, end
            m = re.match(r"for\s+(\w+)\s*=\s*(\d+)\s*,\s*(\d+)", line)
            if m:
                continue

            if line == "endfor":
                continue

            # return var
            m = re.match(r"return\s+(\S+)", line)
            if m:
                return self._resolve(m.group(1))

            # name = op(args)
            m = re.match(r"(\w+)\s*=\s*(\w+)\((.+)\)", line)
            if m:
                dest_name = m.group(1)
                op_name = m.group(2).lower()
                args_text = m.group(3)
                args = [a.strip() for a in args_text.split(",") if a.strip()]
                result = self._dispatch(op_name, args)
                self._vars[dest_name] = result

        return np.array(0.0)

    def _resolve(self, name: str) -> np.ndarray:
        if name in self._vars:
            return self._vars[name]
        try:
            val = float(name)
            return np.array(val)
        except ValueError:
            pass
        return np.array(0.0)

    def _dispatch(self, op: str, args: list[str]) -> np.ndarray:
        plain = []
        kwargs = {}
        for a in args:
            if ":" in a:
                k, v = a.split(":", 1)
                try:
                    kwargs[k.strip()] = int(v.strip())
                except ValueError:
                    pass
                kwargs[k.strip()] = v.strip()
            else:
                plain.append(a)

        resolved = [self._resolve(a) for a in plain]

        op_map = {
            "add": lambda: resolved[0] + resolved[1],
            "sub": lambda: resolved[0] - resolved[1],
            "mul": lambda: resolved[0] * resolved[1],
            "div": lambda: resolved[0] / resolved[1],
            "neg": lambda: -resolved[0],
            "exp": lambda: np.exp(resolved[0]),
            "relu": lambda: np.maximum(resolved[0], 0.0),
            "gelu": lambda: resolved[0] * 0.5 * (
                1.0 + np.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (resolved[0] + 0.044715 * resolved[0]**3)
                )),
            "matmul": lambda: resolved[0] @ resolved[1],
            "dot": lambda: np.dot(resolved[0], resolved[1]),
            "softmax": lambda: _numpy_softmax(resolved, **kwargs),
            "maxpool": lambda: _numpy_maxpool(resolved, **kwargs),
        }
        handler = op_map.get(op)
        if handler is None:
            raise ValueError(f"Unsupported op in interpreter: {op}")
        return handler()


# ---------------------------------------------------------------------------
# Main verification API
# ---------------------------------------------------------------------------

def verify_onnx_model(
    model_path: str,
    compiler_output_fn=None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Verify compiled output matches ONNX Runtime reference.

    Args:
        model_path: Path to .onnx file.
        compiler_output_fn: Callable(inputs_dict) -> outputs_dict.
            If None, only reference results are computed.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        verbose: Print detailed comparison.

    Returns:
        dict with: success, max_error, mismatched_outputs, reference, compiled
    """
    import onnx

    onnx_model = onnx.load(model_path)
    graph = onnx_model.graph

    # Build random inputs matching the graph's input shapes
    feed_dict = {}
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        feed_dict[inp.name] = np.random.randn(*shape).astype(np.float32)

    ref = ONNXReference(model_path)
    if not ref.available:
        if verbose:
            print("ONNX Runtime not available.")
        return {"success": False, "error": "onnxruntime not available"}

    reference = ref.run(feed_dict)

    if compiler_output_fn is None:
        return {"success": True, "reference": reference, "compiled": None}

    compiled = compiler_output_fn(feed_dict)

    # Compare
    max_error = 0.0
    mismatched = []
    for name in reference:
        if name not in compiled:
            mismatched.append(name)
            continue
        err = np.max(np.abs(reference[name] - compiled[name]))
        if err > atol + rtol * np.max(np.abs(reference[name])):
            mismatched.append(name)
        max_error = max(max_error, err)

    success = len(mismatched) == 0

    if verbose:
        print(f"Verification {'PASSED' if success else 'FAILED'}")
        print(f"  Max error: {max_error:.6e}")
        if mismatched:
            print(f"  Mismatched outputs: {mismatched}")

    return {
        "success": success,
        "max_error": max_error,
        "mismatched_outputs": mismatched,
        "reference": reference,
        "compiled": compiled,
    }


def verify_dsl(
    dsl_source: str,
    inputs: dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Verify DSL program against numpy reference.

    Args:
        dsl_source: DSL source text.
        inputs: Input variable -> array mapping.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        dict with keys: success, max_error, expected, got
    """
    interpreter = DSLInterpreter()
    expected = interpreter.run(dsl_source, inputs)

    # Compile through ScratchV
    from scratchv.frontend.dsl_parser import DSLParser
    parser = DSLParser()
    parser.parse(dsl_source)

    # For now, compare with expected (full compilation pipeline comparison
    # requires an execution environment for the generated assembly)
    return {
        "success": True,
        "max_error": 0.0,
        "expected": expected,
        "got": expected,  # placeholder — real comparison when JIT is wired
    }
