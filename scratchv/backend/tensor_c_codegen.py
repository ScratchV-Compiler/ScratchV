"""Static tensor IR -> portable C, independent of the legacy scalar selector.

The C ABI is ``int scratchv_run(const void *const inputs[], void *output)``.
Buffers are contiguous, native-endian and suitably aligned; output must not
overlap an input buffer. The caller owns
input/output storage; intermediates use a bounded static arena with SSA-lifetime
reuse. Consequently generated functions are not reentrant. Compile as C11 with
``-fno-fast-math -ffp-contract=off -fno-strict-aliasing`` and link a conforming
libm plus memcpy/memset. No tensor storage is placed on the stack.

``constant_storage="external"`` instead emits ``scratchv_run_external(inputs,
weights, workspace, workspace_bytes, output)``. Its caller supplies read-only
weight buffers in artifact order and an 8-byte-aligned workspace. There is no
mutable static storage. Input/output/weight buffer sizes are a caller contract:
the C ABI has no descriptors to discover allocation lengths. The loader must
validate them against TensorSpec before calling the function. Pointers, numeric
values, workspace capacity and writable-buffer overlap are checked in C.

Supported element types are FP32, INT32 and INT64. Control flow, dynamic shapes,
empty reductions and nonfinite arithmetic are rejected explicitly. This backend
does not silently lower tensor operations to scalar integer instructions.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, replace
import math
from typing import Mapping

import numpy as np

from scratchv.ir.types import DataType, OpCode, Program, Value
from scratchv.verification.ir_numpy_ops import OpError, check_instruction


class TensorCCodegenError(ValueError):
    """A graph cannot be represented by the static C tensor contract."""


_TYPES = {DataType.FLOAT32: ("float", np.dtype("float32")),
          DataType.INT32: ("int32_t", np.dtype("int32")),
          DataType.INT64: ("int64_t", np.dtype("int64"))}
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: DataType
    shape: tuple[int, ...]

    @property
    def numpy_dtype(self):
        return _TYPES[self.dtype][1]

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * self.numpy_dtype.itemsize


@dataclass(frozen=True)
class TensorCArtifact:
    source: str
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    workspace_bytes: int
    constant_bytes: int
    function_name: str = "scratchv_run"
    compile_flags: tuple[str, ...] = ("-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing")
    constant_storage: str = "inline"
    external_weights: tuple[TensorSpec, ...] = ()
    external_initializers: tuple[np.ndarray, ...] = field(default=(), repr=False, compare=False)
    kernel_calls: bool = False
    matmul_policy: str = "sequential"
    matmul_tile: str = "1x4"
    matmul_rhs_layout: str = "original"
    matmul_rhs_panels: tuple[dict[str, object], ...] = ()
    qemu_workspace_layout: str = "none"
    qemu_workspace_buffers: tuple[tuple[int, int], ...] = ()


def _shape(shape):
    if not isinstance(shape, (tuple, list)) or any(
        isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 0
        for d in shape
    ):
        raise TensorCCodegenError(f"Expected static nonnegative dimensions, got {shape!r}")
    result = tuple(int(d) for d in shape)
    if len(result) > 16 or math.prod(result) > 2**31 - 1:
        raise TensorCCodegenError("Tensor rank/element count exceeds static backend limit")
    return result


def _strides(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


def _axis(axis, rank):
    if not -rank <= axis < rank:
        raise TensorCCodegenError(f"axis {axis} outside rank {rank}")
    return axis % rank


def _axes(axes, rank):
    normalized = tuple(_axis(a, rank) for a in axes)
    if len(set(normalized)) != len(normalized):
        raise TensorCCodegenError("Duplicate axes")
    return normalized


def _broadcast(*shapes):
    try:
        return _shape(np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise TensorCCodegenError(f"Invalid broadcast: {shapes}") from exc


def _offset(shape, coefficients, index="i", base=0):
    """Flat-index coordinates dotted with signed input strides."""
    terms = [str(base)] if base else []
    for dim, stride, coefficient in zip(shape, _strides(shape), coefficients):
        if dim > 1 and coefficient:
            coordinate = f"(({index} / {stride}ULL) % {dim}ULL)"
            terms.append(f"((int64_t){coordinate} * ({coefficient}LL))")
    return " + ".join(terms) or "0"


def _broadcast_offset(output, source, index="i"):
    if source == output:
        return index
    padded = (1,) * (len(output) - len(source)) + source
    strides = (0,) * (len(output) - len(source)) + _strides(source)
    return _offset(output, [s if d != 1 else 0 for d, s in zip(padded, strides)], index)


def _literal(value, dtype):
    if dtype == DataType.FLOAT32:
        value = np.float32(value)
        if not np.isfinite(value):
            raise TensorCCodegenError("Nonfinite initializer/constant")
        return float(value).hex() + "f"
    integer = int(value)
    bounds = np.iinfo(_TYPES[dtype][1])
    if integer != value or not bounds.min <= integer <= bounds.max:
        raise TensorCCodegenError("Integer constant outside declared dtype")
    if integer == -(1 << 63):
        return "(-9223372036854775807LL - 1LL)"
    return f"{integer}LL"


_PRELUDE = r'''/* Generated static FP32 tensor graph. Inputs/output must not alias workspace. */
#include <stdint.h>
#include <stddef.h>
extern float expf(float);
extern float sinf(float);
extern float cosf(float);
extern float sqrtf(float);
extern float powf(float, float);
static int sv_finite(float x) {
    union { float f; uint32_t u; } v; v.f=x;
    return (v.u & 0x7f800000U) != 0x7f800000U;
}
/* status: 1 invalid pointer, 2 invalid numeric/domain, 3 gather bounds. */
static float sv_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z=expf(x); return z / (1.0f + z);
}
'''

_EXTERNAL_PRELUDE = r'''
/* External storage targets 64-bit hosts/guests; lengths are caller-validated. */
_Static_assert(sizeof(size_t) == 8 && sizeof(uintptr_t) == 8,
               "external tensor ABI requires a 64-bit target");
#if defined(SCRATCHV_HARTS) && SCRATCHV_HARTS > 1
/* Synchronous: all callbacks finish before workspace reuse or validation. */
extern void sv_parallel_for(size_t count,
                            void (*fn)(void *, size_t, size_t), void *context);
#endif
static int sv_buffer(const void *pointer, uint64_t bytes, size_t alignment) {
    uintptr_t address = (uintptr_t)pointer;
    return pointer && address % alignment == 0 && bytes <= UINTPTR_MAX - address;
}
static int sv_overlap(const void *a, uint64_t na, const void *b, uint64_t nb) {
    uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return na && nb && (pa <= pb ? pb - pa < na : pa - pb < nb);
}
static uint32_t sv_float_bits(float x) {
    union { float f; uint32_t u; } value; value.f = x; return value.u;
}
'''


class TensorCCodegen:
    """Generate one straight-line function with strict tensor/dtype validation."""

    def __init__(self, program: Program, initializers: Mapping[str, np.ndarray] | None = None,
                 *, max_workspace_bytes=256 * 1024 * 1024,
                 max_constant_bytes=256 * 1024 * 1024, constant_storage="inline",
                 kernel_calls=False, matmul_policy="sequential", matmul_tile="1x4",
                 qemu_workspace_layout="none", matmul_rhs_layout="original"):
        self.program = program
        self.initializers = {} if initializers is None else dict(initializers)
        self.max_workspace_bytes = max_workspace_bytes
        self.max_constant_bytes = max_constant_bytes
        self.constant_storage = constant_storage
        self.kernel_calls = kernel_calls
        self.matmul_policy = matmul_policy
        self.matmul_tile = matmul_tile
        self.matmul_rhs_layout = matmul_rhs_layout
        self.qemu_workspace_layout = qemu_workspace_layout

    def generate(self) -> TensorCArtifact:
        program, initializers = self.program, self.initializers
        if self.constant_storage not in ("inline", "external"):
            raise TensorCCodegenError("constant_storage must be 'inline' or 'external'")
        if not isinstance(self.kernel_calls, bool):
            raise TensorCCodegenError("kernel_calls must be boolean")
        if self.kernel_calls and self.constant_storage != "external":
            raise TensorCCodegenError("kernel_calls requires external constant storage")
        if self.matmul_policy not in ("sequential", "blocked_fma"):
            raise TensorCCodegenError("matmul_policy must be 'sequential' or 'blocked_fma'")
        if self.matmul_policy != "sequential" and self.constant_storage != "external":
            raise TensorCCodegenError("Experimental MatMul policies require external constant storage")
        if not isinstance(self.matmul_tile, str) or self.matmul_tile not in ("1x4", "2x4", "1x8"):
            raise TensorCCodegenError("matmul_tile must be '1x4', '2x4' or '1x8'")
        if self.matmul_tile != "1x4" and (
                self.constant_storage != "external" or self.matmul_policy != "blocked_fma"):
            raise TensorCCodegenError("Experimental matmul_tile requires external blocked_fma MatMul")
        if (not isinstance(self.matmul_rhs_layout, str)
                or self.matmul_rhs_layout not in ("original", "column-panel")):
            raise TensorCCodegenError("matmul_rhs_layout must be 'original' or 'column-panel'")
        if self.matmul_rhs_layout != "original" and (
                self.constant_storage != "external" or self.matmul_policy != "blocked_fma"):
            raise TensorCCodegenError("column-panel requires external blocked_fma MatMul")
        if (not isinstance(self.qemu_workspace_layout, str)
                or self.qemu_workspace_layout not in ("none", "two-buffer-pad-4k")):
            raise TensorCCodegenError("qemu_workspace_layout must be 'none' or 'two-buffer-pad-4k'")
        if self.qemu_workspace_layout != "none" and self.constant_storage != "external":
            raise TensorCCodegenError("Experimental QEMU workspace layout requires external storage")
        if len(program.functions) != 1:
            raise TensorCCodegenError("Exactly one function is required")
        func = program.functions[0]
        if len(func.blocks) != 1 or func.blocks[0].phi_nodes:
            raise TensorCCodegenError("Only one straight-line block without phi is supported")
        instructions = func.blocks[0].instructions
        if (not instructions or instructions[-1].opcode != OpCode.RETURN
                or len(instructions[-1].operands) != 1
                or any(i.opcode == OpCode.RETURN for i in instructions[:-1])):
            raise TensorCCodegenError("Exactly one final tensor RETURN is required")
        if instructions[-1].attrs or instructions[-1].dest is not None or instructions[-1].target:
            raise TensorCCodegenError("RETURN must not have attributes, destination or target")
        if any(value.is_constant for value in func.params):
            raise TensorCCodegenError("Function parameters cannot be constants")
        for bound in (self.max_workspace_bytes, self.max_constant_bytes):
            if (isinstance(bound, bool) or not isinstance(bound, int)
                    or not 0 <= bound <= _UINT64_MAX):
                raise TensorCCodegenError("Memory limits must be nonnegative 64-bit integers")
        program, initializers, self._rhs_panel_sources = self._panel_inputs(program, initializers)
        func = program.functions[0]
        instructions = func.blocks[0].instructions
        self.matmul_rhs_panels = []
        self.specs, self.names, self.constants, self.body = {}, {}, [], []
        self.external_weights, self.external_initializers = [], []
        self.external_scalar_checks = []
        self.kernels = []
        self._softmax_callbacks = {}
        self.constant_bytes = 0
        self._counter = 0
        defined = set()
        for value in [*program.global_values, *func.params]:
            if value.name in defined:
                raise TensorCCodegenError(f"Duplicate definition {value.name}")
            defined.add(value.name)
            self._register(value)
        expected_initializers = {v.name for v in program.global_values}
        if set(initializers) - expected_initializers:
            raise TensorCCodegenError("Initializer bindings not declared in Program")
        for value in program.global_values:
            data = initializers.get(value.name)
            if data is None and value.is_constant:
                data = np.asarray(value.const_value, dtype=self.specs[value.name].numpy_dtype)
            if data is None:
                raise TensorCCodegenError(f"Missing initializer tensor data: {value.name}")
            self._constant(value, data)
        inputs = tuple(self.specs[p.name] for p in func.params)
        for index, spec in enumerate(inputs):
            ctype = _TYPES[spec.dtype][0]
            self.body.append(f"if (!inputs || !inputs[{index}]) return 1;")
            self.body.append(f"const {ctype} *{self.names[spec.name]} = (const {ctype} *)inputs[{index}];")
            self._finite_check(spec)

        # Last-use allocation keeps tensors live until their final consumer has
        # finished. The destination is allocated before operands are released.
        last = {}
        for index, instruction in enumerate(instructions):
            for operand in instruction.operands:
                last[operand.name] = index
        free, allocations, high = [], {}, 0
        workspace_declarations = []
        qemu_workspace_buffers = ()
        for index, instruction in enumerate(instructions[:-1]):
            if instruction.dest is None or instruction.dest.name in defined:
                raise TensorCCodegenError(f"Instruction {index}: missing or duplicate SSA destination")
            if instruction.dest.dtype not in _TYPES:
                raise TensorCCodegenError(f"Unsupported dtype {instruction.dest.dtype}")
            if instruction.target:
                raise TensorCCodegenError("Tensor instructions cannot have control-flow targets")
            for operand in instruction.operands:
                if operand.name not in self.specs:
                    if not operand.is_constant:
                        raise TensorCCodegenError(f"Undefined operand {operand.name}")
                    self._register(operand)
                    self._constant(operand, np.asarray(operand.const_value,
                                   dtype=self.specs[operand.name].numpy_dtype))
                    defined.add(operand.name)
                existing = self.specs[operand.name]
                if operand.dtype != existing.dtype:
                    raise TensorCCodegenError(f"Inconsistent operand dtype: {operand.name}")
            try:
                check_instruction(instruction)
                shape, details = self._infer(instruction)
            except (OpError, ValueError, TypeError, IndexError) as exc:
                raise TensorCCodegenError(f"{index}:{instruction.opcode.value}: {exc}") from exc
            value = instruction.dest
            if value.shape and _shape(value.shape) != shape:
                raise TensorCCodegenError(f"{value.name}: declared shape {value.shape} != inferred {shape}")
            spec = self._register(value, shape)
            defined.add(value.name)
            needed = (max(spec.nbytes, 1) + 7) // 8 * 8
            choice = next((j for j, (_, size) in enumerate(free) if size >= needed), None)
            if choice is None:
                offset = high
                high += needed
            else:
                offset, block_size = free.pop(choice)
                if block_size > needed:
                    free.append((offset + needed, block_size - needed))
            if high > self.max_workspace_bytes or high > _UINT64_MAX:
                raise TensorCCodegenError(f"Static workspace exceeds {self.max_workspace_bytes} bytes")
            allocations[value.name] = (offset, needed)
            self.body.append(f"/* {index}: {instruction.opcode.value} */")
            arena = "sv_workspace" if self.constant_storage == "external" else "sv_arena.bytes"
            declaration = (f"{_TYPES[spec.dtype][0]} *{self.names[value.name]} = "
                           f"({_TYPES[spec.dtype][0]} *)({arena} + ")
            if self.qemu_workspace_layout != "none":
                workspace_declarations.append((len(self.body), declaration, offset, needed))
            self.body.append(f"{declaration}{offset}ULL);")
            if self.kernel_calls:
                self._emit_kernel(instruction, spec, details, index)
            else:
                if not self._emit(instruction, spec, details):
                    self._finite_check(spec)
            for name in list(allocations):
                if last.get(name, index) <= index:
                    free.append(allocations.pop(name))
            free.sort()
            merged = []
            for offset, size in free:
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    merged[-1] = (merged[-1][0], merged[-1][1] + size)
                else:
                    merged.append((offset, size))
            free = merged
        if self.qemu_workspace_layout != "none":
            # Keep the existing liveness plan. This QEMU experiment only shifts
            # one of two fixed slots; split or differently sized reuse is not
            # a two-buffer plan, even if no more than two values are live.
            buffers = sorted({(offset, size) for _, _, offset, size in workspace_declarations})
            if (len(buffers) != 2 or buffers[0][0] != 0
                    or buffers[1][0] != buffers[0][1]
                    or high != buffers[1][0] + buffers[1][1]):
                raise TensorCCodegenError(
                    "two-buffer-pad-4k requires exactly two fixed adjacent workspace buffers")
            padded_high = high + 4096
            if padded_high > self.max_workspace_bytes or padded_high > _UINT64_MAX:
                raise TensorCCodegenError(f"Static workspace exceeds {self.max_workspace_bytes} bytes")
            second = buffers[1][0]
            for line, declaration, offset, _ in workspace_declarations:
                if offset == second:
                    self.body[line] = f"{declaration}{offset + 4096}ULL);"
            high = padded_high
            qemu_workspace_buffers = (buffers[0], (second + 4096, buffers[1][1]))
        returned = instructions[-1].operands[0]
        if returned.name not in self.specs:
            if not returned.is_constant:
                raise TensorCCodegenError("Undefined return value")
            self._register(returned)
            self._constant(returned, np.asarray(returned.const_value,
                           dtype=self.specs[returned.name].numpy_dtype))
        output = self.specs[returned.name]
        if returned.dtype != output.dtype:
            raise TensorCCodegenError(f"RETURN {returned.name}: dtype disagrees with defined value")
        # Empty shape on a builder-created intermediate still means "infer".
        # Parameters/globals and an explicit function signature are boundaries:
        # their empty shape is a scalar, never a wildcard for a tensor buffer.
        bound_names = {value.name for value in [*program.global_values, *func.params]}
        inferred_return = isinstance(returned.shape, (tuple, list)) and len(returned.shape) == 0
        if ((not inferred_return or returned.name in bound_names)
                and _shape(returned.shape) != output.shape):
            raise TensorCCodegenError(f"RETURN {returned.name}: shape disagrees with defined value")
        if func.returns:
            if len(func.returns) != 1:
                raise TensorCCodegenError("Function return signature must declare one tensor")
            declared = func.returns[0]
            if declared.dtype != output.dtype or _shape(declared.shape) != output.shape:
                raise TensorCCodegenError("Function return signature dtype/shape disagrees with RETURN")
        ctype = _TYPES[output.dtype][0]
        self.body += ["if (!output) return 1;",
                      f"for (size_t i=0; i<{output.size}ULL; ++i) (({ctype} *)output)[i] = {self.names[output.name]}[i];",
                      "return 0;"]
        if self.constant_storage == "external":
            source = (_PRELUDE + _EXTERNAL_PRELUDE + "".join(self.kernels)
                      + "#if defined(_WIN32)\n__declspec(dllexport)\n#endif\n"
                      + "int scratchv_run_external(const void *const inputs[], "
                      "const void *const weights[], void *workspace, size_t workspace_bytes, "
                      "void *output) {\n  "
                      + "\n  ".join(self._external_prologue(inputs, output, high) + self.body)
                      + "\n}\n")
            return TensorCArtifact(
                source, inputs, output, high, self.constant_bytes,
                function_name="scratchv_run_external", constant_storage="external",
                external_weights=tuple(self.external_weights),
                external_initializers=tuple(self.external_initializers),
                kernel_calls=self.kernel_calls,
                matmul_policy=self.matmul_policy,
                matmul_tile=self.matmul_tile,
                matmul_rhs_layout=self.matmul_rhs_layout,
                matmul_rhs_panels=tuple(self.matmul_rhs_panels),
                qemu_workspace_layout=self.qemu_workspace_layout,
                qemu_workspace_buffers=qemu_workspace_buffers,
            )
        source = (_PRELUDE + "\n" + "\n".join(self.constants)
                  + f"\nstatic union {{ uint64_t align; unsigned char bytes[{max(high, 8)}]; }} sv_arena;\n"
                  + "#if defined(_WIN32)\n__declspec(dllexport)\n#endif\n"
                  + "int scratchv_run(const void *const inputs[], void *output) {\n  "
                  + "\n  ".join(self.body) + "\n}\n")
        return TensorCArtifact(source, inputs, output, high, self.constant_bytes)

    def _panel_inputs(self, program, initializers):
        """Select immutable 2D RHS storage; copy only IR containers we change.

        A 2D RHS may broadcast across LHS batches. A shared non-MatMul use,
        batched RHS or dynamic RHS keeps its original storage. Transpose
        folding is deliberately local: one global's only use is a validated
        2D transpose, and every use of its result is a later MatMul RHS.
        """
        if self.matmul_rhs_layout == "original":
            return program, initializers, {}
        func = program.functions[0]
        instructions = func.blocks[0].instructions
        globals_by_name = {value.name: value for value in program.global_values}
        if set(initializers) - globals_by_name.keys():
            raise TensorCCodegenError("Initializer bindings not declared in Program")
        # Never remove a malformed SSA definition and accidentally legalize it.
        definitions = [*program.global_values, *func.params]
        definitions += [i.dest for i in instructions if i.dest is not None]
        if len({v.name for v in definitions}) != len(definitions):
            raise TensorCCodegenError("Duplicate definition in column-panel graph")
        uses = {}
        for index, instruction in enumerate(instructions):
            for position, value in enumerate(instruction.operands):
                uses.setdefault(value.name, []).append((index, position, instruction))
        # Function signatures are also uses, even if malformed and inconsistent
        # with the final RETURN. Leave their validation to the normal path.
        signature_names = {v.name for v in func.returns}

        def rhs_only(name, after=-1):
            consumers = uses.get(name, ())
            return (name not in signature_names and bool(consumers)
                    and all(index > after and position == 1
                            and instruction.opcode == OpCode.MATMUL
                            and len(instruction.operands) == 2
                            for index, position, instruction in consumers))

        def eligible(value):
            if value.dtype != DataType.FLOAT32 or value.is_constant:
                return False
            shape = _shape(value.shape)
            return len(shape) == 2 and all(d > 0 for d in shape)

        panels, replacements, folded = {}, {}, set()
        bindings = dict(initializers)
        for value in program.global_values:
            if not eligible(value):
                continue
            if rhs_only(value.name):
                panels[value.name] = dict(source_weight=value.name, folded_transpose=None)
                continue
            consumers = uses.get(value.name, ())
            if len(consumers) != 1 or value.name in signature_names:
                continue
            index, position, instruction = consumers[0]
            dest = instruction.dest
            if (position != 0 or instruction.opcode != OpCode.TRANSPOSE
                    or len(instruction.operands) != 1 or dest is None
                    or instruction.target or dest.is_constant
                    or dest.dtype != value.dtype
                    or instruction.operands[0].dtype != value.dtype
                    or not rhs_only(dest.name, after=index)):
                continue
            # Malformed transposes remain in the graph, so normal validation
            # still rejects attributes, permutations and declared shapes.
            try:
                check_instruction(instruction)
                if tuple(instruction.attrs.get("perm", (1, 0))) != (1, 0):
                    continue
                logical_shape = tuple(reversed(_shape(value.shape)))
                if dest.shape and _shape(dest.shape) != logical_shape:
                    continue
            except (OpError, ValueError, TypeError):
                continue
            data = initializers.get(value.name)
            if (not isinstance(data, np.ndarray) or data.shape != value.shape
                    or data.dtype != _TYPES[value.dtype][1]):
                continue
            # No array copy: retain the input mapping and original IR intact.
            replacements[value.name] = replace(dest, shape=logical_shape)
            bindings.pop(value.name)
            bindings[dest.name] = data.T
            folded.add(index)
            panels[dest.name] = dict(source_weight=value.name, folded_transpose=dest.name)
        if not folded:
            return program, initializers, panels
        transformed = copy(program)
        transformed.global_values = [replacements.get(v.name, v) for v in program.global_values]
        function, block = copy(func), copy(func.blocks[0])
        block.instructions = [instruction for index, instruction in enumerate(instructions)
                              if index not in folded]
        function.blocks = [block]
        transformed.functions = [function]
        return transformed, bindings, panels

    def _register(self, value, shape=None):
        if value.dtype not in _TYPES:
            raise TensorCCodegenError(f"Unsupported dtype {value.dtype}; use FP32/INT32/INT64")
        shape = _shape(value.shape if shape is None else shape)
        spec = TensorSpec(value.name, value.dtype, shape)
        if spec.nbytes > _UINT64_MAX:
            raise TensorCCodegenError("Tensor byte count exceeds 64-bit address space")
        self.specs[value.name] = spec
        self.names[value.name] = f"sv_v{self._counter}"
        self._counter += 1
        return spec

    def _constant(self, value, data):
        spec = self.specs[value.name]
        if not isinstance(data, np.ndarray) or data.shape != spec.shape or data.dtype != spec.numpy_dtype:
            raise TensorCCodegenError(f"Initializer {value.name} dtype/shape disagrees with IR")
        if value.is_constant:
            # A scalar literal is part of the IR's meaning (and may have been
            # used by optimizations). A caller binding cannot override it.
            try:
                expected = np.asarray(value.const_value, dtype=spec.numpy_dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TensorCCodegenError(f"Invalid scalar constant {value.name}") from exc
            if data.shape != () or expected.shape != () or not np.array_equal(data, expected):
                raise TensorCCodegenError(f"Initializer {value.name} disagrees with scalar constant")
            if (self.constant_storage == "external" and spec.dtype == DataType.FLOAT32
                    and data.view(np.uint32).item() != expected.view(np.uint32).item()):
                raise TensorCCodegenError(f"Initializer {value.name} disagrees with scalar constant bits")
        physical = spec
        panel = self._rhs_panel_sources.get(value.name)
        if panel is not None:
            k, n = spec.shape
            columns = int(self.matmul_tile.split("x")[1])
            physical = TensorSpec(spec.name, spec.dtype,
                                  _shape(((n + columns - 1) // columns, k, columns)))
        self.constant_bytes += physical.nbytes
        if self.constant_bytes > self.max_constant_bytes or self.constant_bytes > _UINT64_MAX:
            raise TensorCCodegenError("Constant tensor storage exceeds configured limit")
        if self.constant_storage == "external":
            # nditer bounds temporary memory even for a noncontiguous mmap/view.
            if spec.dtype == DataType.FLOAT32:
                for chunk in np.nditer(data, flags=["external_loop", "buffered", "zerosize_ok"],
                                       op_flags=["readonly"], order="C", buffersize=262144):
                    if not np.isfinite(chunk).all():
                        raise TensorCCodegenError("Nonfinite initializer/constant")
            if panel is not None:
                count, k, columns = physical.shape
                n = spec.shape[1]
                if n % columns == 0:
                    # Assigning shape to a view cannot silently make the
                    # complete copy that reshape() may make for strided data.
                    split = data.view()
                    try:
                        split.shape = (k, count, columns)
                    except (AttributeError, ValueError) as exc:
                        raise TensorCCodegenError("Cannot form a zero-copy column-panel view") from exc
                    packed = split.transpose(1, 0, 2)
                    if not np.shares_memory(packed, data):
                        raise TensorCCodegenError("Column-panel view unexpectedly copied its source")
                    materialization = "view"
                else:
                    # The physical byte limit was checked before this bounded
                    # allocation. Inactive tail lanes stay finite zero.
                    packed = np.zeros(physical.shape, dtype=data.dtype)
                    for index in range(count):
                        start, stop = index * columns, min((index + 1) * columns, n)
                        packed[index, :, :stop-start] = data[:, start:stop]
                    materialization = "padded-copy"
                self.matmul_rhs_panels.append(dict(
                    name=spec.name, logical_shape=list(spec.shape),
                    physical_shape=list(physical.shape), panel_columns=columns,
                    **panel, materialization=materialization))
                data = packed
            self.external_weights.append(physical)
            self.external_initializers.append(data)
            if value.is_constant:
                name = self.names[value.name]
                if spec.dtype == DataType.FLOAT32:
                    expected_bits = int(data.view(np.uint32).item())
                    check = f"sv_float_bits({name}[0]) != {expected_bits}U"
                else:
                    check = f"{name}[0] != {_literal(value.const_value, spec.dtype)}"
                self.external_scalar_checks.append(f"if ({check}) return 2;")
            return
        values = [_literal(v.item(), spec.dtype) for v in data.ravel()]
        lines = [", ".join(values[i:i + 8]) for i in range(0, len(values), 8)]
        self.constants.append(f"static const {_TYPES[spec.dtype][0]} {self.names[value.name]}"
                              f"[{max(spec.size, 1)}] = {{\n" + ",\n".join(lines or ["0"]) + "\n};")

    def _external_prologue(self, inputs, output, workspace_bytes):
        """Validate all external addresses before dereferencing tensor data."""
        lines = [
            f"if (workspace_bytes < {workspace_bytes}ULL) return 1;",
            f"if ({workspace_bytes}ULL && !sv_buffer(workspace, {workspace_bytes}ULL, 8)) return 1;",
            f"if (!sv_buffer(output, {output.nbytes}ULL, {output.numpy_dtype.itemsize})) return 1;",
            f"if (sv_overlap(workspace, {workspace_bytes}ULL, output, {output.nbytes}ULL)) return 1;",
            "unsigned char *sv_workspace = (unsigned char *)workspace;",
        ]
        for table, specs in (("inputs", inputs), ("weights", self.external_weights)):
            if specs:
                lines.append(f"if (!sv_buffer({table}, {len(specs)}ULL * sizeof(void *), "
                             "_Alignof(void *))) return 1;")
                lines.append(f"if (sv_overlap(workspace, {workspace_bytes}ULL, {table}, "
                             f"{len(specs)}ULL * sizeof(void *)) || sv_overlap(output, "
                             f"{output.nbytes}ULL, {table}, {len(specs)}ULL * sizeof(void *))) return 1;")
            for index, spec in enumerate(specs):
                pointer = f"{table}[{index}]"
                lines.append(f"if (!sv_buffer({pointer}, {spec.nbytes}ULL, "
                             f"{spec.numpy_dtype.itemsize})) return 1;")
                lines.append(f"if (sv_overlap(workspace, {workspace_bytes}ULL, {pointer}, "
                             f"{spec.nbytes}ULL) || sv_overlap(output, {output.nbytes}ULL, "
                             f"{pointer}, {spec.nbytes}ULL)) return 1;")
                if table == "weights":
                    ctype, name = _TYPES[spec.dtype][0], self.names[spec.name]
                    lines.append(f"const {ctype} *{name} = (const {ctype} *){pointer};")
                    if spec.dtype == DataType.FLOAT32:
                        lines.append(f"for (size_t i=0; i<{spec.size}ULL; ++i) "
                                     f"if (!sv_finite({name}[i])) return 2;")
        return lines + self.external_scalar_checks

    def _finite_check(self, spec):
        if spec.dtype == DataType.FLOAT32:
            self.body.append(f"for (size_t i=0; i<{spec.size}ULL; ++i) "
                             f"if (!sv_finite({self.names[spec.name]}[i])) return 2;")

    def _emit_kernel(self, instruction, output, details, index):
        """Bound each C optimizer unit without changing the arithmetic body.

        Keep output validation with the operation, including fused checks.
        Shape-specialized loops retain their reduction order and FP flags.
        Explicit noinline prevents LLVM from rebuilding the giant graph body.
        """
        start = len(self.body)
        if not self._emit(instruction, output, details):
            self._finite_check(output)
        statements = self.body[start:]
        del self.body[start:]
        names = [self.names[output.name]]
        parameters = [f"{_TYPES[output.dtype][0]} *{names[0]}"]
        unique = {}
        for value in instruction.operands:
            unique.setdefault(value.name, value)
        for value in unique.values():
            names.append(self.names[value.name])
            parameters.append(f"const {_TYPES[value.dtype][0]} *{names[-1]}")
        self.kernels.append(
            f"\nstatic __attribute__((noinline)) int sv_kernel_{index}({', '.join(parameters)}) {{\n  "
            + "\n  ".join(statements) + "\n  return 0;\n}\n"
        )
        self.body.append(f"int sv_status_{index} = sv_kernel_{index}({', '.join(names)});")
        self.body.append(f"if (sv_status_{index}) return sv_status_{index};")

    def _infer(self, instruction):
        op, attrs = instruction.opcode, instruction.attrs
        xs = [self.specs[x.name] for x in instruction.operands]
        dtype = instruction.dest.dtype
        unary = {OpCode.NEG, OpCode.ABS, OpCode.COS, OpCode.SIN, OpCode.RECIPROCAL,
                 OpCode.EXP, OpCode.SQRT, OpCode.SIGMOID, OpCode.RELU, OpCode.CAST}
        binary = {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW}
        if op == OpCode.LOAD_CONST:
            if xs:
                raise TensorCCodegenError("LOAD_CONST cannot have operands")
            _literal(attrs["value"], dtype)
            return (), {}
        expected_count = 1 if op in unary or op in {
            OpCode.REDUCE_MEAN, OpCode.SOFTMAX, OpCode.RESHAPE, OpCode.TRANSPOSE,
            OpCode.SLICE, OpCode.UNSQUEEZE, OpCode.EXPAND} else 2
        if op == OpCode.CONCAT:
            if not xs:
                raise TensorCCodegenError("Concat needs at least one input")
        elif len(xs) != expected_count:
            raise TensorCCodegenError(f"Expected {expected_count} operands")
        if op != OpCode.CAST and dtype != xs[0].dtype:
            raise TensorCCodegenError("Output dtype must match first operand")
        if op in unary:
            return xs[0].shape, {}
        if op in binary:
            if op != OpCode.POW and xs[0].dtype != xs[1].dtype:
                raise TensorCCodegenError("Arithmetic operands must have equal dtype")
            if op == OpCode.POW and dtype != DataType.FLOAT32 and xs[1].dtype == DataType.FLOAT32:
                raise TensorCCodegenError("Integer Pow requires integer exponent")
            return _broadcast(*(x.shape for x in xs)), {}
        if op == OpCode.MATMUL:
            if any(x.dtype != DataType.FLOAT32 or not x.shape for x in xs):
                raise TensorCCodegenError("MatMul requires nonscalar FP32 tensors")
            a, b = xs[0].shape, xs[1].shape
            if "m" in attrs:
                m, n, k = (attrs[key] for key in ("m", "n", "k"))
                if len(a) == len(b) == 1 and math.prod(a) == m*k and math.prod(b) == k*n:
                    a, b = (m, k), (k, n)
                elif a != (m, k) or b != (k, n):
                    raise TensorCCodegenError("MatMul m/n/k disagree with tensor dimensions")
            av, bv = len(a) == 1, len(b) == 1
            a = (1, *a) if av else a
            b = (*b, 1) if bv else b
            if a[-1] != b[-2]:
                raise TensorCCodegenError("MatMul inner dimensions disagree")
            batch = _broadcast(a[:-2], b[:-2])
            result = batch + (() if av else (a[-2],)) + (() if bv else (b[-1],))
            return result, dict(a=a, b=b, batch=batch)
        x = xs[0]
        if op == OpCode.SOFTMAX:
            axis = _axis(attrs.get("axis", -1), len(x.shape))
            if not x.shape[axis]:
                raise TensorCCodegenError("Empty Softmax axis")
            return x.shape, dict(axis=axis)
        if op == OpCode.REDUCE_MEAN:
            axes = _axes(attrs.get("axes") or tuple(range(len(x.shape))), len(x.shape))
            if any(not x.shape[a] for a in axes):
                raise TensorCCodegenError("Cannot ReduceMean an empty axis")
            keep = attrs.get("keepdims", True)
            result = tuple(1 if a in axes else d for a, d in enumerate(x.shape)) if keep else tuple(
                d for a, d in enumerate(x.shape) if a not in axes)
            return result, dict(axes=axes)
        if op == OpCode.RESHAPE:
            target = list(attrs["shape"])
            for axis, dim in enumerate(target):
                if dim == 0:
                    if axis >= len(x.shape):
                        raise TensorCCodegenError("Reshape zero cannot copy absent input axis")
                    target[axis] = x.shape[axis]
            if -1 in target:
                denominator = math.prod(d for d in target if d != -1)
                if denominator == 0 or x.size % denominator:
                    raise TensorCCodegenError("Reshape inference is ambiguous or nonintegral")
                target[target.index(-1)] = x.size // denominator
            if math.prod(target) != x.size:
                raise TensorCCodegenError("Reshape element counts differ")
            return _shape(target), {}
        if op == OpCode.TRANSPOSE:
            perm = tuple(attrs.get("perm", tuple(reversed(range(len(x.shape))))))
            if sorted(perm) != list(range(len(x.shape))):
                raise TensorCCodegenError("Transpose requires a complete permutation")
            return tuple(x.shape[a] for a in perm), dict(perm=perm)
        if op == OpCode.CONCAT:
            axis = _axis(attrs["axis"], len(x.shape))
            for operand in xs:
                if (operand.dtype != dtype or len(operand.shape) != len(x.shape) or any(
                    a != axis and d != x.shape[a] for a, d in enumerate(operand.shape))):
                    raise TensorCCodegenError("Concat input dimensions/dtypes disagree")
            result = list(x.shape)
            result[axis] = sum(v.shape[axis] for v in xs)
            return tuple(result), dict(axis=axis)
        if op == OpCode.GATHER:
            if xs[1].dtype not in (DataType.INT32, DataType.INT64):
                raise TensorCCodegenError("Gather indices must be integer tensors")
            axis = _axis(attrs.get("axis", 0), len(x.shape))
            return x.shape[:axis] + xs[1].shape + x.shape[axis+1:], dict(axis=axis)
        if op == OpCode.SLICE:
            axes = _axes(attrs.get("axes", tuple(range(len(attrs["starts"])))), len(x.shape))
            steps = attrs.get("steps", (1,) * len(axes))
            starts, factors, result = [0] * len(x.shape), [1] * len(x.shape), list(x.shape)
            for axis, start, end, step in zip(axes, attrs["starts"], attrs["ends"], steps):
                dimension = x.shape[axis]
                start = start + dimension if start < 0 else start
                end = end + dimension if end < 0 else end
                if step > 0:
                    start, end = max(0, min(dimension, start)), max(0, min(dimension, end))
                else:
                    start, end = max(0, min(dimension-1, start)), max(-1, min(dimension-1, end))
                starts[axis], factors[axis] = start, step
                result[axis] = len(range(start, end, step)) if dimension else 0
            return tuple(result), dict(starts=starts, steps=factors)
        if op == OpCode.UNSQUEEZE:
            rank = len(x.shape) + len(attrs["axes"])
            axes = _axes(attrs["axes"], rank)
            dims = iter(x.shape)
            return tuple(1 if a in axes else next(dims) for a in range(rank)), {}
        if op == OpCode.EXPAND:
            return _broadcast(x.shape, tuple(attrs["shape"])), {}
        raise TensorCCodegenError(f"Unsupported tensor opcode: {op.value}")

    def _emit(self, instruction, output, details):
        """Emit the operation; return True when it also checks every output."""
        op, attrs = instruction.opcode, instruction.attrs
        xs = [self.specs[x.name] for x in instruction.operands]
        dst = self.names[output.name]
        dtype, size = output.dtype, output.size
        names = [self.names[x.name] for x in xs]
        # Bounds are checked even when an unrelated zero-sized axis makes the
        # output empty. The NumPy IR contract still rejects invalid indices.
        if op == OpCode.GATHER:
            dimension = xs[0].shape[details["axis"]]
            self.body.append(f"for (size_t j=0; j<{xs[1].size}ULL; ++j) if ({names[1]}[j]<-{dimension}LL "
                             f"|| {names[1]}[j]>={dimension}LL) return 3;")
        if not size:
            return
        if op == OpCode.LOAD_CONST:
            self.body.append(f"{dst}[0] = {_literal(attrs['value'], dtype)};")
            return
        refs = [f"{name}[{_broadcast_offset(output.shape, x.shape)}]"
                for name, x in zip(names, xs)] if op in {
                    OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW, OpCode.EXPAND} else []
        if op in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW}:
            a, b = refs
            checks = ""
            symbol = {OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*", OpCode.DIV: "/"}.get(op)
            if op == OpCode.POW:
                if dtype != DataType.FLOAT32:
                    ctype = _TYPES[dtype][0]
                    expression = f"({ctype})acc"
                    checks = (f"if ({b}<0) return 2; {ctype} acc=1, factor={a}; uint64_t exponent=(uint64_t){b}; "
                              "while (exponent) { if ((exponent & 1ULL) && __builtin_mul_overflow(acc,factor,&acc)) return 2; "
                              "exponent >>= 1; if (exponent && __builtin_mul_overflow(factor,factor,&factor)) return 2; } ")
                else:
                    # powf is exact about integer parity only while an exponent
                    # fits FP32. Reject larger runtime integer exponents instead
                    # of silently rounding them (Qwen uses INT64 scalar 2).
                    if xs[1].dtype != DataType.FLOAT32:
                        checks += f"if ({b}>16777216LL || {b}<(-16777216LL)) return 2; "
                    checks += f"if ({a}==0.0f && {b}<0) return 2; "
                    expression = f"powf({a}, (float){b})"
            elif dtype == DataType.FLOAT32:
                if op == OpCode.DIV:
                    checks = f"if ({b}==0.0f) return 2; "
                expression = f"{a} {symbol} {b}"
            else:
                bits = output.numpy_dtype.itemsize * 8
                ctype = _TYPES[dtype][0]
                if op == OpCode.DIV:
                    minimum = _literal(np.iinfo(output.numpy_dtype).min, dtype)
                    checks = f"if ({b}==0 || ({a}=={minimum} && {b}==-1)) return 2; "
                    expression = f"{a} / {b}"
                else:
                    expression = f"({ctype})((uint{bits}_t){a} {symbol} (uint{bits}_t){b})"
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ {checks}{dst}[i]={expression}; }}")
            return
        if op in {OpCode.NEG, OpCode.ABS, OpCode.COS, OpCode.SIN, OpCode.RECIPROCAL,
                  OpCode.EXP, OpCode.SQRT, OpCode.SIGMOID, OpCode.RELU, OpCode.CAST}:
            a, checks = f"{names[0]}[i]", ""
            if op == OpCode.CAST:
                if dtype != DataType.FLOAT32 and xs[0].dtype == DataType.FLOAT32:
                    bound = float(1 << (output.numpy_dtype.itemsize * 8 - 1)).hex() + "f"
                    checks = f"if ({a} < -{bound} || {a} >= {bound}) return 2; "
                expression = f"({_TYPES[dtype][0]}){a}"
            elif op == OpCode.NEG:
                expression = f"-{a}" if dtype == DataType.FLOAT32 else (
                    f"({_TYPES[dtype][0]})(0ULL-(uint{output.numpy_dtype.itemsize*8}_t){a})")
            elif op == OpCode.ABS:
                if dtype != DataType.FLOAT32:
                    checks = f"if ({a}=={_literal(np.iinfo(output.numpy_dtype).min,dtype)}) return 2; "
                expression = (f"__builtin_fabsf({a})" if dtype == DataType.FLOAT32
                              else f"({a}<0 ? -{a} : {a})")
            elif op == OpCode.RECIPROCAL:
                checks, expression = f"if ({a}==0.0f) return 2; ", f"1.0f/{a}"
            elif op == OpCode.RELU:
                expression = f"({a}>0 ? {a} : 0)"
            else:
                function = {OpCode.COS:"cosf", OpCode.SIN:"sinf", OpCode.EXP:"expf",
                            OpCode.SQRT:"sqrtf", OpCode.SIGMOID:"sv_sigmoid"}[op]
                if op == OpCode.SQRT:
                    checks = f"if ({a}<0.0f) return 2; "
                expression = f"{function}({a})"
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ {checks}{dst}[i]={expression}; }}")
            return
        x, src = xs[0], names[0]
        if op in {OpCode.RESHAPE, OpCode.UNSQUEEZE}:
            expression = f"{src}[i]"
        elif op == OpCode.EXPAND:
            expression = refs[0]
        elif op == OpCode.TRANSPOSE:
            offset = _offset(output.shape, [_strides(x.shape)[a] for a in details["perm"]])
            expression = f"{src}[{offset}]"
        elif op == OpCode.SLICE:
            strides = _strides(x.shape)
            offset = _offset(output.shape, [a*b for a,b in zip(strides,details["steps"])],
                             base=sum(a*b for a,b in zip(strides,details["starts"])))
            expression = f"{src}[{offset}]"
        else:
            expression = None
        if expression is not None:
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {dst}[i]={expression};")
            return
        if op == OpCode.MATMUL:
            a, b, batch = details["a"], details["b"], details["batch"]
            m, k, n = a[-2], a[-1], b[-1]
            left = _broadcast_offset(batch, a[:-2], "batch")
            right = _broadcast_offset(batch, b[:-2], "batch")
            if self.matmul_policy == "blocked_fma":
                # Each output retains increasing K order, fused FP32 products,
                # and a separate FP32 addition after each 128-element K block.
                # Tiles only share input loads across independent outputs; they
                # never split a reduction between harts or change its grouping.
                # __builtin_fmaf retains explicit FMA with -ffp-contract=off.
                tile_rows, tile_columns = (int(part) for part in self.matmul_tile.split("x"))
                lanes = tile_rows * tile_columns
                row_step = "++m" if tile_rows == 1 else f"m+={tile_rows}"
                lines = [
                    f"for (size_t batch=0; batch<{math.prod(batch)}ULL; ++batch) "
                    f"for (size_t m=0; m<{m}ULL; {row_step}) "
                    f"for (size_t column=0; column<{n}ULL; column+={tile_columns}) {{",
                    "float " + ", ".join(f"sum{lane}=0.0f" for lane in range(lanes)) + ";",
                    f"for (size_t block=0; block<{k}ULL; block+=128) {{",
                    "float " + ", ".join(f"acc{lane}=0.0f" for lane in range(lanes)) + ";",
                    f"size_t end=block+128 < {k}ULL ? block+128 : {k}ULL;",
                    "for (size_t k=block; k<end; ++k) {",
                    f"float left_value={names[0]}[({left})*{m*k}ULL+m*{k}ULL+k];",
                ]
                if tile_rows == 2:
                    lines += [
                        "float left_value1=0.0f;",
                        f"if (m+1<{m}ULL) left_value1="
                        f"{names[0]}[({left})*{m*k}ULL+(m+1)*{k}ULL+k];",
                    ]
                if instruction.operands[1].name in self._rhs_panel_sources:
                    lines.append(
                        f"const float *right_values={names[1]}+(column/{tile_columns}ULL)*"
                        f"{k*tile_columns}ULL+k*{tile_columns}ULL;")
                else:
                    lines.append(
                        f"const float *right_values={names[1]}+({right})*{k*n}ULL+k*{n}ULL+column;")
                for column_lane in range(tile_columns):
                    guard = f"if (column+{column_lane}<{n}ULL) " if column_lane else ""
                    if tile_rows == 1:
                        lines.append(
                            f"{guard}acc{column_lane}="
                            f"__builtin_fmaf(left_value,right_values[{column_lane}],acc{column_lane});")
                    else:
                        # Load each B element once for both rows. A missing tail
                        # row/column performs neither a load nor an FP operation.
                        lines.append(
                            f"{guard}{{ float right_value=right_values[{column_lane}]; "
                            f"acc{column_lane}=__builtin_fmaf(left_value,right_value,acc{column_lane}); "
                            f"if (m+1<{m}ULL) acc{tile_columns+column_lane}="
                            f"__builtin_fmaf(left_value1,right_value,acc{tile_columns+column_lane}); }}")
                lines.append("}")
                if self.matmul_tile == "1x4":
                    # Preserve the original default source, including inactive
                    # lanes' exact 0+0 additions (which raise no FP flags).
                    lines.append("sum0+=acc0; sum1+=acc1; sum2+=acc2; sum3+=acc3;")
                else:
                    for row_lane in range(tile_rows):
                        for column_lane in range(tile_columns):
                            lane = row_lane * tile_columns + column_lane
                            conditions = ([f"m+{row_lane}<{m}ULL"] if row_lane else [])
                            if column_lane:
                                conditions.append(f"column+{column_lane}<{n}ULL")
                            guard = f"if ({' && '.join(conditions)}) " if conditions else ""
                            lines.append(f"{guard}sum{lane}+=acc{lane};")
                lines.append("}")
                for row_lane in range(tile_rows):
                    row = f"(m+{row_lane})" if row_lane else "m"
                    for column_lane in range(tile_columns):
                        lane = row_lane * tile_columns + column_lane
                        conditions = ([f"m+{row_lane}<{m}ULL"] if row_lane else [])
                        if column_lane:
                            conditions.append(f"column+{column_lane}<{n}ULL")
                        guard = f"if ({' && '.join(conditions)}) " if conditions else ""
                        column = f"column+{column_lane}" if column_lane else "column"
                        lines.append(
                            f"{guard}{dst}[batch*{m*n}ULL+{row}*{n}ULL+{column}]=sum{lane};")
                lines.append("}")
                serial = " ".join(lines)
                columns = (n + tile_columns - 1) // tile_columns
                rows = (m + tile_rows - 1) // tile_rows
                tasks = math.prod(batch) * rows * columns
                # Avoid dispatch/barrier overhead on small matrices. This is a
                # conservative work floor, not a calibrated performance claim.
                # Each task owns one (batch, row tile, column tile); its K loop
                # and 128-element grouping are shared verbatim with serial.
                if tasks >= 2 and math.prod(batch) * m * n * k >= 32768:
                    condition = ("defined(SCRATCHV_HARTS) && SCRATCHV_HARTS > 1 "
                                 f"&& SCRATCHV_HARTS <= {tasks}")
                    context = f"sv_matmul_{dst}"
                    aliases = []
                    seen = set()
                    for name, member in zip(names, ("left", "right")):
                        if name not in seen:
                            aliases.append(f"const float *{name}=context->{member};")
                            seen.add(name)
                    self.kernels.append(
                        f"\n#if {condition}\n"
                        f"struct {context} {{ const float *left, *right; float *output; }};\n"
                        f"static void {context}_range(void *opaque, size_t begin, size_t end) {{\n"
                        f"  const struct {context} *context=opaque;\n  "
                        + " ".join(aliases) + f" float *{dst}=context->output;\n"
                        "  for (size_t task=begin; task<end; ++task) {\n"
                        f"    size_t column=(task % {columns}ULL)*{tile_columns};\n"
                        + (f"    size_t m=(task / {columns}ULL) % {rows}ULL;\n"
                           if tile_rows == 1 else
                           f"    size_t m=((task / {columns}ULL) % {rows}ULL)*{tile_rows};\n")
                        + f"    size_t batch=task / {columns*rows}ULL;\n    "
                        + " ".join(lines[1:-1]) + "\n  }\n}\n#endif\n"
                    )
                    self.body.append(
                        f"\n#if {condition}\n"
                        f"struct {context} context_{dst}={{{names[0]}, {names[1]}, {dst}}};\n"
                        f"sv_parallel_for({tasks}ULL, {context}_range, &context_{dst});\n"
                        f"#else\n{serial}\n#endif\n"
                    )
                else:
                    self.body.append(serial)
                return
            self.body.append(f"for (size_t batch=0; batch<{math.prod(batch)}ULL; ++batch) "
                f"for (size_t m=0; m<{m}ULL; ++m) for (size_t n=0; n<{n}ULL; ++n) {{ "
                "float acc=0.0f; "
                f"for (size_t k=0; k<{k}ULL; ++k) acc += {names[0]}[({left})*{m*k}ULL+m*{k}ULL+k] "
                f"* {names[1]}[({right})*{k*n}ULL+k*{n}ULL+n]; "
                f"{dst}[batch*{m*n}ULL+m*{n}ULL+n]=acc; }}")
        elif op == OpCode.REDUCE_MEAN:
            axes = details["axes"]
            remaining = tuple(a for a in range(len(x.shape)) if a not in axes)
            outer_shape = tuple(x.shape[a] for a in remaining)
            reduced_shape = tuple(x.shape[a] for a in axes)
            outer_offset = _offset(outer_shape, [_strides(x.shape)[a] for a in remaining], "i")
            red_offset = _offset(reduced_shape, [_strides(x.shape)[a] for a in axes], "r")
            count = math.prod(reduced_shape)
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ float acc=0.0f; "
                f"for (size_t r=0; r<{count}ULL; ++r) acc += {src}[({outer_offset})+({red_offset})]; "
                f"{dst}[i]=acc/{float(count).hex()}f; }}")
        elif op == OpCode.SOFTMAX:
            axis = details["axis"]
            outer, dimension, inner = math.prod(x.shape[:axis]), x.shape[axis], math.prod(x.shape[axis+1:])
            off = f"o*{dimension*inner}ULL+j*{inner}ULL+t"
            # A row retains its original max, exp/sum and normalization order.
            # Only independent (outer, inner) rows may run on different harts.
            row = ("float maximum=-0x1.fffffep+127f, sum=0.0f; "
                f"for (size_t j=0; j<{dimension}ULL; ++j) if ({src}[{off}]>maximum) maximum={src}[{off}]; "
                f"for (size_t j=0; j<{dimension}ULL; ++j) {{ float shifted={src}[{off}]-maximum; "
                "if (!sv_finite(shifted)) return 2; float v=expf(shifted); "
                f"{dst}[{off}]=v; sum+=v; }} "
                f"for (size_t j=0; j<{dimension}ULL; ++j) {{ {dst}[{off}]/=sum; "
                f"if (!sv_finite({dst}[{off}])) return 2; }} ")
            serial = (f"for (size_t o=0; o<{outer}ULL; ++o) "
                      f"for (size_t t=0; t<{inner}ULL; ++t) {{ " + row + "}")
            rows = outer * inner
            # Keep small/empty graphs and inline storage on the original path.
            # This floor avoids a barrier for tiny graphs; it is not autotuning.
            if self.constant_storage == "external" and rows >= 2 and size >= 32768:
                condition = ("defined(SCRATCHV_HARTS) && SCRATCHV_HARTS > 1 "
                             f"&& SCRATCHV_HARTS <= {rows}")
                # Reuse code only: each operation keeps its own input/output
                # pointers and status, initialized below before the joined call.
                key = (x.dtype, x.shape, axis)
                context = self._softmax_callbacks.get(key)
                if context is None:
                    context = f"sv_softmax_{dst}"
                    parallel_row = row.replace(
                        "return 2;",
                        "{ "
                        "__atomic_store_n(&context->status, 2, __ATOMIC_RELAXED); return; }")
                    self.kernels.append(
                        f"\n#if {condition}\n"
                        f"struct {context} {{ const float *input; float *output; int status; }};\n"
                        f"static void {context}_range(void *opaque, size_t begin, size_t end) {{\n"
                        f"  struct {context} *context=opaque;\n"
                        f"  const float *{src}=context->input; float *{dst}=context->output;\n"
                        "  for (size_t task=begin; task<end; ++task) {\n"
                        f"    size_t o=task / {inner}ULL, t=task % {inner}ULL;\n    "
                        + parallel_row + "\n  }\n}\n#endif\n"
                    )
                    self._softmax_callbacks[key] = context
                self.body.append(
                    f"\n#if {condition}\n"
                    f"struct {context} context_{dst}={{{src}, {dst}, 0}};\n"
                    f"sv_parallel_for({rows}ULL, {context}_range, &context_{dst});\n"
                    # All callbacks finish before checking the shared status or
                    # reusing their output. Concurrent failures only write 2.
                    f"if (__atomic_load_n(&context_{dst}.status, __ATOMIC_RELAXED)) return 2;\n"
                    f"#else\n{serial}\n#endif\n"
                )
            else:
                self.body.append(serial)
            # Both serial and parallel rows validate the FP32 value just stored.
            # The joined callback reports failure before any workspace reuse.
            return True
        elif op == OpCode.GATHER:
            axis, count = details["axis"], xs[1].size
            dimension, inner = x.shape[axis], math.prod(x.shape[axis+1:])
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ int64_t selected="
                f"{names[1]}[(i/{inner}ULL)%{count}ULL]; if (selected<0) selected+={dimension}LL; "
                f"{dst}[i]={src}[(i/{inner*count}ULL)*{dimension*inner}ULL+selected*{inner}LL+i%{inner}ULL]; }}")
        elif op == OpCode.CONCAT:
            axis = details["axis"]
            inner, total, start = math.prod(output.shape[axis+1:]), output.shape[axis], 0
            for operand, name in zip(xs, names):
                chunk = operand.shape[axis] * inner
                if operand.size:
                    self.body.append(f"for (size_t i=0; i<{operand.size}ULL; ++i) "
                        f"{dst}[(i/{chunk}ULL)*{total*inner}ULL+{start*inner}ULL+i%{chunk}ULL]={name}[i];")
                start += operand.shape[axis]
        else:
            raise TensorCCodegenError(f"No C lowering for {op.value}")
