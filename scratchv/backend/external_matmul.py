"""Explicit, synchronous external FP32 MatMul contract for Tensor-C.

Providers use row-major buffers and overwrite C without retaining pointers or
modifying A/B. The default numerical contract is increasing-K, separate FP32
multiply/add. Optimized libraries may declare provider_fp32, which requires
explicit permission at generation time. No additional caller scratch space is
supported; a Host library may manage its own internal buffers and runtime state.
Sources share the model's strict floating-point compilation flags.
This is an opt-in build description, not a plugin registry or a library loader.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re


@dataclass(frozen=True)
class ExternalMatMul:
    """One implementation of int symbol(A, B, C, uint64_t M, N, K).

    Sources may contain C sources, objects or static archives. Relative paths
    (including include_dirs) are resolved from the build caller's cwd.
    Targets declare compatible runtime environments, not just CPU ISAs:
    "riscv64" means ScratchV's freestanding RV64/LP64D guest, not Linux.

    parallel_rows explicitly permits concurrent calls on independent row
    ranges. The provider must be reentrant for shared read-only A/B and
    disjoint C buffers, and splitting M must preserve each output's arithmetic.
    It must not retain pointers or start a nested ScratchV worker dispatch.
    This opt-in does not override dimension limits or allow splitting K.
    """

    name: str
    symbol: str = "sv_matmul_f32_v1"
    sources: tuple[str | os.PathLike[str], ...] = ()
    include_dirs: tuple[str | os.PathLike[str], ...] = ()
    link_flags: tuple[str, ...] = ()
    targets: tuple[str, ...] = ("host",)
    max_dimension: int = 2**31 - 1
    alignment: int = 4
    numerics: str = "sequential"
    parallel_rows: bool = False

    def __post_init__(self):
        if type(self.parallel_rows) is not bool:
            raise ValueError("External MatMul parallel_rows must be boolean")
        if self.numerics not in ("sequential", "provider_fp32"):
            raise ValueError("External MatMul numerics must be sequential or provider_fp32")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("External MatMul requires a nonempty name")
        if (not isinstance(self.symbol, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.symbol)):
            raise ValueError("External MatMul symbol must be a C identifier")
        for field in ("sources", "include_dirs", "link_flags", "targets"):
            items = getattr(self, field)
            if not isinstance(items, (tuple, list)):
                raise ValueError(f"External MatMul {field} must be a sequence")
            object.__setattr__(self, field, tuple(items))
        for path in (*self.sources, *self.include_dirs):
            if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
                raise ValueError("External MatMul paths must be nonempty paths")
        if any(not isinstance(flag, str) or not flag for flag in self.link_flags):
            raise ValueError("External MatMul link_flags must contain nonempty strings")
        if not self.targets or any(target not in ("host", "riscv64") for target in self.targets):
            raise ValueError("External MatMul targets must contain host or riscv64")
        if (isinstance(self.max_dimension, bool) or not isinstance(self.max_dimension, int)
                or not 0 <= self.max_dimension <= 2**64 - 1):
            raise ValueError("External MatMul max_dimension must be a nonnegative uint64")
        if (isinstance(self.alignment, bool) or not isinstance(self.alignment, int)
                or self.alignment < 1 or self.alignment & (self.alignment - 1)):
            raise ValueError("External MatMul alignment must be a positive power of two")


@dataclass(frozen=True)
class MatMulSelection:
    """The generation-time choice and reason for one MatMul result."""

    output: str
    implementation: str
    reason: str


@dataclass(frozen=True)
class MatMulMeasurement:
    """One same-condition comparison, supplied explicitly by the caller.

    Full operand shapes distinguish broadcasting and batches. Timings use the
    same scope and include the operation's layout/call costs; a microkernel
    timing must not be presented as an end-to-end operation measurement.
    """

    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...]
    rhs_kind: str
    builtin_rhs_layout: str
    builtin_seconds: float
    external_seconds: float

    def __post_init__(self):
        for field in ("left_shape", "right_shape"):
            shape = getattr(self, field)
            if (not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 16
                    or any(type(d) is not int or d < 0 for d in shape)):
                raise ValueError("Measured MatMul shapes must contain nonnegative integer dimensions")
            object.__setattr__(self, field, tuple(shape))
        if self.rhs_kind not in ("constant", "batched_constant", "transposed_constant", "dynamic"):
            raise ValueError("Unknown measured MatMul RHS kind")
        if self.builtin_rhs_layout not in ("original", "column-panel"):
            raise ValueError("Unknown measured MatMul builtin RHS layout")
        for field in ("builtin_seconds", "external_seconds"):
            value = getattr(self, field)
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("Measured MatMul times must be finite positive numbers")

    @property
    def key(self):
        return (self.left_shape, self.right_shape, self.rhs_kind, self.builtin_rhs_layout)


@dataclass(frozen=True)
class MatMulPerformance:
    """An explicit local measurement profile, not an automatic tuning cache.

    The provider descriptor and generation options must match exactly. The
    environment label records the caller's library version, compiler/machine
    configuration and threading conditions; it is not hardware detection or
    automatic validation of library replacement, CPU load or thread settings.
    Host measurements currently require one ScratchV worker and a single-thread
    library. The caller must query and maintain that library thread setting.
    RISC-V builders additionally enforce the declared hart count.
    """

    provider: ExternalMatMul
    target: str
    environment: str
    harts: int
    measurements: tuple[MatMulMeasurement, ...]
    matmul_policy: str = "blocked_fma"
    matmul_tile: str = "2x4"
    matmul_rhs_layout: str = "column-panel"
    kernel_calls: bool = True
    qemu_workspace_layout: str = "none"
    minimum_gain: float = 0.1

    def __post_init__(self):
        if not isinstance(self.provider, ExternalMatMul):
            raise ValueError("MatMul performance requires an explicit provider")
        if self.target not in ("host", "riscv64"):
            raise ValueError("Unknown MatMul performance target")
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ValueError("MatMul performance requires an environment description")
        if type(self.harts) is not int or self.harts not in (1, 2, 4):
            raise ValueError("MatMul performance harts must be 1, 2 or 4")
        if self.target == "host" and self.harts != 1:
            raise ValueError("Host MatMul performance currently requires harts=1 and a single-thread library")
        if not isinstance(self.measurements, (tuple, list)) or any(
                not isinstance(row, MatMulMeasurement) for row in self.measurements):
            raise ValueError("MatMul performance measurements must be MatMulMeasurement records")
        object.__setattr__(self, "measurements", tuple(self.measurements))
        keys = [row.key for row in self.measurements]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate MatMul performance measurement key")
        if self.matmul_policy not in ("sequential", "blocked_fma"):
            raise ValueError("Unknown measured MatMul policy")
        if self.matmul_tile not in ("1x4", "2x4", "1x8"):
            raise ValueError("Unknown measured MatMul tile")
        if self.matmul_rhs_layout not in ("original", "column-panel"):
            raise ValueError("Unknown measured MatMul RHS layout")
        if type(self.kernel_calls) is not bool:
            raise ValueError("Measured MatMul kernel_calls must be boolean")
        if self.qemu_workspace_layout not in ("none", "two-buffer-pad-4k"):
            raise ValueError("Unknown measured MatMul workspace layout")
        if (type(self.minimum_gain) not in (int, float)
                or not math.isfinite(self.minimum_gain) or not 0 <= self.minimum_gain < 1):
            raise ValueError("Measured MatMul minimum_gain must be finite in [0, 1)")
