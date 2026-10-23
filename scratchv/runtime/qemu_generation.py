"""Fixed-ABI Host generation over one reusable RV64 executable and bundle.

Each token launches a fresh QEMU guest with updated inputs. The ELF and bundle
are built once by the caller; this adapter neither recompiles nor changes hart
selection, code generation policy, weights, or workspace layout.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import math
from numbers import Real
from pathlib import Path
import threading
import time

import numpy as np

from .generation import GenerationCancelledError
from .llm_inputs import PreparedInputs, SEQ_LENGTH
from .riscv_external import ExternalExecutable, run_external


class QemuForward:
    """Callable backend for ``generation.generate`` using the fixed W5 ABI.

    ``timeout`` bounds each guest invocation; ``deadline`` optionally bounds the
    entire generation using an absolute ``time.monotonic()`` deadline. A shared
    cancellation callback is checked during guest UART/QMP waits. Instances are
    intentionally serial, but may be reused by independent generation contexts.
    Step directories are unique and failed invocations retain diagnostic data.
    """

    def __init__(self, executable: ExternalExecutable, output_dir, *,
                 timeout: float = 3600.0, cancelled: Callable[[], bool] | None = None,
                 deadline: float | None = None):
        for name, value in (("timeout", timeout), ("deadline", deadline)):
            if value is None and name == "deadline":
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or (name == "timeout" and value <= 0):
                raise ValueError(f"{name} must be finite" + (" and positive" if name == "timeout" else ""))
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if executable.evidence.get("qemu_workspace_layout", "none") != "none":
            raise ValueError("W5 generation requires qemu_workspace_layout=none")
        specs = {spec.name: spec for spec in executable.inputs}
        expected = {"input_ids": ((1, SEQ_LENGTH), np.dtype("int64")),
                    "attention_mask": ((1, 1, SEQ_LENGTH, SEQ_LENGTH), np.dtype("float32"))}
        if len(specs) != len(executable.inputs) or set(specs) != set(expected):
            raise ValueError("W5 requires exactly input_ids and attention_mask inputs")
        for name, (shape, dtype) in expected.items():
            if tuple(specs[name].shape) != shape or specs[name].numpy_dtype != dtype:
                raise ValueError(f"W5 {name} shape/dtype differs from the fixed ABI")
        output = executable.output
        if (len(output.shape) != 3 or tuple(output.shape[:2]) != (1, SEQ_LENGTH)
                or output.shape[2] < 1 or output.numpy_dtype != np.dtype("float32")):
            raise ValueError("W5 output must be full FP32 [1,256,vocab] logits")
        self.executable = executable
        self.vocab_size = output.shape[2]
        self.output_dir = Path(output_dir).resolve()
        # Refuse stale step/report reuse instead of overwriting previous runs.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.timeout = float(timeout)
        self.cancelled = cancelled
        self.deadline = None if deadline is None else float(deadline)
        self.runs: list[dict] = []
        self._lock = threading.Lock()
        self._next_step = 0

    def _cancelled(self):
        if self.cancelled is not None and self.cancelled():
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("generation timeout expired")
        return False

    def __call__(self, prepared: PreparedInputs) -> np.ndarray:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("QemuForward does not support overlapping invocations")
        try:
            if not isinstance(prepared, PreparedInputs):
                raise TypeError("QemuForward requires PreparedInputs")
            if (isinstance(prepared.valid_length, (bool, np.bool_))
                    or not isinstance(prepared.valid_length, (int, np.integer))
                    or not 1 <= prepared.valid_length <= SEQ_LENGTH):
                raise ValueError("valid_length must be in [1,256]")
            if self._cancelled():
                raise GenerationCancelledError("generation cancelled")
            timeout = self.timeout
            if self.deadline is not None:
                timeout = min(timeout, self.deadline - time.monotonic())
                if timeout <= 0:
                    raise TimeoutError("generation timeout expired")
            step_dir = self.output_dir / f"step-{self._next_step:04d}"
            self._next_step += 1
            checkpoint = self._cancelled if self.cancelled is not None or self.deadline is not None else None
            try:
                output, report = run_external(
                    self.executable, prepared.as_feed(), step_dir,
                    timeout=timeout, cancelled=checkpoint,
                )
                # run_external already verifies every output element is finite;
                # generate retains its full-output check before selecting a row.
                if output.shape != (1, SEQ_LENGTH, self.vocab_size) or output.dtype != np.float32:
                    raise ValueError("QEMU returned logits outside the fixed ABI")
                if self._cancelled():
                    raise GenerationCancelledError("generation cancelled")
                self.runs.append(dict(report, step=len(self.runs), run_dir=str(step_dir),
                                      valid_length=int(prepared.valid_length)))
                return output
            except BaseException as exc:
                report_path = step_dir / "run.json"
                report = {}
                try:
                    def finite_number(value):
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError("run.json must contain finite JSON numbers")
                        return number

                    candidate = json.loads(report_path.read_text(encoding="utf-8"),
                                           parse_constant=finite_number, parse_float=finite_number)
                    if not isinstance(candidate, dict):
                        raise ValueError("run.json must contain a JSON object")
                    report = candidate
                except FileNotFoundError:
                    # Failures before guest launch need not produce a report.
                    pass
                except Exception as report_error:
                    # Diagnostic failures must not replace the backend error,
                    # including cancellation/interrupts, or lose this attempt.
                    report["report_read_error"] = f"{type(report_error).__name__}: {report_error}"
                self.runs.append(dict(report, passed=False, status="FAIL", step=len(self.runs),
                                      run_dir=str(step_dir), valid_length=int(prepared.valid_length),
                                      error=f"{type(exc).__name__}: {exc}"))
                raise
        finally:
            self._lock.release()
