"""Opt-in MatMul using NMSIS-DSP source on the freestanding RV64GC target."""
from __future__ import annotations

import os
from pathlib import Path

from .external_matmul import ExternalMatMul


def nmsis_matmul(source_dir: str | os.PathLike[str], *, parallel_rows: bool = False) -> ExternalMatMul:
    """Describe an existing NMSIS checkout; do not download or vendor its code.

    source_dir is the checkout root containing NMSIS/DSP and NMSIS/Core.
    The adapter builds only the official FP32 matrix multiplication source.
    This integration is validated against NMSIS 1.6.0 on RV64GC/LP64D with no
    vector or packed-DSP extensions. It requires explicit provider arithmetic
    permission and supports matrix dimensions up to the library's uint16 limit.
    parallel_rows opts the scalar adapter into the existing joined worker team:
    each call has local descriptors, shares read-only inputs and writes separate
    output rows. The default remains serial, including in multi-hart graphs.
    """
    if not isinstance(source_dir, (str, os.PathLike)) or not os.fspath(source_dir):
        raise ValueError("NMSIS source_dir must be a nonempty checkout path")
    root = Path(source_dir).resolve() / "NMSIS"
    dsp = root / "DSP"
    core = root / "Core"
    source = dsp / "Source" / "MatrixFunctions" / "riscv_mat_mult_f32.c"
    required = (
        source,
        dsp / "Include" / "dsp" / "matrix_functions.h",
        core / "Include" / "nmsis_core.h",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"NMSIS checkout is missing required file: {path}")
    return ExternalMatMul(
        name="nmsis-dsp",
        symbol="sv_nmsis_matmul_f32_v1",
        sources=(str(Path(__file__).with_name("nmsis_matmul.c").resolve()), str(source)),
        include_dirs=(str(dsp / "Include"), str(core / "Include")),
        targets=("riscv64",),
        max_dimension=65535,
        alignment=4,
        numerics="provider_fp32",
        parallel_rows=parallel_rows,
    )
