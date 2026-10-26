"""Opt-in host MatMul using an explicitly installed LP64 OpenBLAS library."""
from __future__ import annotations

import os
from pathlib import Path

from .external_matmul import ExternalMatMul


def openblas_matmul(
    *,
    include_dirs: tuple[str | os.PathLike[str], ...] = (),
    library_dirs: tuple[str | os.PathLike[str], ...] = (),
) -> ExternalMatMul:
    """Describe OpenBLAS without finding, downloading, or loading a library.

    Empty directories use the compiler's normal system search paths. Explicit
    library directories are resolved now and added to both the link search path
    and the runtime search path, so the generated shared library can be loaded
    after the build. Headers and libraries must belong to the same LP64 build.

    OpenBLAS may reorder FP32 operations. Callers must explicitly allow provider
    arithmetic when generating code. The adapter does not change OpenBLAS's
    process-wide threading settings or claim freestanding RISC-V compatibility.
    """
    if not isinstance(library_dirs, (tuple, list)):
        raise ValueError("OpenBLAS library_dirs must be a sequence")
    link_flags: list[str] = []
    for directory in library_dirs:
        if not isinstance(directory, (str, os.PathLike)) or not os.fspath(directory):
            raise ValueError("OpenBLAS library_dirs must contain nonempty paths")
        resolved = str(Path(directory).resolve())
        # Pass rpath as its own linker argument, including paths with commas.
        link_flags.extend((f"-L{resolved}", "-Xlinker", "-rpath", "-Xlinker", resolved))
    link_flags.append("-lopenblas")
    return ExternalMatMul(
        name="openblas",
        symbol="sv_openblas_matmul_f32_v1",
        sources=(str(Path(__file__).with_name("openblas_matmul.c").resolve()),),
        include_dirs=include_dirs,
        link_flags=tuple(link_flags),
        targets=("host",),
        max_dimension=2**31 - 1,
        alignment=4,
        numerics="provider_fp32",
    )
