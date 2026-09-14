"""Deprecated transitional alias for :mod:`scratchv.backend.regalloc_linear`.

The topic-17 linear-scan allocator converged into ``regalloc_linear``; this
module forwards every legacy import for one release cycle and will be
removed afterwards.  New code must import from
``scratchv.backend.regalloc_linear`` directly.
"""

from scratchv.backend.regalloc_linear import *  # noqa: F401,F403
from scratchv.backend.regalloc_linear import (  # noqa: F401
    LinearScanAllocator,
    LiveInterval,
    LsInstruction,
    RegAllocError,
    RegisterAliasError,
    SpillFallbackError,
    block_from_machine_instrs,
    machine_instrs_from_block,
    _DEFAULT_PHYS_REGS,
    _FP_REGS,
    _INT_REGS,
    _REG_NUMS,
)
