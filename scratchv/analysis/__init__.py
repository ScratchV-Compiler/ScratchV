"""ScratchV analysis package: CFG builder, IR verifier, and analysis passes."""

from scratchv.analysis.cfg_builder import CFGBuilder, CFG
from scratchv.analysis.ir_verifier import IRVerifier, VerificationError

__all__ = ["CFGBuilder", "CFG", "IRVerifier", "VerificationError"]
