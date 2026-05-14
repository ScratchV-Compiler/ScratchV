"""Simulator adapters for verifying generated RISC-V assembly.

Provides a TinyFive-based profiled machine that counts instructions,
allowing optimization passes to be measured quantitatively.
"""

from .tinyfive import ProfiledMachine, verify_assembly

__all__ = ["ProfiledMachine", "verify_assembly"]
