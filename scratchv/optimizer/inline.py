"""Compatibility alias for the Topic 15 inliner module.

The implementation lives in :mod:`scratchv.optimizer.inliner` (the frozen
interface contract); this module re-exports the public names so both
``scratchv.optimizer.inliner`` and ``scratchv.optimizer.inline`` work.
"""

from .inliner import Inliner, InlinerConfig

__all__ = ["Inliner", "InlinerConfig"]
