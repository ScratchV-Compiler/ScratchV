from .constant_folding import ConstantFolder
from .dead_code import DeadCodeEliminator
from .peephole import PeepholeOptimizer
from .muladd_fusion import MulAddFusion
from .licm import LICM

__all__ = [
    "ConstantFolder",
    "DeadCodeEliminator",
    "PeepholeOptimizer",
    "MulAddFusion",
    "LICM",
]
