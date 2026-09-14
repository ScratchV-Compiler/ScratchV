from .constant_folding import ConstantFolder
from .dead_code import DeadCodeEliminator
from .peephole import IRPeepholeOptimizer
from .muladd_fusion import MulAddFusion
from .licm import LICM
from .loop_unroll import LoopUnroll, UnrollPlan

__all__ = [
    "ConstantFolder",
    "DeadCodeEliminator",
    "IRPeepholeOptimizer",
    "MulAddFusion",
    "LICM",
    "LoopUnroll",
    "UnrollPlan",
]
