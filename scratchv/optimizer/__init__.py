from .constant_folding import ConstantFolder
from .dead_code import DeadCodeEliminator
from .peephole import IRPeepholeOptimizer
from .muladd_fusion import MulAddFusion
from .licm import LICM
from .inliner import Inliner, InlinerConfig

__all__ = [
    "ConstantFolder",
    "DeadCodeEliminator",
    "IRPeepholeOptimizer",
    "MulAddFusion",
    "LICM",
    "Inliner",
    "InlinerConfig",
]
