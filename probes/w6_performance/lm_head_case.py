"""One shared Qwen3 embedding/LM Head weight, a fixed input, and ORT reference."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Program, Value


VOCAB = 151936
HIDDEN = 1024
ROWS = 256
SMALL_VOCAB = 257


@dataclass
class LMHeadCase:
    program: Program
    bindings: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray]
    reference: np.ndarray

    @property
    def weight(self):
        return self.bindings["weight"]

    @property
    def vocab(self):
        return self.weight.shape[0]


def load_case(asset_dir, vocab):
    """Select the first vocab weight rows and matching columns of the saved ORT output."""
    if not 2 <= vocab <= VOCAB:
        raise ValueError(f"vocab must be between 2 and {VOCAB}")
    asset_dir = Path(asset_dir)
    meta = json.loads((asset_dir / "metadata.json").read_text())
    if (meta["initializer_name"] != "model.lm_head.weight"
            or meta["left_shape"] != [1, ROWS, HIDDEN]
            or meta["initializer_shape"] != [VOCAB, HIDDEN]
            or meta["output_shape"] != [1, ROWS, VOCAB]):
        raise ValueError("LM Head metadata differs from the fixed Qwen3 case")
    weight_file = asset_dir / "weight.bin"
    if weight_file.stat().st_size != VOCAB * HIDDEN * 4:
        raise ValueError("LM Head weight file has the wrong length")
    weight = np.memmap(weight_file, dtype="<f4", mode="r", shape=(VOCAB, HIDDEN))[:vocab]
    x = np.load(asset_dir / "input_x.npy", mmap_mode="r", allow_pickle=False)
    ort = np.load(asset_dir / "ort.npy", mmap_mode="r", allow_pickle=False)
    if x.dtype != np.float32 or x.shape != (1, ROWS, HIDDEN):
        raise ValueError("Expected the saved FP32 input [1,256,1024]")
    if ort.dtype != np.float32 or ort.shape != (1, ROWS, VOCAB):
        raise ValueError("Expected saved ORT FP32 logits [1,256,151936]")

    ir = IRBuilder()
    hidden = Value("x", shape=x.shape)
    index = Value("index", dtype=DataType.INT64, shape=(1,))
    expected = Value("expected_embedding", shape=(1, HIDDEN))
    shared = Value("weight", shape=weight.shape)
    ir.new_function("main", [hidden, index, expected])
    ir.new_block("entry")
    ir.program.global_values.append(shared)
    # A data-dependent zero for this input keeps Gather alive before the timed head.
    gathered = ir.gather(shared, index, axis=0)
    head_input = ir.add(hidden, ir.sub(gathered, expected))
    ir.ret(ir.matmul(head_input, ir.transpose(shared, (1, 0))))
    selected = vocab - 1
    inputs = {
        "x": x, "index": np.array([selected], dtype=np.int64),
        "expected_embedding": weight[selected:selected + 1].copy(),
    }
    return LMHeadCase(ir.program, {"weight": weight}, inputs, ort[..., :vocab])
