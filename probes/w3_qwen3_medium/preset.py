"""Candidate medium preset: larger depth, hidden state, and feed-forward width.

This describes randomly initialized official Qwen3, not a pretrained model.
It intentionally retains W2's vocabulary, sequence length, and GQA geometry.
"""

from probes.w2_qwen3_small.model import MODEL_CONFIG, build_model

SEQUENCE_LENGTH = 256
MEDIUM_CONFIG = {
    **MODEL_CONFIG,
    "num_hidden_layers": 6,
    "hidden_size": 64,
    "intermediate_size": 192,
}
PRESET_NAME = "qwen3-medium-6l-h64-ffn192-q4-kv2-d16-random"


def build_medium_model(seed=0):
    return build_model(seed=seed, model_config=MEDIUM_CONFIG,
                       sequence_length=SEQUENCE_LENGTH)
