# W3 medium Qwen3 host numerical probe

This is a fixed candidate for the W3 six-layer experiment. The plan fixes six
layers; the dimensions below are the candidate chosen for this experiment,
not a claim that the plan mandated a specific hidden or feed-forward width.

| Property | W2 small default | W3 medium candidate |
|---|---:|---:|
| Decoder layers | 2 | 6 |
| Hidden width | 32 | 64 |
| FFN intermediate width | 96 | 192 |
| Query heads / KV heads | 4 / 2 | 4 / 2 |
| Head dimension | 16 | 16 |
| Vocabulary / sequence length | 128 / 256 | 128 / 256 |
| Batch / precision | 1 / FP32 | 1 / FP32 |
| Named checkpoints | 29 | 81 |

Weights are fixed-seed random initialization of the official
`transformers==4.51.3` `Qwen3ForCausalLM`. No weights or tokenizer are downloaded.
This gate measures architecture and numerical semantics. It does not establish
pretrained language quality, full Qwen3-0.6B execution, or RISC-V/QEMU correctness.
Attention is eager, positions are `0..255`, embeddings and LM head are tied, and
there is no KV cache. All numerical backends use one CPU thread.

## Run

Use Python 3.12 and the existing pinned
`requirements/qwen3-small-probe.txt` environment. From the W3 worktree:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
D:/cyq/code/ScratchV/.venv-qwen-export/Scripts/python.exe -m probes.w3_qwen3_medium.run --output-dir output/w3-medium --model-seed 0
```

The output directory must not already exist, even if it is empty. Preserve a
failed directory and choose a new path for a retry. The runner exits 0 only if the entire gate passes; numeric,
parsing, optimization, execution, and missing-reference failures exit 1.

## Acceptance and evidence

The gate uses the existing seven W2 inputs: two full random sequences, lengths
1, 17, and 255 with right padding, an altered future suffix, and altered masked
padding tokens. It exports the official model with named observations, then
produces ordinary logits-only and diagnostic packed-output ONNX graphs.

Both graphs run in ORT with graph optimization disabled and in ScratchV IR at
`none`, `basic`, and `all`. Every IR path compares against both PyTorch and its
own ORT graph. Ordinary graph, diagnostic graph, and every optimization level
record independent results. One failure cannot be masked by another passing
path. Independent NumPy GQA/probability/context and blocked-attention checks
also run against PyTorch.

The report uses uppercase `PASS` / `FAIL`. Its `coverage_complete` field requires
the exact seven named cases and lengths, both ORT graphs, all six IR variants
per case, and all 18 named/backend/graph isolation checks. Missing or duplicated
evidence cannot pass by reducing the set of checks being summarized.

All comparisons require matching shape and dtype, finite values, and strict
`max_abs < 1e-5`, with `rtol=0`. The threshold is fixed and is not a CLI option.
All positions are included, with valid-token and padding-query errors reported
separately. Causal and padding isolation compare the valid prefix at **every
checkpoint**, for PyTorch, both ORT graphs, and all six IR variants.

Checkpoint count is `3 + 13 * num_hidden_layers`: embedding, 13 observations per
layer, final norm, and logits. Shapes are derived from the actual model config,
including query/KV head count, head dimension, hidden width, vocabulary, and
sequence length. `probes/w2_qwen3_small/model.py` retains its two-layer defaults
and accepts explicit `model_config` overrides plus `sequence_length`.

The output contains:

- `model.onnx`, `checkpoints.onnx`, and `diagnostics.onnx`.
- `config.json`, `checkpoint_metadata.json`, packed `checkpoints.json`, and
  versioned named-NPZ `trace_schema.json` / `logits_schema.json`.
- `inputs_<case>.npz` and `traces/<case>/reference.npz` (PyTorch),
  `ort_normal.npz`, `ort_diagnostic.npz`, and
  `ir_<normal|diagnostic>_<none|basic|all>.npz`.
- `report.json`, `report.md`, and `report.html`, including config, environment,
  source/weight/artifact hashes, failures, timings, and memory observations.

Every named NPZ has exactly the keys described by its schema. The report's
`trace_artifacts` list supplies each path, schema, case/backend/graph, hashes,
and IR optimization level for an independent layer-diff tool. Actual numerical
failures retain their NPZ. Malformed IR diagnostic packs are instead saved as
`*_invalid_pack.npy` and explicitly fail the schema contract.

IR memory statistics describe the interpreter's retained arrays at instruction
boundaries. Logical bytes and deduplicated NumPy backing storage are distinct;
the interpreter's existing copy/retention behavior is unchanged. Temporary
NumPy buffers, allocator overhead, PyTorch, and ORT are outside that accounting.
`process_peak_rss_bytes` is separately reported as the whole process high-water
mark, including export and every backend; it is not an IR memory measurement.

## Regression tests

```powershell
D:/cyq/code/ScratchV/.venv-qwen-export/Scripts/python.exe -m pytest tests/test_qwen3_medium_model.py tests/test_qwen3_medium_probe.py tests/test_qwen3_small_model.py tests/test_qwen3_small_probe.py tests/test_qwen3_small_gate.py -q
```

Tests exercise all 81 medium shapes, a second configuration changing every
schema dimension and its real ONNX/ORT round trip, invalid configuration
rejection, failure isolation across all six IR paths, nonfinite/shape/value
faults, and the original small-model regressions. The command above includes
the real W2 gate and its injected-failure checks; it does not run QEMU.
