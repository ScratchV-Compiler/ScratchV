# W3 offline checkpoint comparison

Reproduction targets **Ubuntu 24.04 x86_64, Bash and Python 3.12**. Complete the [Linux environment and asset setup](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md) first. Run commands from the repository root in the same Bash session; the setup exports `SCRATCHV_PYTHON`, `SCRATCHV_CC` and `SCRATCHV_QEMU`.

This tool compares recorded NPZ tensors without executing a model. It reuses
`probes.w2_qwen3_small.diagnostics.compare_outputs`: exact shape and dtype,
finite numeric values, and strict `max_abs < atol` (default `1e-5`).

```bash
set -euo pipefail
"$SCRATCHV_PYTHON" probes/w3_layer_diff/run.py --actual output/w3/medium/traces/full_seed_0/ir_diagnostic_none.npz --reference output/w3/medium/traces/full_seed_0/reference.npz --schema output/w3/medium/trace_schema.json --out output/w3/layerdiff-full
```

`--out` must be new. The shared W3 publisher writes `report.json`, `report.md`
and `report.html` atomically per file, with JSON last. Publication errors fail
the command. No network, pickle loading or model execution is involved.

The schema is UTF-8 JSON with an explicit ordered checkpoint list:

```json
{"version":1,"checkpoints":[{"name":"layer_0.hidden","shape":[1,256,128],"dtype":"float32","layer":0,"checkpoint":"hidden","sequence_axis":1}]}
```

Only `name`, `shape`, and `dtype` are mandatory in each entry. `layer` is a
nonnegative integer; `checkpoint` is a local selector name; `sequence_axis` is
an optional valid axis or null. `--reference-schema` can provide a second schema
which must match exactly, including checkpoint order and metadata. Every NPZ
must have exactly the schema names, no missing/extra arrays or duplicate ZIP
members. Duplicate JSON keys/checkpoint names, object arrays, shape/type
disagreements and nonfinite values fail before comparison. ZIP expanded size
and schema logical size are limited by `--max-bytes` (default 2 GiB per NPZ);
schema size is limited to 8 MiB. NPY v1/v2 headers and payload sizes are checked
against the schema before NumPy allocates an array. This is validation of local evidence, not a
claim to safely process arbitrary hostile archives.

Repeat `--layer N` or `--checkpoint NAME` to select checkpoints. A checkpoint
selector matches an exact name or the optional local `checkpoint` field.
Layer and checkpoint filters intersect; unknown, duplicate and empty selections
fail. All evidence names, shapes, dtypes and finite values are still validated.
Every explicitly selected comparison reports **PARTIAL**, even when the
selection happens to cover every checkpoint; it cannot produce a full PASS.

Exit codes: `0` = all recorded checkpoints compared and passed; `1` = numeric
difference, invalid evidence or report publication failure; `2` = selected
checkpoints passed with partial coverage. Reports include ordered per-checkpoint
errors, the first divergence, the worst absolute-error element, source evidence
and artifact SHA-256 hashes. A full checkpoint PASS establishes no claim about
whole-model correctness, causal/padding tests or backend acceptance.

## IR memory observations

`IRInterpreter.run(..., collect_memory_stats=True)` populates
`ExecutionResult.memory_stats`; it is `None` by default. The optional observation
records input/initializer logical bytes, peak retained logical bytes, peak
deduplicated NumPy backing storage, and retained value counts. Inputs retain
their existing copy semantics; initializers count referenced bound globals
(including scalar global constants). Views count in logical bytes but share one
storage owner. ALLOCA buffers are counted. Final return copies are included in
peaks and separately reported, but excluded from retained counts.

These are instruction-boundary snapshots of values retained by the interpreter,
not RSS or total process allocation. They exclude caller arrays, transient
NumPy kernel buffers, Python object sizes and allocator overhead. The observer
does not release values, change kernels, alter control flow or add tensor copies.
Use the separate W3 process RSS metric for process-level memory evidence.
