# W3 authenticated Qwen3 pretrained subgraphs

This offline gate exercises selected first-layer subgraphs of the pinned
`Qwen/Qwen3-0.6B` checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. It loads individual tensors with
`safetensors.safe_open`; it never constructs a full Torch model or downloads
anything. Checkpoint weights and source assets remain read-only.

Before any tensor is accessed, the gate streams SHA256 verification of the
checkpoint and all seven source assets against the existing W1 manifest.
The checkpoint SHA256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Each selected tensor also records its shape, original BF16 dtype, promoted
FP32 hash and byte count. No generated or random weights enter this gate.

Run from this checkout with the pinned CPU export environment:

```powershell
& 'D:/cyq/code/ScratchV/.venv-qwen-export/Scripts/python.exe' -X utf8 -B `
  probes/w3_qwen3_subgraphs/run.py `
  --source-dir D:/cyq/code/ScratchV/models/qwen3-source/c1899de289a04d12100db370d81485cdf75e47ca `
  --output-dir output/w3-qwen3-subgraphs-l256 `
  --seq-len 256
```

The output directory must not exist. Only L=256 can produce an official PASS.
`--seq-len 2..255` is a debug subset: even numerical success produces PARTIAL,
`passed=false`, and exit code 2. Formal success exits 0; every gate failure
exits 1. OMP, OpenBLAS, MKL, Torch and ORT are limited to one thread.

## Coverage and inputs

All 14 cases are mandatory, with ordinary output and diagnostic checkpoints
executed by ORT and the production ONNX parser/IR interpreter under
`none`, `basic` and `all` optimization levels.

| Cases | Real dimensions / learned parameters |
|---|---|
| Q/K/V/O projections | 1024→2048, 1024→1024, 1024→1024, 2048→1024 |
| Hidden/Q/K RMSNorm | Last dimensions 1024/128/128; authentic learned weights; epsilon 1e-6 |
| Q/K RoPE | 16/8 heads, full head dimension 128; every position 0..255; theta 1e6 |
| GQA full and key-padded | 16 query heads, 8 KV heads, 128 coordinates, L256 causal masking |
| GQA changed future/padding | Explicit perturbations verify causality and key-padding isolation |
| SwiGLU | 1024→3072→1024, authentic gate/up/down weights and intermediate checkpoints |

Activations are explicitly synthetic: NumPy PCG64 seed 20261004 generates
standard-normal hidden and O-projection input arrays. Q/K RMSNorm receives
the authentic Q/K projections of that hidden input. RoPE receives those
normalized Q/K arrays; GQA receives their rotations and the authentic V
projection. These preparatory activations are computed in Torch and saved
with shape/dtype/hash evidence. They are not claimed to come from token
embeddings or a complete decoder-layer forward. No residual, full-layer
composition, logits, generation, or language quality is established here.

## Independent reference and acceptance

Graphs are assembled from ONNX primitives. References separately use Torch
`F.linear`, `rsqrt`, direct half-pair rotations, `F.silu`, and per-query-head
attention. The independent attention loop maps query head `h` to KV head
`h // 2`, instead of reusing the graph's Expand/Reshape repetition. A manual
uniform-attention unit fixture additionally verifies this mapping and masks.

For every checkpoint, the gate requires matching shapes and FP32 dtypes,
finite values, and **max absolute error strictly less than 1e-4, rtol=0**.
It compares ORT against Torch, every IR level against ORT and Torch, and
ordinary outputs against diagnostic outputs. Padding queries are included.
Blocked attention must be zero, probability rows must sum to one, and both
causality and padding isolation must pass for Torch, ORT and all IR levels.

This W3 real-weight baseline does **not** change W2's strict 1e-5 threshold.
There is no full-dimension RISC-V execution claim for this gate.

## Evidence and memory

Every case saves `model.onnx`, `diagnostics.onnx`, `checkpoints.json`,
`inputs.npz`, `torch.npz`, `ort.npz`, three `ir_<level>.npz` files, and ordinary
outputs. Failure evidence remains in the new output directory, including
the first failing checkpoint/index, values and errors. The runner continues
other optimization levels and cases after a case failure when possible.

`report.json`, `report.md` and `report.html` contain source hashes, package
versions, authenticated weight identity, comparisons, timing and memory.
JSON is published atomically **last**, after the other required report files.
Report-write failure cannot leave a new PASS marker.

`process_peak_rss_bytes` is the process lifetime resident-memory high water
mark, including Torch, ORT and IR. It is not per-case incremental memory.
`memory_stats` separately describes interpreter retained arrays and backing
storage; it excludes kernel temporaries and allocator overhead. Ordinary
graphs and packed diagnostic graphs are labeled separately because packing
adds output storage. No peak-memory improvement is claimed.

Unit checks:

```powershell
& 'D:/cyq/code/ScratchV/.venv-qwen-export/Scripts/python.exe' -m pytest `
  tests/test_w3_qwen3_subgraphs.py -q
```

Unit-test safetensors and attention inputs are explicitly synthetic fixtures
for contract/error tests; they are never accepted as pretrained evidence.
