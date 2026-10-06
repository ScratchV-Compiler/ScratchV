# W1: complete Qwen3 ONNX export gate

This gate checks the complete 28-layer Qwen3-0.6B export: FP32, batch 1,
sequence length 256, no KV cache, eager attention, ONNX opset 18 / IR 10.
It fails with a nonzero exit status for missing artifacts or dependencies,
unresolved LFS pointers, bad hashes, malformed external data, incorrect I/O,
or a failed/nonfinite ONNX Runtime execution. A previous `verification.json`
or a previous successful probe report cannot make a new run pass.

The model has two inputs: `input_ids` (`int64 [1,256]`) and `attention_mask`
(`float32 [1,1,256,256]`). Its only output is `logits`
(`float32 [1,256,151936]`). The mask combines causal attention and right
padding, using zero for allowed positions and the smallest finite FP32 value
for blocked positions. Short-input predictions use the last valid token.

## Environment

The published reproduction target is Ubuntu 24.04 x86_64 with Bash and Python 3.12. See the [Linux reproduction contract](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md). The existing `requirements/qwen3-small-probe.txt` CPU
environment also satisfies this gate. For a separate reproducible export
environment:

```bash
set -euo pipefail
python3.12 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/qwen3-export.txt
```

Keep this environment activated for all `python` commands below; the interpreter is `.venv-linux/bin/python`. Record the checkout SHA and actual dependency versions with each new run.

Verify/download
mode enforces NumPy 2.2.6, ONNX 1.18.0, ONNX Runtime 1.22.1 and protobuf 5.29.5;
export mode additionally enforces the complete fixed export stack, including
`torch==2.7.1+cpu` and `tokenizers==0.21.4`.

## Verify the fixed release

```bash
python probes/w1_qwen3_export/run.py --mode download \
  --model-dir output/qwen3-full-model \
  --output-dir output/qwen3-full-probe --threads 2
```

The release tag is `qwen3-0.6b-onnx-l256` in `ScratchV-Compiler/ScratchV`.
The 1,243,477,776-byte ZIP has SHA-256
`f186b7dce98df466160b36ab09818cc04195029f091233c83dd31484b28bb080`.
The committed `manifest.json` additionally pins the size and SHA-256 of the
graph and all three external weight shards (about 2.24 GiB in total).
Extraction rejects absolute/traversing paths, links and ambiguous model roots.
An existing model directory is reused only after every pinned file passes;
a corrupt existing directory is not overwritten. `--archive path/to.zip`
uses a previously downloaded ZIP with the same size/hash checks.

For an already available complete model directory, skip network access:

```bash
python probes/w1_qwen3_export/run.py --mode verify \
  --model-dir models/qwen3-0.6b-onnx \
  --output-dir output/qwen3-full-probe --threads 2
```

Both commands inspect every tensor's dtype and every external-data offset,
length and file reference, run the full ONNX checker, then execute fresh ORT
CPU inference for a short padded input and a full 256-token input. All logits
must have the fixed shape, FP32 dtype and finite values. ORT graph optimization
is disabled. These modes verify the existing export; they **do not rerun the
exporter or perform a new PyTorch/ORT numerical comparison**. They do not test
ScratchV's full-model execution or language quality.

## Reproduce the export

Obtain the official `Qwen/Qwen3-0.6B` Hugging Face snapshot at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. For example, in the pinned environment:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='output/qwen3-source')"
python probes/w1_qwen3_export/run.py --mode export \
  --source-dir output/qwen3-source \
  --model-dir output/qwen3-reexport \
  --output-dir output/qwen3-reexport-report --threads 2
```

This mode checks the pinned source weights, configuration and tokenizer
hashes, then delegates to the existing `export_qwen3_onnx.py`. It does not
implement a second exporter. The model directory must be empty; each heavy
reference/export/packing/verification stage runs in its own process. The
exporter's numerical gate compares valid-token logits with PyTorch using
`atol=rtol=1e-4`; the padded full-tensor comparison is reported separately and
can fail without failing the valid-token gate. This limitation is retained
explicitly in `export_validation`. The newly exported files are verified
against their fresh manifest, since serialized ONNX bytes need not match the
published release byte for byte. Allow substantially more memory/disk and time
than the download/verify path for a fresh export.

## Reports and lightweight regression tests

`--output-dir` must be outside the model directory. It receives `report.json`
and `report.md`, including mode, revision, package versions, file hashes,
SHA-256 fingerprints of this wrapper, the existing exporter and fixed manifest,
structure checks, fresh ORT cases, stage/run durations and peak process RSS
where the OS exposes it. RSS covers the verifier process, **not exporter
subprocesses**; unsupported platforms report `null` instead of a fabricated
measurement. Export mode also saves `export.log` and its reference artifacts.
Source fingerprints are included in failed reports as well as successful ones.
Upload the report directory as CI evidence, not the multi-gigabyte model.

```bash
python -m pytest tests/test_w1_qwen3_export.py -q
```

These tests use tiny generated ONNX/external-data artifacts and real ORT,
plus injected corruption and execution failures. They need no PyTorch,
checkpoint, model download or large tensor. The complete release verification
is a separate heavy CI job; passing these unit tests alone does not mean the
complete model was executed.
