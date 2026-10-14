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

The reproduction target is Ubuntu 24.04 x86_64 and Python 3.12. Install the
`llm-deploy` extra as described in the [Linux reproduction contract](../../docs/llm-deploy-v1.0/LINUX_REPRODUCTION.md).
All Python dependencies and the CPU wheel source are declared in `pyproject.toml`.
Use that Python environment for the commands below, and record the checkout SHA
and actual dependency versions with each new run.

Verify/download
mode enforces NumPy 2.2.6, ONNX 1.18.0, ONNX Runtime 1.22.1 and protobuf 5.29.5;
export mode additionally enforces the complete fixed export stack, including
`torch==2.7.1+cpu` and `tokenizers==0.21.4`.

## Verify the fixed LFS assets

The complete ONNX graph and three external weight shards are tracked by Git LFS
under `assets/qwen3-0.6b/onnx` (about 2.24 GiB). Fetch them as described in the
Linux reproduction contract, then run the default offline verification:

```bash
python probes/w1_qwen3_export/run.py --output-dir output/qwen3-full-probe --threads 2
```

The committed `manifest.json` pins every file's size and SHA-256, the original
source revision, and the upstream release ZIP. `--model-dir` can select another
local directory. The explicit `download` mode remains available for legacy
Release acquisition, including `--archive path/to.zip`; ordinary reproduction
uses the files checked out by LFS. A corrupt directory is never overwritten.

Verification inspects every tensor's dtype and every external-data offset,
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
