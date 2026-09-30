Qwen3-0.6B FP32 ONNX / W1 export probe

Unzip the entire qwen3-0.6b-onnx folder. Keep model.onnx and all three
weights-*.data files together with their original names. These are actual
model files, not Git LFS pointers. No Git LFS installation is needed.

Configuration: batch=1, sequence length=256, FP32, no KV cache, 28 layers.
ONNX opset 18, IR version 10. Tested with ONNX Runtime 1.22.1 on CPU.
input_ids: int64 [1,256]
attention_mask: float32 [1,1,256,256], additive causal + right-padding mask;
allowed positions=0, blocked positions=numpy.finfo(numpy.float32).min.
logits: float32 [1,256,151936]. Position IDs are fixed to 0..255.
For a right-padded input, select logits[0, valid_length-1, :] for the next
token; do not always select position 255. Tokenizer assets are not included;
use the tokenizer from the pinned official model revision below.

Validation scope: output shapes and finite values pass; valid-token
PyTorch/ORT comparison passes with atol=rtol=1e-4. The short-input case
fails the full-tensor comparison at padding positions (max error ~2.16e-3).
See verification.json for file SHA-256 hashes, exact results and versions.
This artifact does not imply complete ScratchV compiler support.

Source: Qwen/Qwen3-0.6B
Revision: c1899de289a04d12100db370d81485cdf75e47ca
https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca
License: included LICENSE (Apache-2.0).
