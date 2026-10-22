"""Text -> tokenizer -> full-vocabulary two-layer model -> next token."""
import gc

import numpy as np
import onnx
import pytest

from probes.w2_runtime_model.run import (
    CASES, build_model, checked_inputs, check_onnx_contract, compare_logits, sample_checked,
)
from scratchv.runtime.llm_inputs import prepare_inputs


@pytest.fixture(scope="module")
def runtime_model_inputs(tokenizers) -> dict:
    adapter, fast, slow = tokenizers
    prepared = {}
    for name, text in CASES:
        ids = adapter.encode(text)
        assert ids == fast.encode(text) == slow.encode(text)
        inputs = prepare_inputs(ids, pad_token_id=adapter.pad_token_id, vocab_size=adapter.model_vocab_size)
        prepared[name] = len(ids), checked_inputs(ids, inputs, adapter.pad_token_id)
    return prepared


@pytest.fixture(scope="module")
def runtime_model_artifacts(tmp_path_factory, tokenizers, runtime_model_inputs, torch_cpu, ort_session):
    """Persist large references so tests can read them without retaining four logits arrays."""
    directory = tmp_path_factory.mktemp("qwen3-runtime-model")
    wrapper, config = build_model(tokenizers[0])
    assert config["num_hidden_layers"] == 2
    assert config["vocab_size"] == 151936
    for name, (_, feed) in runtime_model_inputs.items():
        args = tuple(torch_cpu.from_numpy(feed[key]) for key in ("input_ids", "attention_mask"))
        with torch_cpu.inference_mode():
            logits = wrapper(*args).numpy()
        np.save(directory / f"{name}.torch.npy", logits)
        del logits
    first_feed = runtime_model_inputs[CASES[0][0]][1]
    args = tuple(torch_cpu.from_numpy(first_feed[key]) for key in ("input_ids", "attention_mask"))
    with torch_cpu.inference_mode():
        exported = torch_cpu.onnx.export(
            wrapper, args, dynamo=True, opset_version=18,
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_shapes=None, optimize=False, fallback=False,
        )
        exported.save(str(directory / "model.onnx"), external_data=False)
    del exported, wrapper
    gc.collect()
    check_onnx_contract(onnx.load(directory / "model.onnx"))
    session = ort_session(directory / "model.onnx")
    for name, (_, feed) in runtime_model_inputs.items():
        logits = session.run(["logits"], feed)[0]
        np.save(directory / f"{name}.ort.npy", logits)
        del logits
    return directory


@pytest.mark.parametrize("name", [name for name, _ in CASES])
@pytest.mark.parametrize("engine", ["torch", "ort", "ir-none", "ir-all"])
def test_text_to_next_token(name, engine, runtime_model_artifacts, runtime_model_inputs,
                            tokenizers, run_ir):
    directory = runtime_model_artifacts
    length, feed = runtime_model_inputs[name]
    torch_reference = np.load(directory / f"{name}.torch.npy", mmap_mode="r")
    ort_reference = np.load(directory / f"{name}.ort.npy", mmap_mode="r")
    if engine.startswith("ir-"):
        actual = run_ir(directory / "model.onnx", feed, engine.removeprefix("ir-"))
    else:
        actual = torch_reference if engine == "torch" else ort_reference
    for reference in (torch_reference, ort_reference):
        comparison = compare_logits(actual, reference, length)
        assert comparison["passed"], comparison
    sample = sample_checked(actual, torch_reference, length, tokenizers[0], tokenizers[1:])
    assert sample["passed"], sample
    assert sample["position"] == length - 1
