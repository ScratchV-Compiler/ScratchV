"""Complete local Qwen3: connectivity, real parsing and actual tensor bindings."""
import gc

import onnx
import pytest

from probes.w1_qwen3_export import run as artifacts
from probes.w2_qwen3_parse.audit import AuditedONNXParser, audit_parsed_model
from probes.w2_qwen3_parse.validation import audit_graph_structure


@pytest.fixture(scope="module")
def complete_graph(qwen3_model_path):
    return onnx.load(qwen3_model_path, load_external_data=False)


def test_full_onnx_structure(qwen3_model_dir, qwen3_model_files):
    artifacts.environment()
    structure = artifacts.inspect_model(qwen3_model_dir, qwen3_model_files)
    assert structure["opset"] == 18
    assert structure["external_tensors"] > 0


def test_all_28_layers_reach_logits(complete_graph):
    result = audit_graph_structure(complete_graph)
    assert result["passed"], result
    assert result["layer_count"] == 28
    assert result["live_node_count"] == len(complete_graph.graph.node)
    assert result["tied_embeddings"]
    assert [row["layer"] for row in result["layers"]] == list(range(28))
    assert all(row["passed"] for row in result["layers"])


@pytest.fixture(scope="module")
def parsed_qwen3(qwen3_model_path):
    parser = AuditedONNXParser()
    program = parser.parse(str(qwen3_model_path))
    yield parser, program
    # The full parser owns several GiB of arrays; do not retain it for other modules.
    parser.initializers.clear()
    del program, parser
    gc.collect()


def test_complete_model_ir(parsed_qwen3, complete_graph, assert_valid_ir):
    parser, program = parsed_qwen3
    assert_valid_ir(program, "full-qwen3-after-parse")
    assert len(parser.node_records) == len(complete_graph.graph.node)
    assert all(row["status"] == "translated" for row in parser.node_records)


def test_every_node_and_weight_binding(parsed_qwen3, complete_graph, qwen3_model_dir):
    parser, program = parsed_qwen3
    result = audit_parsed_model(complete_graph, parser, program, qwen3_model_dir)
    assert result["passed"], result
    assert result["node_count"] == len(complete_graph.graph.node)
    assert result["binding_count"] == len(parser.initializers)
    assert result["initializer_count"] == len(complete_graph.graph.initializer)
    assert len(result["tensor_bindings"]) == result["binding_count"]
    assert result["return"]["onnx_name"] == "logits"
